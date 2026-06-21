#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/claude_peer_review.sh [base-ref] [--prompt REVIEW.md] [--extra "notes"]

Runs Claude Code as a read-only peer reviewer for the current Git repository.
Set CLAUDE_BIN=/path/to/claude if the Claude CLI is not on this shell's PATH.
Authentication uses CLAUDE_CODE_OAUTH_TOKEN when set, otherwise the script
looks for ~/.config/langchain_2026_agent_1/claude_code_oauth_token and exports
it only for the duration of the script. Override that path with
CLAUDE_CODE_OAUTH_TOKEN_FILE.
Set CLAUDE_REVIEW_TIMEOUT_SECONDS to control the review timeout (default 600;
0 disables the timeout).
Set CLAUDE_REVIEW_MAX_TURNS to control Claude Code's agentic turn cap (default
12; 0 omits --max-turns and uses the Claude CLI default).

base-ref defaults to main and is used to compute the review diff scope:
  git diff --stat base-ref...HEAD
  git diff --name-only base-ref...HEAD

The harness uses three layers:
  1. --tools removes mutation tools from Claude's available surface.
  2. --allowedTools grants only read actions without prompting.
  3. --permission-mode dontAsk auto-denies anything that would prompt.

These flags follow the Claude Code CLI docs; `claude --help` may not list every supported flag.
Official references:
  https://code.claude.com/docs/en/permission-modes
  https://code.claude.com/docs/en/cli-reference

The script must be run from inside the repository to review.
Only run this harness on repositories and branches whose filenames and Git metadata are trusted.
USAGE
}

prompt_file="REVIEW.md"
extra_notes=""
base_ref="main"
base_ref_set=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prompt)
      prompt_file="${2:?missing value for --prompt}"
      shift 2
      ;;
    --extra)
      extra_notes="${2:?missing value for --extra}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      if [[ "$base_ref_set" -eq 1 ]]; then
        echo "ERROR: only one base-ref positional argument is supported." >&2
        usage >&2
        exit 2
      fi
      base_ref="$1"
      base_ref_set=1
      shift
      ;;
  esac
done

if [[ $# -gt 0 ]]; then
  if [[ "$base_ref_set" -eq 1 ]]; then
    echo "ERROR: unexpected base-ref after --; base-ref was already set." >&2
    usage >&2
    exit 2
  fi
  if [[ $# -gt 1 ]]; then
    echo "ERROR: only one base-ref positional argument is supported after --." >&2
    usage >&2
    exit 2
  fi
  base_ref="$1"
  base_ref_set=1
fi

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

if [[ ! -f "$prompt_file" ]]; then
  echo "ERROR: review prompt not found: $prompt_file" >&2
  exit 2
fi

current_branch="$(git branch --show-current || true)"
head_rev="$(git rev-parse --short HEAD)"
status_short="$(git status --short)"

if ! git rev-parse --verify --quiet "$base_ref" >/dev/null; then
  echo "ERROR: base ref not found: $base_ref" >&2
  echo "Fetch the ref or pass a valid local base ref." >&2
  exit 2
fi

diff_stat="$(git diff --stat "$base_ref...HEAD")"
diff_files="$(git diff --name-only "$base_ref...HEAD")"
diff_warning=""
if [[ -z "$diff_files" ]]; then
  diff_warning="WARNING: diff scope $base_ref...HEAD is empty. This run will review repository context without changed-file scope."
  echo "$diff_warning" >&2
fi

claude_bin="${CLAUDE_BIN:-claude}"
if ! claude_cmd="$(command -v "$claude_bin")"; then
  echo "ERROR: claude CLI not found: $claude_bin" >&2
  echo "Install Claude Code, set CLAUDE_BIN=/path/to/claude, or run from an environment where 'claude' is available." >&2
  exit 127
fi

if [[ -z "${CLAUDE_CODE_OAUTH_TOKEN:-}" ]]; then
  token_file="${CLAUDE_CODE_OAUTH_TOKEN_FILE:-$HOME/.config/langchain_2026_agent_1/claude_code_oauth_token}"
  if [[ -f "$token_file" ]]; then
    token_mode="$(stat -f '%Lp' "$token_file" 2>/dev/null || stat -c '%a' "$token_file" 2>/dev/null || true)"
    if [[ -z "$token_mode" ]]; then
      echo "ERROR: Could not verify Claude OAuth token file permissions: $token_file" >&2
      exit 2
    fi
    if [[ "$token_mode" != "600" ]]; then
      echo "ERROR: Claude OAuth token file must be mode 600: $token_file" >&2
      echo "Run: chmod 600 \"$token_file\"" >&2
      exit 2
    fi
    export CLAUDE_CODE_OAUTH_TOKEN
    CLAUDE_CODE_OAUTH_TOKEN="$(tr -d '\r\n' < "$token_file")"
  fi
fi

review_timeout="${CLAUDE_REVIEW_TIMEOUT_SECONDS:-600}"
if ! [[ "$review_timeout" =~ ^[0-9]+$ ]]; then
  echo "ERROR: CLAUDE_REVIEW_TIMEOUT_SECONDS must be a non-negative integer." >&2
  exit 2
fi
review_max_turns="${CLAUDE_REVIEW_MAX_TURNS:-12}"
if ! [[ "$review_max_turns" =~ ^[0-9]+$ ]]; then
  echo "ERROR: CLAUDE_REVIEW_MAX_TURNS must be a non-negative integer." >&2
  exit 2
fi

mkdir -p reviews/claude
date_str="$(date -u +%Y-%m-%d)"
branch_safe="${current_branch:-detached}"
branch_safe="${branch_safe//\//_}"
out_file="reviews/claude/${date_str}-${branch_safe}.md"
latest_file="reviews/claude/latest.md"
tmp_file="${out_file}.tmp.$$"
cleanup() {
  rm -f "$tmp_file"
}
trap cleanup EXIT

TOOLS=(
  "Read"
  "Glob"
  "Grep"
  "Bash(git status:*)"
  "Bash(git diff:*)"
  "Bash(git show:*)"
  "Bash(git log:*)"
  "Bash(git ls-files:*)"
  "Bash(git rev-parse:*)"
)

ALLOWED_TOOLS=(
  "${TOOLS[@]}"
)

review_prompt="$(
  {
    printf 'Repository root: %s\n' "$repo_root"
    printf 'Current branch: %s\n' "${current_branch:-<detached>}"
    printf 'Base ref: %s\n' "$base_ref"
    printf 'HEAD: %s\n\n' "$head_rev"
    printf 'Current git status --short:\n'
    if [[ -n "$status_short" ]]; then
      printf '%s\n' "$status_short"
    else
      printf '<clean>\n'
    fi
    printf '\nDiff stat for %s...HEAD:\n' "$base_ref"
    if [[ -n "$diff_stat" ]]; then
      printf '%s\n' "$diff_stat"
    else
      printf '<empty or unavailable>\n'
    fi
    printf '\nChanged files for %s...HEAD:\n' "$base_ref"
    if [[ -n "$diff_files" ]]; then
      printf '%s\n' "$diff_files"
    else
      printf '<empty>\n'
    fi
    if [[ -n "$diff_warning" ]]; then
      printf '\nDiff-scope warning:\n%s\n' "$diff_warning"
    fi
    if [[ -n "$extra_notes" ]]; then
      printf '\nAdditional user notes:\n%s\n' "$extra_notes"
    fi
    printf '\n--- REVIEW INSTRUCTIONS ---\n'
    cat "$prompt_file"
  }
)"

CLAUDE_ARGS=(
  -p "$review_prompt"
  --permission-mode dontAsk
  --tools "${TOOLS[@]}"
  --allowedTools "${ALLOWED_TOOLS[@]}"
)

if [[ "$review_max_turns" -gt 0 ]]; then
  CLAUDE_ARGS+=(--max-turns "$review_max_turns")
fi

set +e
if [[ "$review_timeout" -gt 0 ]]; then
  perl -e '
    $timeout = shift;
    $pid = fork();
    die "fork failed: $!\n" unless defined $pid;
    if ($pid == 0) {
      exec @ARGV or die "exec failed: $!\n";
    }
    $SIG{ALRM} = sub {
      kill "TERM", $pid;
      waitpid($pid, 0);
      exit 124;
    };
    alarm $timeout;
    waitpid($pid, 0);
    $status = $?;
    exit(($status & 127) ? 128 + ($status & 127) : ($status >> 8));
  ' \
    "$review_timeout" "$claude_cmd" "${CLAUDE_ARGS[@]}" \
    | tee "$tmp_file"
else
  "$claude_cmd" "${CLAUDE_ARGS[@]}" | tee "$tmp_file"
fi
pipe_status=("${PIPESTATUS[@]}")
claude_status=${pipe_status[0]}
tee_status=${pipe_status[1]}
set -e

if [[ "$claude_status" -ne 0 ]]; then
  if [[ "$claude_status" -eq 124 && "$review_timeout" -gt 0 ]]; then
    echo "ERROR: Claude review timed out after ${review_timeout}s; latest review was not updated." >&2
    exit "$claude_status"
  fi
  echo "ERROR: Claude review command failed with exit code $claude_status; latest review was not updated." >&2
  exit "$claude_status"
fi

if [[ "$tee_status" -ne 0 ]]; then
  echo "ERROR: Failed to write Claude review output; latest review was not updated." >&2
  exit "$tee_status"
fi

if [[ ! -s "$tmp_file" ]]; then
  echo "ERROR: Claude review output was empty; latest review was not updated." >&2
  exit 1
fi

mv "$tmp_file" "$out_file"
cp "$out_file" "$latest_file"
trap - EXIT
printf '\nReview written to %s\nLatest review copied to %s\n' "$out_file" "$latest_file"
