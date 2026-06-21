#!/usr/bin/env bash
# scripts/codex_peer_review.sh - run Codex CLI as a read-only peer reviewer.
#
# The mirror of scripts/claude_peer_review.sh: that script lets Codex or any
# operator invoke Claude headless; this one lets Claude or any operator invoke
# Codex headless, so the two agents can review each other without a human
# copy-pasting between them.
#
# Usage:
#   scripts/codex_peer_review.sh [base-ref] [--prompt REVIEW.md] [--extra "notes"]
#
# base-ref defaults to main and scopes the review diff (base-ref...HEAD).
#
# Auth: uses the existing ~/.codex login (ChatGPT or API key) — no env needed.
# Set CODEX_BIN=/path/to/codex if the CLI is not on this shell's PATH.
# Set CODEX_REVIEW_TIMEOUT_SECONDS to bound the run (default 900; 0 disables).
# Set CODEX_MODEL / CODEX_REASONING_EFFORT to override (default gpt-5.5 / medium).
#
# LEAN + BOUNDED: runs
#   codex exec --ignore-user-config --ignore-rules --ephemeral -s read-only -c model=... -c model_reasoning_effort=...
# `--ignore-user-config` skips ~/.codex/config.toml so the Codex desktop app's
# `notify` turn-end hook (which launches & leaks the Computer-Use client), its
# bundled plugins, the node_repl MCP server, and `xhigh` reasoning effort are
# NOT inherited — those made earlier runs leak processes and stall for hours.
# `--ignore-rules` additionally skips any user/project execpolicy `.rules` files,
# so out-of-band command-approval rules can't alter what this bounded read-only
# harness is allowed to run (this repo has none today, but a user-level rule
# could otherwise change approval behaviour the harness did not intend).
# `--ephemeral` avoids persisting session files. A pure-bash watchdog enforces
# the timeout (macOS has no gtimeout/timeout) and kills the whole process TREE,
# so a stalled run (and any child helpers) is TERM/KILLed at the deadline.
#
# REQUIRES a Codex-capable shell: even with the flags above, `codex exec` still
# initializes a local app-server + state DB under CODEX_HOME (~/.codex), so the
# invoking shell needs read/write there. This works from Claude Code's Bash
# tool; a more restricted sandbox (e.g. Codex's own read-only review sandbox)
# may fail to init; in that case run from a normal terminal.
#
# Sandbox: -s read-only — Codex may read files and run read-only shell (git
# diff, grep, cat) but cannot modify the repo. The full transcript (incl. the
# verdict) is written to reviews/codex/<date>-<branch>.md; only the tail is
# echoed to stdout, and latest.md is updated ONLY on a successful (rc=0) review.
# Only run on repositories/branches you trust.

set -euo pipefail

usage() {
  # Print the entire leading comment block (line 2 through the first
  # non-comment line), stripping the leading "# ". Self-adjusting, so the help
  # text never drifts from the header as the header grows.
  awk 'NR==1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$0"
}

prompt_file="REVIEW.md"
extra_notes=""
base_ref="main"
base_ref_set=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prompt) prompt_file="${2:?missing value for --prompt}"; shift 2 ;;
    --extra)  extra_notes="${2:?missing value for --extra}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; break ;;
    -*) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    *)
      if [[ "$base_ref_set" -eq 1 ]]; then
        echo "ERROR: only one base-ref positional argument is supported." >&2
        exit 2
      fi
      base_ref="$1"; base_ref_set=1; shift ;;
  esac
done
if [[ $# -gt 0 ]]; then
  if [[ "$base_ref_set" -eq 1 ]]; then
    echo "ERROR: unexpected base-ref after --; base-ref was already set." >&2
    usage >&2; exit 2
  fi
  if [[ $# -gt 1 ]]; then
    echo "ERROR: only one base-ref positional argument is supported after --." >&2
    usage >&2; exit 2
  fi
  base_ref="$1"; base_ref_set=1
fi

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

if [[ ! -f "$prompt_file" ]]; then
  echo "ERROR: review prompt not found: $prompt_file" >&2
  exit 2
fi

# Resolve the codex binary. Add the common nvm global-bin to PATH defensively
# so this works from non-login shells where nvm's PATH is not exported.
for d in "$HOME"/.nvm/versions/node/*/bin; do [[ -d "$d" ]] && PATH="$d:$PATH"; done
codex_bin="${CODEX_BIN:-codex}"
if ! codex_cmd="$(command -v "$codex_bin")"; then
  echo "ERROR: codex CLI not found: $codex_bin" >&2
  echo "Install with 'npm install -g @openai/codex' or set CODEX_BIN=/path/to/codex." >&2
  exit 127
fi

if ! "$codex_cmd" login status >/dev/null 2>&1; then
  echo "ERROR: codex is not logged in. Run 'codex login' once (reuses ~/.codex)." >&2
  exit 2
fi

if ! git rev-parse --verify --quiet "$base_ref" >/dev/null; then
  echo "ERROR: base ref not found: $base_ref" >&2
  exit 2
fi

current_branch="$(git branch --show-current || true)"
head_rev="$(git rev-parse --short HEAD)"
status_short="$(git status --short)"
diff_stat="$(git diff --stat "$base_ref...HEAD")"
diff_files="$(git diff --name-only "$base_ref...HEAD")"
diff_warning=""
if [[ -z "$diff_files" ]]; then
  diff_warning="WARNING: diff scope $base_ref...HEAD is empty."
  echo "$diff_warning" >&2
fi

review_timeout="${CODEX_REVIEW_TIMEOUT_SECONDS:-900}"
if ! [[ "$review_timeout" =~ ^[0-9]+$ ]]; then
  echo "ERROR: CODEX_REVIEW_TIMEOUT_SECONDS must be a non-negative integer." >&2
  exit 2
fi

mkdir -p reviews/codex
date_str="$(date -u +%Y-%m-%d)"
branch_safe="${current_branch:-detached}"; branch_safe="${branch_safe//\//_}"
out_file="reviews/codex/${date_str}-${branch_safe}.md"
latest_file="reviews/codex/latest.md"
tmp_file="${out_file}.tmp.$$"
timed_out_marker="${tmp_file}.timedout"
trap 'rm -f "$tmp_file" "$timed_out_marker"' EXIT

review_prompt="$(
  {
    printf 'You are a read-only adversarial peer reviewer. Review the diff %s...HEAD.\n\n' "$base_ref"
    printf 'Repository root: %s\n' "$repo_root"
    printf 'Current branch: %s\n' "${current_branch:-<detached>}"
    printf 'Base ref: %s\nHEAD: %s\n\n' "$base_ref" "$head_rev"
    printf 'git status --short:\n%s\n\n' "${status_short:-<clean>}"
    printf 'Diff stat (%s...HEAD):\n%s\n\n' "$base_ref" "${diff_stat:-<empty>}"
    printf 'Changed files:\n%s\n' "${diff_files:-<empty>}"
    [[ -n "$diff_warning" ]] && printf '\n%s\n' "$diff_warning"
    [[ -n "$extra_notes" ]] && printf '\nAdditional notes:\n%s\n' "$extra_notes"
    printf '\nUse read-only shell (git diff/show/log, grep, cat) to inspect the\n'
    printf 'changes. End with a clear DISPOSITION: sign-off, or a severity-ranked\n'
    printf 'findings list (Critical/High/Medium/Low) each with file:line and a fix.\n'
    printf '\n--- REVIEW INSTRUCTIONS ---\n'
    cat "$prompt_file"
  }
)"

# Run lean and headless. --ignore-user-config drops ~/.codex/config.toml, which
# on a machine with the Codex desktop app carries: a `notify` turn-end hook that
# launches (and leaks) the Computer-Use client after every turn; bundled plugins
# (computer-use, browser); a node_repl MCP server; and model_reasoning_effort =
# "xhigh". Skipping all of that is what keeps an automated review fast and
# non-leaking. Auth still resolves from CODEX_HOME. Model and effort are set
# explicitly here (overridable via CODEX_MODEL / CODEX_REASONING_EFFORT).
# --ephemeral: do not persist session files (leaner; fewer writes under CODEX_HOME).
# NOTE: even with these flags `codex exec` still initializes a local app-server +
# state DB under CODEX_HOME (~/.codex), so this needs a Codex-capable shell with
# read/write there. It works from Claude Code's Bash tool; a more restricted
# sandbox (e.g. Codex's own read-only review sandbox) may fail to init; run from
# a normal shell. (See the header note.)
codex_args=(exec --ignore-user-config --ignore-rules --ephemeral -s read-only --color never)
codex_args+=(-c "model=\"${CODEX_MODEL:-gpt-5.5}\"")
codex_args+=(-c "model_reasoning_effort=\"${CODEX_REASONING_EFFORT:-medium}\"")
codex_args+=("$review_prompt")

# Kill a process AND all its descendants (macOS has no setsid; codex may spawn
# child/grandchild helpers that would survive a parent-only kill). Post-order:
# signal children before the parent so nothing is reparented to init and lost.
_kill_tree() {
  local sig="$1" pid="$2" child
  for child in $(pgrep -P "$pid" 2>/dev/null); do
    _kill_tree "$sig" "$child"
  done
  kill "$sig" "$pid" 2>/dev/null || true
}

echo "Running Codex peer review (base=$base_ref, timeout=${review_timeout}s, lean) ..." >&2
rm -f "$timed_out_marker"
set +e
# Real wall-clock cap via a pure-bash watchdog (macOS has no gtimeout/timeout).
# Run codex in the background; a watchdog TERM/KILLs its whole process tree past
# the deadline and drops a marker so we report a clean timeout (124) instead of
# an opaque signal code.
#
# CRITICAL (2026-06-13 hang fix): the watchdog subshell MUST NOT inherit this
# script's stdout. `( sleep N; ... ) &` forks `sleep` as a child of the subshell;
# both inherit our fd 1. If the caller piped us into a reader (e.g. `... | tail`),
# that inherited fd keeps the pipe's write end open, so the reader never sees EOF
# and the WHOLE invocation appears to hang for the entire timeout window even
# though codex finished in seconds. Worse, the early-completion cancel below used
# to `kill` only the subshell, orphaning the `sleep` (reparented to init), which
# held the pipe open AND lingered as a stray process. Two defenses:
#   (1) redirect the subshell's std fds to /dev/null so it can never hold the pipe;
#   (2) on early completion, _kill_tree the watchdog so its `sleep` child dies too.
"$codex_cmd" "${codex_args[@]}" >"$tmp_file" 2>&1 &
codex_pid=$!
watchdog_pid=""
if [[ "$review_timeout" -gt 0 ]]; then
  (
    sleep "$review_timeout"
    if kill -0 "$codex_pid" 2>/dev/null; then
      : >"$timed_out_marker"
      _kill_tree -TERM "$codex_pid"
      sleep 5
      _kill_tree -KILL "$codex_pid"
    fi
  ) </dev/null >/dev/null 2>&1 &
  watchdog_pid=$!
fi
wait "$codex_pid"
rc=$?
if [[ -n "$watchdog_pid" ]]; then
  _kill_tree -TERM "$watchdog_pid"   # whole tree: kills the sleep child, not just the subshell
  wait "$watchdog_pid" 2>/dev/null
fi
[[ -f "$timed_out_marker" ]] && { rc=124; rm -f "$timed_out_marker"; }
set -e

if [[ "$rc" -eq 0 ]]; then
  mv "$tmp_file" "$out_file"
  cp "$out_file" "$latest_file"   # ONLY a successful review becomes authoritative
  echo "Codex review written to $out_file (and $latest_file)."
  echo "----- review tail (full review in $out_file) -----"
  tail -40 "$out_file"
  exit 0
fi

# Failure/timeout: preserve the (partial) output under a status-suffixed name for
# debugging, but do NOT touch latest.md — the previous good review stays
# authoritative, so an unattended agent never mistakes a failed/timed-out run for
# a current review.
fail_file="${out_file%.md}.FAILED-rc${rc}.md"
mv "$tmp_file" "$fail_file"
if [[ "$rc" -eq 124 ]]; then
  echo "ERROR: Codex review timed out after ${review_timeout}s; latest.md NOT updated. Partial output: $fail_file" >&2
else
  echo "ERROR: codex exec exited $rc; latest.md NOT updated. Output: $fail_file" >&2
fi
exit "$rc"
