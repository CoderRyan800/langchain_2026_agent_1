# User Guide

This guide covers day-to-day use of both agents in this project.

---

## Contents

1. [Initial Setup](#1-initial-setup)
2. [Bob — General Purpose Agent](#2-bob--general-purpose-agent)
3. [Litter Box Agent — First-Time Setup](#3-litter-box-agent--first-time-setup)
4. [Litter Box Agent — Daily Workflow](#4-litter-box-agent--daily-workflow)
5. [Litter Box Agent — Querying Records](#5-litter-box-agent--querying-records)
6. [Sensor Integration](#6-sensor-integration)
7. [Understanding Health Alerts](#7-understanding-health-alerts)
8. [Data and Storage](#8-data-and-storage)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Initial Setup

### API keys

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
```

Both keys are required to use Bob. The litter box agent only needs `OPENAI_API_KEY`.

### Install dependencies

**Conda (recommended):**
```bash
conda create -n langchain_env_2026_1 python=3.11 -y
conda activate langchain_env_2026_1
pip install -r requirements.txt
```

**Python venv:**
```bash
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### First-run note — CLIP model download

The litter box agent uses a local vision model (`clip-ViT-B-32`) for cat identification. On the first run it will download approximately 350 MB from HuggingFace and cache it. Subsequent starts load the cached model in about one second. No API key is required for CLIP — it runs entirely on your machine.

---

## 2. Bob — General Purpose Agent

Bob is a conversational assistant you can use for web searches, answering questions, and analysing images or audio files you upload.

### Starting Bob

```bash
python src/basic_agent.py
```

### Talking to Bob

Type anything at the `You:` prompt and press Enter. Bob remembers the full conversation across restarts — his history is stored in `agent_memory.db`.

```
You: What's the weather like in Tokyo today?
You: Summarise the top AI news from this week
You: What did we talk about last time?
```

### Uploading files

Use `/UPLOAD` followed by an absolute path to share an image or audio file:

```
You: /UPLOAD /Users/yourname/Desktop/xray.jpg
You: What do you see in that image?

You: /UPLOAD /tmp/voice_memo.mp3
You: Can you transcribe that?
```

**Supported image formats:** `.jpg` `.jpeg` `.png` `.gif` `.webp`

**Supported audio formats:** `.mp3` `.wav` `.ogg` `.m4a` `.flac` `.opus`

> **Audio note:** Audio input requires changing `MODEL = "gpt-4o-audio-preview"` at the top of `src/basic_agent.py`. The default `gpt-4o` supports images only.

### Quitting

```
You: /STOP
```

### Bob and the litter box data

Bob can query the litter box database using plain English. Since `data/litterbox.db` is a standard SQLite file, you can ask Bob to run read-only SQL queries against it:

```
You: How many times did Whiskers use the litter box this week?
You: Show me all visits that were flagged as anomalous in the last 30 days
You: Which cat visited the most in March?
```

---

## 3. Litter Box Agent — First-Time Setup

Before the automated monitoring system can identify cats, you must register at least one reference photo per cat.

### Starting the agent in interactive mode

```bash
python src/litterbox_agent.py
```

### Registering a cat

Use `/UPLOAD` followed by the path to a clear photo of your cat:

```
You: /UPLOAD /Users/yourname/Photos/whiskers_sofa.jpg
Assistant: I want to register this cat reference image. What is this cat's name?
You: Whiskers
Assistant: Registered reference image #1 for 'Whiskers'.
           Stored at images/cats/whiskers/001_3a2f1c.jpg
```

You can register the name in the same message if you prefer:

```
You: /UPLOAD /path/to/photo.jpg — this is my cat Marmalade
```

### Registering multiple photos per cat

More reference images improve identification accuracy, especially across different lighting conditions and angles. Register as many as you have:

```
You: /UPLOAD /path/to/whiskers_window.jpg
Assistant: What is this cat's name?
You: Whiskers
Assistant: Registered reference image #2 for 'Whiskers'.
```

> **Tip:** Register photos taken from the angle and distance your litter box camera will see. Photos taken at ground level looking slightly upward will give the best identification performance.

### Checking your registered cats

```
You: Which cats are registered?
Assistant: Registered cats:
             Marmalade: 3 reference image(s) (registered 2026-03-14)
             Whiskers: 2 reference image(s) (registered 2026-03-14)
```

---

## 4. Litter Box Agent — Daily Workflow

Once cats are registered and sensors are connected, the system runs automatically. Your day-to-day interaction will mostly be **confirming tentative identities** and **reviewing health alerts**.

### Reviewing unconfirmed visits

```
You: Show me visits that need confirmation
Assistant: 4 unconfirmed visit(s):
             #12: ~Whiskers (sim=0.94) at 2026-03-14 07:22:11
             #11: ~Marmalade (sim=0.88) at 2026-03-14 06:55:02
             #10: Unknown at 2026-03-13 23:10:44
             #9:  ~Whiskers (sim=0.91) ⚠️ at 2026-03-13 20:15:33
```

The `~` prefix means tentative (not yet confirmed). The similarity score shows how confident the system was. A `⚠️` means that visit has a health flag.

### Confirming an identity

```
You: Visit 12 is confirmed as Whiskers
Assistant: Visit #12 confirmed: cat is 'Whiskers'.

You: Confirm visit 11 as Marmalade
Assistant: Visit #11 confirmed: cat is 'Marmalade'.
```

### Handling an Unknown visit

When the system could not identify the cat:

```
You: Show me the images for visit 10
Assistant: Images for visit #10:
             Entry: images/visits/2026-03-13/a4b5c6d7_entry.jpg
             Exit:  images/visits/2026-03-13/a4b5c6d7_exit.jpg

You: That's Whiskers — confirm visit 10 as Whiskers
Assistant: Visit #10 confirmed: cat is 'Whiskers'.
```

You can open the image files directly from the paths the agent returns to inspect them visually before confirming.

---

## 5. Litter Box Agent — Querying Records

All queries work in natural language in the interactive session.

### By date

```
You: Show me all visits from today
You: What happened on 14th March?
You: Show me visits from 2026-03-13
```

### By cat

```
You: How many times did Whiskers visit this week?
You: Show me all of Marmalade's visits
You: When did Whiskers last use the litter box?
```

### Health flags

```
You: Show me all flagged health events
You: Are there any anomalous visits this month?
```

### Images

```
You: Show me the images for visit 9
```

The agent returns the relative file paths — open them in any image viewer.

---

## 6. Sensor Integration

### How sensor events are triggered

The sensor system calls the agent directly from the command line when a camera detects motion at the litter box:

```bash
# Cat detected entering
python src/litterbox_agent.py --event entry --image images/captures/entry_001.jpg

# Cat detected exiting
python src/litterbox_agent.py --event exit --image images/captures/exit_001.jpg
```

The agent processes the image, runs the identification pipeline, writes the visit record to the database, and exits. No human input is required.

### Recommended directory layout for sensor captures

Save incoming sensor images to `images/captures/` using any filename convention your camera system produces. The agent copies the image to permanent storage under `images/visits/YYYY-MM-DD/` with a UUID filename, so the original capture filename does not matter.

### Exit event association

When an exit event arrives, the agent automatically associates it with the **most recent open visit** (a visit with an entry image but no exit image yet). This means:

- The entry event must always be processed before the exit event.
- If two cats use the litter box simultaneously, the system will associate the exit with whichever entry was most recent — review these cases manually.

### Orphan exit records

If an exit event is received with no corresponding open visit (e.g. the entry sensor misfired, or the agent was not running when the cat entered), an **orphan exit record** is created:

```
⚠️  WARNING: No open visit found. Orphan exit record #18 created — human review required.
    Exit image stored at: images/visits/2026-03-14/c3d4e5f6_exit.jpg
```

Orphan records are visible when querying visits and are flagged for review. They can be confirmed manually just like regular visits once you have identified the cat from the exit image.

---

## 7. Understanding Health Alerts

### What is analysed

After every exit event, GPT-4o compares the entry and exit images of the litter box. It looks for visual differences that may indicate a health concern:

- Blood in urine (pink, red, or dark discolouration of urine clumps)
- Blood in stool
- Unusual stool colour, consistency, or quantity
- Evidence of diarrhoea or mucus
- Abnormal deposits or clumping patterns
- Any other unexpected visual changes

### What a flagged alert looks like

```
Health: ⚠️  ANOMALY FLAGGED — veterinary review recommended

CONCERNS_PRESENT: yes
DESCRIPTION: The exit image shows a pink-tinged clump in the lower left
that was not present in the entry image.
OWNER_SUMMARY: There appears to be discolouration consistent with blood in
the urine. This warrants prompt veterinary attention.

⚠️ This analysis is preliminary and must be reviewed by a licensed
veterinarian before any medical decisions are made.
```

### Viewing all flagged visits

```
You: Show me all anomalous visits
```

### Critical disclaimer

**This system is a monitoring aid, not a medical device.** The AI analysis is:

- **Preliminary** — it cannot replace a veterinarian's examination
- **Not always accurate** — lighting, camera angle, and litter type all affect image quality and therefore the reliability of the analysis
- **Not a substitute for regular vet check-ups**

If a visit is flagged as anomalous, or if you observe any change in your cat's behaviour or litter box habits, consult a licensed veterinarian promptly.

---

## 8. Data and Storage

### Where everything lives

| What | Where | Gitignored |
|---|---|---|
| Cat reference photos | `images/cats/<catname>/` | Yes |
| Visit entry/exit images | `images/visits/YYYY-MM-DD/` | Yes |
| Sensor capture drop zone | `images/captures/` | Yes |
| Cat and visit metadata | `data/litterbox.db` | Yes |
| CLIP vector index | `data/chroma/` | Yes |
| Litter box agent conversation history | `data/agent_litterbox_memory.db` | Yes |
| Bob's conversation history | `agent_memory.db` | Yes |

### Backing up your data

All persistent state lives in the `data/` and `images/` directories and the `agent_memory.db` file. Back these up regularly — they are not version-controlled.

### Starting fresh

To wipe all litter box data and start over:

```bash
rm -rf data/ images/
```

Bob's conversation history is separate:

```bash
rm agent_memory.db
```

### Starting a new conversation thread

All history is keyed by `thread_id`. To start a fresh conversation without deleting the database, change the `thread_id` in the relevant agent file:

```python
# In src/basic_agent.py or src/litterbox_agent.py
config = {"configurable": {"thread_id": "2"}}   # increment for a new thread
```

### Future migration to Postgres

The SQLite checkpointer (`SqliteSaver`) is a drop-in development store. When you need multi-user access or larger history, swap it for LangGraph's `PostgresSaver` — the interface is identical and all stored history can be migrated.

---

## 9. Troubleshooting

### "No module named 'langgraph.checkpoint.sqlite'"

Install the separate checkpoint package:
```bash
pip install langgraph-checkpoint-sqlite
```

### CLIP model warnings on startup

```
Key vision_model.embeddings.position_ids | UNEXPECTED
```

These are harmless. The `sentence-transformers` library reports unused keys when loading a CLIP model from the HuggingFace cache. They do not affect embedding quality or identification accuracy.

### GPT-4o refuses to analyse images

The health analysis prompt includes veterinary terminology that can occasionally trigger OpenAI's content safety filter, particularly when images do not clearly resemble a litter box (e.g. test images, very dark captures, extreme close-ups). In production this is rare with genuine camera footage. If it occurs regularly, check that your camera is correctly framed to show the full litter box interior.

### Cat identified as "Unknown"

This means either:
1. The cat's photo was not registered — run `/UPLOAD` with a clear reference photo
2. The similarity score was below the 0.82 threshold — add more reference images taken from the camera's viewing angle
3. GPT-4o did not confirm the CLIP match — this can happen in poor lighting. Improve camera lighting or add reference images taken in similar conditions.

### Orphan exit records accumulating

This indicates the entry sensor is misfiring or the agent was not running when the cat entered. Check:
- That the entry sensor has a reliable power supply and network connection
- That the agent process is running continuously (consider a `systemd` service or `launchd` plist on macOS)
- The camera trigger threshold — if it is too sensitive it may miss slow-moving entries

### Agent loses conversation history

History is tied to `thread_id`. If the thread ID in the config was changed or the database was deleted, prior history will not be available. Check `data/agent_litterbox_memory.db` exists and is non-empty.
