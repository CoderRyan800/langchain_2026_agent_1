# Quick Start — for Cat Owners

This guide is for cat owners who want to use the Litter Box Monitor to keep
an eye on their cat's health. **No technical background assumed.** If you
can install a phone app, you can run this.

> **Important upfront**: this system is a screening tool. It can tell you
> "something looks unusual, please look closer" but it cannot diagnose
> illness. Every concerning result must be reviewed by a licensed
> veterinarian. Treat it like a smoke alarm — useful warning, not a
> medical opinion.

---

## What is it?

The Litter Box Monitor watches photos and (optionally) sensor readings from
your cat's litter box and tells you when something looks different from
your cat's normal pattern. It does three useful things:

1. **Identifies which cat just used the box** (in multi-cat households) by
   comparing a photo of the visitor to reference photos you registered.
2. **Spots unusual single visits** — for example, a visit where the
   ammonia or methane reading is way higher than usual for that cat, or
   the visual analysis spots something concerning.
3. **Spots slow drifts over weeks or months** — for example, gradual
   weight loss that you might not notice day-to-day, or a cat that has
   started leaving less waste than usual (which can mean constipation).

You talk to it in plain English through a chat interface. It does the
math under the hood and gives you human-readable answers.

---

## What you need

| | |
|---|---|
| Computer | A Mac, Windows, or Linux machine. A small Linux box like a Raspberry Pi works for permanent installation. |
| Internet | The vision and chat features call OpenAI's servers. You don't need a fast connection, but you need one. |
| Python | Version 3.11 or newer. If you don't already have it, download from python.org. |
| OpenAI API key | Sign up at platform.openai.com and create an API key. The agent uses GPT-4o, which is roughly 1–3 cents per litter box visit analysed. A heavy day might cost a dollar; a typical month is $5–20. |
| Camera (optional) | Any camera that can save a JPEG of the inside of the litter box on cat entry and exit. A cheap USB webcam pointed at the box works. |
| Sensors (optional) | A weight scale under the box and gas sensors (ammonia + methane) make the system much more useful. None are required — the system works with photos alone, just with less information. |

You can start without sensors and add them later. The system gracefully
handles missing data.

---

## Getting it running

### Step 1 — Install

Open a terminal (the black/white text window on your computer) and type:

```bash
git clone https://github.com/CoderRyan800/langchain_2026_agent_1.git
cd langchain_2026_agent_1
conda create -n litterbox python=3.11 -y
conda activate litterbox
pip install -r requirements.txt
```

Conda is a free tool that keeps Python projects from interfering with each
other. If you don't have conda, install Miniconda from
https://docs.conda.io/en/latest/miniconda.html first.

### Step 2 — Add your API key

Create a file called `.env` in the project folder, with one line:

```
OPENAI_API_KEY=your-key-from-openai-here
```

(Replace the placeholder with your actual key. Keep this file private —
treat it like a password.)

### Step 3 — Start the agent

```bash
python src/litterbox_agent.py
```

You'll see:

```
Litter Box Agent ready.
Thread: interactive-1746843921
Commands:
  /UPLOAD <filepath>  — register a reference photo for a cat
  /NEW                — start a fresh conversation thread
  /STOP               — quit

You: 
```

That's it — you're talking to the agent.

---

## Registering your cats (do this once per cat)

Take a clear photo of each cat. A well-lit photo of the cat looking at the
camera works best. Save the file somewhere you can find it.

In the agent, type:

```
/UPLOAD /path/to/luna.jpg
```

It'll ask for the cat's name. Type the name. Then it'll ask when you got
the cat (so it can scan old unidentified visits and try to match them
retroactively). Answer in YYYY-MM-DD format, or just give your best guess.

You can register multiple cats by repeating this. Adding more reference
photos of the same cat improves identification accuracy — register a few
for each cat if you can.

To see who's registered:

```
You: List all the cats.
```

---

## What you'll do day to day

**Most of the time you do nothing.** The system runs in the background.
When the camera triggers a visit, the agent records it, identifies the cat,
and analyses the photos and sensor readings. If something looks unusual,
the visit is flagged.

**Once a day or so**, you might check what's been happening:

```
You: What's been happening with the cats?
You: Show me any flagged visits from this week.
You: How many times did Luna use the box yesterday?
```

The agent will pull the data and summarise.

**When you want a deeper look**, ask focused questions:

```
You: Tell me about visit 142.
You: Show me a chart of Luna's weight over the last 90 days.
You: Has Luna lost weight lately?
You: Could Anna be constipated?
```

---

## Understanding the alarms

The system uses four alarm levels for every channel it watches:

| Level | What it means | What to do |
|---|---|---|
| **normal** | This visit (or this trend) looks like the cat's usual pattern. | Nothing. |
| **mild** | Slightly outside the cat's typical range. About 1 in 20 normal visits will trip "mild" purely by chance. | Make a mental note. Watch for a pattern. |
| **significant** | Clearly outside the cat's typical range. About 1 in 300 normal visits trip this. | Look at it. Ask the agent for the details. Consider calling the vet if this becomes a pattern. |
| **severe** | Way outside the cat's typical range. Practically never happens by chance. | Treat as something to act on. Call the vet. |

The system errs toward "screening" — it would rather show you a few
mild flags than miss a real problem. A mild flag here and there is normal
and not a reason for alarm. Repeated significant or any severe flag
deserves attention.

### Example: per-visit alarm

```
You: Show me anomalous visits.
Assistant:
  3 anomalous visit(s):

    Visit #142 — Luna (confirmed) at 2026-05-09T08:14:22
      Gas anomaly: tier=severe, NH₃ z=+6.55, CH₄ z=+0.67
      GPT-4o: Both images appear normal; ammonia reading is unusually high
      compared to Luna's history. Recommend veterinary evaluation if this
      pattern persists.
```

In plain English: Luna had a visit where the ammonia reading was about 6.5
standard deviations above her usual reading. That's "almost never happens
by chance" territory. The visual analysis didn't flag anything, but the
gas reading alone is worth showing to a vet.

### Example: long-term trend alarm

```
You: Show me anomalous visits.
Assistant:
  No per-visit anomalies on record.

  Long-term trend alarms (1 cat(s) currently flagged):

    Luna — overall tier: significant
      • weight [significant]  (z=-7.32, pct=-6.4%)
```

In plain English: Luna's average weight over the last two weeks is about
6.4% below her average from the prior 75 days. That's a clinically
meaningful drop. The system is recommending you investigate — gradual
weight loss can mean kidney disease, hyperthyroidism, dental pain, or
just a diet change. Worth a vet visit.

### When to call the vet

There's no algorithm that can replace a vet. But here's a rough guide:

- **One mild flag, no pattern** → probably nothing.
- **Repeated mild flags in the same channel for the same cat** → ask the
  agent for the trend. If the trend is also flagged, call the vet.
- **Any significant flag on a single visit** → look at the details, save
  them. If you see another in the next few days, call the vet.
- **Any severe flag** → call the vet. Even if the cat looks fine to you.
- **A trend alarm at significant or severe** → call the vet. Slow drifts
  catch problems before symptoms show.

The OpenAI vision analysis always ends with a disclaimer reminding you
to consult a vet. That's not boilerplate — it's there because no
automated tool should be treated as a diagnosis.

---

## Useful things to ask

The agent understands plain English. Here are some common requests:

**Browsing recent activity**

- "What's been happening today?"
- "Show me visits from yesterday."
- "Show me Luna's visits this week as a table."
- "How many times has Anna used the box this month?"

**Understanding a specific visit**

- "Tell me about visit 142."
- "Why was visit 142 flagged?"
- "What were the readings on visit 99?"

**Health-focused queries**

- "Show me any anomalous visits."
- "Are any of the cats trending differently lately?"
- "Has Luna lost weight?"
- "Could Anna be constipated?"
- "Have any cats had high ammonia readings recently?"

**Charts**

- "Plot Luna's weight history."
- "Show me a chart of Anna's gas readings over the last 90 days."

The agent will save chart files (HTML format) that you open in any web
browser to see the actual graph.

**Confirming who a visit belongs to**

If the system isn't sure which cat used the box ("tentative" identity),
you can confirm:

- "Confirm visit 142 as Luna."
- "Show me unconfirmed visits so I can review them."

---

## Things that go wrong, and what to do

**The agent gives weird short answers and ignores my question.**

Type `/NEW` to start a fresh conversation. The previous thread may have
old context confusing it.

**It says "no anomalous visits and no trend alarms" but the cat is clearly
sick.**

The system is a screening tool, not a diagnostic. It can miss things —
especially health problems that don't show up in weight, gas, or visual
appearance of waste. Always trust your own observation of the cat over
any automated tool.

**It keeps printing weird "LangSmith 429" errors.**

That's tracing telemetry hitting a usage cap. It's harmless but noisy.
Open the `.env` file and add these two lines:

```
LANGSMITH_TRACING=false
LANGCHAIN_TRACING_V2=false
```

Save and restart the agent. The errors stop.

**OpenAI says I've used too many tokens / costs are higher than expected.**

Each visit that triggers the camera-based health check costs about 1–3
cents in OpenAI fees (mostly for GPT-4o vision). A box that triggers 50
times a day will add up to $20–30/month. Sensor-only monitoring (no
camera analysis) costs almost nothing.

**The vision analysis says "I'm sorry, I can't analyze these images."**

OpenAI's content filter occasionally false-positives on litter box
images. The system has a safeguard that ignores these refusals and falls
back to the data-driven gas anomaly detector. The visit is still scored;
the visual side just didn't contribute.

**A visit says "Unknown" and won't identify the cat.**

The CLIP model couldn't match the photo to any registered cat with high
enough confidence. Causes:

- The photo is dark, blurry, or only shows part of the cat.
- The cat's appearance changed (haircut, weight gain).
- A cat visiting that isn't registered.

You can confirm an unknown visit manually:

```
You: Show me unconfirmed visits.
You: Confirm visit 142 as Luna.
```

If it keeps misidentifying or marking unknown, register more reference
photos of that cat (varying angles, lighting).

---

## What the system is NOT

To set expectations clearly:

- **Not a diagnostic tool.** It cannot tell you what's wrong with your
  cat. It only flags "this looks different from normal" — interpretation
  needs a vet.
- **Not a replacement for vet visits.** Use it as a heads-up between
  routine checkups, not as a substitute for them.
- **Not a real-time alerting system.** Nothing will text you when an
  alarm fires. You have to check in periodically.
- **Not perfect.** It will miss things and it will occasionally false-
  alarm. The calibration documented in `simulator/oc_report.md` shows
  the false-positive and detection rates honestly.

---

## Where to go next

- **`docs/USER_GUIDE.md`** — much more detailed reference for power
  users. Covers sensor wiring, the database, troubleshooting, all the
  query tools.
- **`docs/DEVELOPER_INTRO.md`** — for someone who wants to extend the
  system, add new tools, or debug it. No machine learning background
  assumed.
- **`docs/ML_TUTORIAL.md`** — for the curious or technically inclined:
  what each math algorithm in the system actually does, and why.
