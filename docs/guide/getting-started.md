# Getting started

LoRA Dataset Studio turns one reference photo into a trained, ranked LoRA —
curation, captioning, face-scoring and training behind a single browser tab, on
your own machine. The useful part of LoRA training isn't the training; it's
building a clean, balanced, well-captioned image set. This app puts that whole
pipeline behind one UI.

> **In a hurry?** Launch the app, let the **Setup** wizard scan your machine,
> and create your first dataset from your own photos — no API key, no GPU, no
> external tool required for that first step.

---

## Two ways to run it

| | API-only | Full local |
|---|---|---|
| **What works** | Create datasets, generate via Gemini/ChatGPT, curate, caption via API, export ZIP | Everything — plus local (Klein) generation, JoyCaption, face scoring, masks, training, Test Studio |
| **Needs** | Python 3.10–3.12, an API key | ComfyUI and/or ai-toolkit + an NVIDIA GPU (12 GB+ for local generation) |
| **Good for** | Laptops, first try, cloud training | The full pipeline on a training rig |

You can start API-only and add the local tools later — features light up
automatically when their tool is detected.

## First launch

**Windows (one command):** download `LoRA-Dataset-Studio-windows.zip` from the
[latest release](https://github.com/perfectgf/lora-dataset-studio/releases/latest),
extract it, then double-click `start.bat`. Releases contain an archive/source, not
a prebuilt executable launcher. `start.bat` finds or downloads a compatible Python
(3.10–3.12), creates `.venv`, installs the requirements, and opens the app at
`http://127.0.0.1:5050/`.

**Any OS (manual venv):**

```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
python backend/run.py
```

**Docker (API-only):** `cp .env.example .env`, then `docker compose up --build`.

The full install matrix (Windows release ZIP, GPU requirements, external tools)
lives in the README on GitHub.

## What is running on your machine {#architecture}

A full-local install is **three separate programs**, in **two folders**, on
**two ports**, with **two Python environments**. Nothing in the app used to say
so, and the cost of guessing is real: a user spent hours patching ai-toolkit's
own web UI — a component this app never talks to — while the actual problem was
one line of `config.json` (reported by strouder, GitHub #19).

| Component | Where | What it does | Port |
|---|---|---|---|
| **LoRA Dataset Studio** (Flask) | the folder you extracted, **its own `.venv`** | The UI you are reading this in. Curation, captioning, and the thing that **starts and stops training**. | **5050** |
| **ai-toolkit** (`run.py`) | the folder you point at in Settings, **its own venv — the one with `torch`** | The training **engine**. It is run as a command-line process; it has no UI of its own. | — |
| ai-toolkit's Next.js UI (`ui/`) | inside the ai-toolkit folder | An **unrelated** web interface that ships with ai-toolkit. This app never launches it, never reads it, and never writes to it. | 8675 |

Two consequences worth remembering:

- **The Studio reads `config.json` and drives training.** That file belongs to
  this app (it sits in its data folder, and every key in it has a field in
  **Settings**). Editing anything inside ai-toolkit's `ui/` folder changes
  nothing here — if you ever find yourself editing `ui/dist/…`, you are in the
  wrong project.
- **The two Python environments are not interchangeable.** The Studio's `.venv`
  runs the web app; ai-toolkit's venv is the one that must have `torch` and the
  training dependencies. Settings ▸ Local tools ▸ **Python interpreter** is how
  you tell the app which interpreter is ai-toolkit's — and its **Test** button
  now checks that the interpreter can really `import torch`, not just that the
  file exists.

There is no process stacking to worry about on our side: the Studio is a single
Flask process, and starting a second one on the same port fails loudly instead
of running invisibly alongside the first.

### Supported Python versions {#python-versions}

- **LoRA Dataset Studio: Python 3.10 – 3.12.** `start.bat` finds or downloads
  one for you; the optional ML extras (insightface, onnxruntime, `numpy<2`)
  only publish wheels for those versions.
- **ai-toolkit: Python 3.11 is the safe choice.** On **3.13** its pinned
  `scipy==1.12.0` has no wheel, pip falls back to building from source and dies
  on a missing Fortran compiler (measured and reported by strouder, GitHub #19).
  On Windows, install **3.11.9** — it is the last 3.11 with a binary installer;
  later 3.11.x are source-only security releases.
- The two do **not** have to match. They are separate environments on purpose.

### If Hugging Face downloads fail {#hf-downloads}

If a base-model download dies with something that reads like a network error,
check whether `HF_HUB_ENABLE_HF_TRANSFER=1` is set in your environment. That
turns on an optional fast-download accelerator which needs the `hf_xet` (or
`hf_transfer`) package installed **in the environment doing the downloading**;
without it, transfers abort with a misleading message. This app never sets that
variable — it comes from your shell, ai-toolkit's `.env`, or a ComfyUI launcher.
Either fix works: set `HF_HUB_ENABLE_HF_TRANSFER=0`, or `pip install hf_xet`.
The training failure panel now recognises this case and says so.
*(Reported by bobba84, GitHub #18.)*

## The Setup wizard

On first launch you land in **Setup**. It scans your machine automatically and
walks through five steps — each one unlocks a set of features:

1. **Image generation** — add a Gemini or OpenAI API key (or point at a local
   Klein model) so the app can generate dataset images.
2. **ComfyUI** — unlocks local (Klein) generation and the Test Studio.
3. **Ollama** — the local vision model behind auto-captioning, framing
   auto-classify and head-crop.
4. **Quality tools** — face-similarity scoring and person masks (a one-click
   `pip install`).
5. **ai-toolkit** — the training engine.

Nothing is mandatory: **Skip setup** is always available, and every step can be
revisited later from **Settings**, where each tool has a Test button that tells
you immediately whether the app can see it.

## Around the app

- **Datasets** — the home tab and your **library**: photo tiles of every
  dataset, grouped by model family, with a search box and a badge for each
  family you've already trained. Create one and work it through the guided
  flow (source → curate → caption → train).
- **🏋️ Runs** — every training in one place, cloud *and* local: live progress,
  the settings each launch used, retry a failed run (↻), continue a finished
  one (▶), and download the LoRA (appears once ai-toolkit or a vast.ai key is set).
- **Test Studio** — grid-test a trained LoRA across checkpoints and strengths,
  vote, and rank (appears once ComfyUI is reachable).
- **Guide** — this manual.
- **Setup** — the guided wizard, re-runnable anytime.
- **Settings** — everything the wizard configures, plus server, updates,
  maintenance and the diagnostic report.

Next chapter: **Using the app** — the full walkthrough, dataset type by dataset
type.
