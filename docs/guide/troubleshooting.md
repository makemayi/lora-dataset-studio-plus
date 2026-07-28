# Troubleshooting

Symptom-first, most-reported first. If your problem isn't here, the next
chapter (**Getting help**) shows how to report it with one click.

---

## "No Z-Image model available" in the Test Studio or training panel

**Why:** the Test Studio generates through ComfyUI, so the Z-Image *base model*
must physically live in your ComfyUI install — and the scanner only accepts it
inside a sub-folder whose name contains `z image` (or `zimage`). A file dropped
loose in `models/unet` is **not** detected.

**Fix:** lay the stack out like this inside your ComfyUI folder, then re-test:

```
models/unet/z image/<your Z-Image checkpoint>.safetensors
models/text_encoders/Z image/qwen_3_4b.safetensors
models/vae/z ae.safetensors
```

**The text encoder and the VAE are flexible** — only the base model needs that
sub-folder. The app resolves those two itself: any capitalisation, any separator
and any sub-folder work, so `models/vae/z_ae.safetensors`, `models/vae/ae.safetensors`
(the name ComfyUI's own Z-Image page uses), `text_encoders/Z Image/qwen_3_4b.safetensors`
and a bare `text_encoders/qwen_3_4b.safetensors` are all found, including under an
`extra_model_paths.yaml` root. **Do not rename your files to match the layout above.**
If the app still says one is missing, the message lists what it accepted and where it
looked; you can also pin either file by hand with the `zimage.vae` /
`zimage.text_encoder` settings (see *Settings reference → Config-file-only settings*).

A Z-Image LoRA only works on a Z-Image base — a regular SD/SDXL graph
(20–30 steps, CFG 7) renders garbage. The two Z-Image builds then want opposite
sampler settings, and the Test Studio proposes the right pair per base model:
**Z-Image-Turbo** is guidance-distilled and wants euler / simple / **8 steps /
CFG 1.0**, while the non-distilled **Z-Image Base** needs roughly **30–50 steps at
CFG 3–5** (ComfyUI's own recommendation) — run Base at CFG 1 and it renders mush.
Those are starting points on a sweepable axis, not measured optima: the Studio grid
exists to let you find yours.

## "No SDXL checkpoint found" on a fresh install

**Why:** the app derives the models folder from **Settings → Local tools →
ComfyUI install directory**. If only the API URL is set, there's nothing to scan.

**Fix:** point the install directory at the folder that contains `models/` and
`main.py` (the Setup wizard detects it for you), then hit **Test**. SDXL
checkpoints are scanned from `models/checkpoints`.

## The Krea 2 Turbo Test Studio says a custom node is missing

**Why:** the Krea grid rebalances the Qwen3-VL conditioning through a small
community node (class `ConditioningKrea2Rebalance`). It isn't a stock ComfyUI
node, so a ComfyUI that doesn't have it can't run the Krea pipeline and the
Studio stops before wasting a run — the same up-front check used for missing
model files.

**Fix:** install the **ComfyUI-Conditioning-Rebalance** pack (in ComfyUI-Manager,
search **"Krea 2 Conditioning"** — repo
`https://github.com/nova452/ComfyUI-Conditioning-Rebalance`), then restart
ComfyUI and relaunch the test. The Studio's error banner names this pack and
links it directly. Either that original pack or its `comfyui-krea2-conditioning`
fork works — the app pins the node so your rebalance-strength setting is applied
the same way on both.

## The reference crop isn't centered on the face

**Why:** on a fresh clone the configured Ollama vision model isn't pulled yet,
so head detection silently falls back to a centered square crop. The app now
shows a warning toast naming the missing model when this happens.

**Fix:** **Setup → Ollama** — pull the vision model (use the **Instruct**
variant, not *Thinking*), or click the tile's crop button and frame it by hand.
**↺ Reset to auto** re-runs the auto-crop after the model is installed.

## Training log looks frozen for several minutes

**Why:** ai-toolkit's output is block-buffered during model load and latent
caching — nothing prints even though it's working. A "warming up" phase before
the first logged step is expected, and Krea-2-Raw runs are *hours* long by
design.

**Fix:** nothing to fix — check GPU utilization or watch the ai-toolkit output
folder for new files if you want proof of life. The cloud runs page has a
stall watchdog that kills genuinely stuck runs.

## ai-toolkit isn't detected (conda / uv / no venv)

**Why:** the app auto-detects ai-toolkit's Python from a `venv/` or `.venv/`
folder next to its `run.py`. Installs that use conda, uv or the system Python
have no such folder, so the Test button can't find an interpreter — training
and JoyCaption stay hidden.

**Fix:** in **Settings → Local tools → ai-toolkit**, keep the directory pointing
at the ai-toolkit folder and fill the optional **Python interpreter** field with
the full path to the python that has ai-toolkit's dependencies (e.g.
`C:\miniconda3\envs\aitk\python.exe`), then hit **Test**. ComfyUI Desktop installs
are recognized automatically — no extra step.

## Reddit scan says "rate limiting requests, retry in Ns" (429)

**Why:** out of the box, Reddit scans authenticate with a **public client id
shared by many people** (the gallery-dl one). Reddit's quota — about 1000
requests per 10-minute window — is attached to that id, so other users can
exhaust it before your very first scan of the day. The "retry in Ns" number is
just the time left in the current 10-minute window.

**Fix:** get your own free client ID (one minute, no app secret involved):
**Settings → Scraping & sources** has the field plus a built-in step-by-step
guide. The one trap: on reddit.com/prefs/apps, pick the app type
**installed app** — a *web app* or *script* id comes with a client secret and
Reddit then rejects the anonymous login this app uses (every scan fails
with 401). Takes effect immediately, no restart needed.

## ComfyUI shows as unreachable

Check **Settings → Local tools → ComfyUI API URL** (default
`http://127.0.0.1:8188`), confirm ComfyUI is actually running, and check that a
firewall or a different bind interface isn't blocking the connection. The
**Test** button answers immediately.

## ComfyUI runs in another container (or WSL, or another machine) and generation fails

**Symptom:** setup goes green — the ComfyUI URL answers, the install directory is
accepted — and then every generation fails.

**Why:** the app talks to ComfyUI through **two** channels, and only one of them is
the network.

1. **The HTTP API** (`Settings → Local tools → ComfyUI API URL`). This is what the
   *Test* button and the Setup wizard check.
2. **The filesystem.** Every local engine (Klein, Krea 2 Edit, Klein watermark
   cleaning) hands ComfyUI its source image by **copying the file into ComfyUI's
   `input/` folder**, and the result comes back from its `output/` folder. There is
   no upload over the API on that path.

A ComfyUI in a separate container, in WSL, or on another host does not share those
folders with the app by default. The URL answers, so everything looks configured —
and then the copy writes into a folder ComfyUI cannot see, or fails outright.

**What it takes to work:**

- `input/` and `output/` must be visible **to both sides at the same path**. Not
  "an equivalent folder": the app writes `<input>/edit_source_….png` and then tells
  ComfyUI to load `edit_source_….png` from *its own* input folder — the two must be
  the same directory.
- The app's process must be able to **write** into `input/` (a read-only bind mount
  is not enough), and ComfyUI must be able to read it.
- If ComfyUI was started with `--input-directory` / `--output-directory`, set the
  matching paths in **Settings → Local tools → Advanced: ComfyUI folder overrides**.
  Those fields take the path **as seen by the app**.

With Docker, that means bind-mounting the same host folders into both containers at
identical paths, e.g.:

```yaml
# both services
volumes:
  - /srv/comfyui/input:/srv/comfyui/input
  - /srv/comfyui/output:/srv/comfyui/output
```

and then pointing the two override fields at `/srv/comfyui/input` and
`/srv/comfyui/output`. The shipped `docker-compose.yml` deliberately does **not**
do this: it runs the app in API-only mode, where ComfyUI is out of scope.

**How you'll know:** the failure now says so. Settings flags an override folder it
cannot write into, the Setup wizard warns while you configure (a warning, never a
blocker — mounting volumes afterwards is fine), and a generation that cannot reach
the folder answers with the folder path and the reason instead of a bare `500`.
Those messages are path-redacted, so they are safe to paste in a help thread.

**Everything else keeps working without shared folders**: the API engines (Gemini,
ChatGPT, OpenRouter), scraping, curation, captioning through Ollama, training, and
Hugging Face publishing. Only the ComfyUI-local engines need the filesystem.

*(Reported by nofaceman on Discord.)*

## Klein engine stays greyed out

Klein needs a reachable ComfyUI **and** the Klein model files (~16 GB VRAM
class). **Setup → ComfyUI** offers the download; the license-gated fp8 model
needs a Hugging Face token (Settings → Local tools).

Whatever the cause, the greyed-out engine now **names it**, and the Setup wizard
shows the same sentence — the two screens read one verdict, so they cannot send
you to fix different things. The causes, and what each one means:

| What it says | What it means | What to do |
| --- | --- | --- |
| `Configure ComfyUI in Settings` | ComfyUI is not answering | Start it, or fix the API URL |
| `Klein <file(s)> missing` | that weight is not on disk | Download it in Setup ▸ Install components |
| `… is on disk but cannot be loaded` | the file is there but unreadable | See below |
| `Your ComfyUI doesn't have <value>` | the graph pins a widget value your ComfyUI doesn't offer | Install the named node pack, restart ComfyUI |
| `disabled in Settings (engines)` | you turned the engine off | Re-enable it in Settings ▸ Engines |

### "On disk but cannot be loaded"

A `.safetensors` file declares the length of its JSON header in its first eight
bytes. A download that was **cut short or corrupted** leaves a file that is
shorter than it claims — plausible size, right name, right folder, and no loader
can open it. A licence or login page saved as `.safetensors` fails the same way.

Setup used to tick these as **✓ Installed** (the file existed) while the
generation page refused the engine. It now shows **⚠ On disk, unreadable** with
the file name and the reason, and **↻ Download again** replaces the bad file
instead of reporting "already present" and doing nothing.

If you placed the file by hand somewhere other than the folder Setup downloads
into, delete it yourself first — the app only replaces files at its own path.

*(Reported by zigzag4794 on Discord.)*

### "Not checked" is not "ready"

With ComfyUI stopped, the checks that need it cannot run, and the app reports no
gap rather than inventing one. Your files being on disk is therefore **not** a
clean bill of health, and Setup says so instead of showing a tick it did not
earn. Start ComfyUI and re-check.

## Port 5000 conflict on macOS

macOS reserves port 5000 for AirPlay Receiver. Change the port in
**Settings → Server & access** (e.g. 5050) and restart.

## Garbled characters in the Windows console

Cosmetic only — some UTF-8 text renders wrong on the legacy console codepage.
The app itself is unaffected.

## `npm install` fails with `Cannot find module @rollup/rollup-<platform>-...`

Only relevant if you rebuild the frontend yourself (the repo ships `dist/`
prebuilt). It's a known npm bug: delete `frontend/node_modules` +
`frontend/package-lock.json` and run `npm install` again on this machine.

## A cloud run seems stuck

Open the **Cloud** tab: every run shows its live phase, and the stall watchdog
(Settings → Training → stall timeout) rescues logs and kills the pod if no step
progress happens for too long. Orphaned pods are also destroyed automatically
at every app start — you never pay for a forgotten GPU.
