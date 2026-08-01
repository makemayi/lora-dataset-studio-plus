# Docker guide

[← Documentation index](../README.md) · [Quick install](../../README.md#setup--install) · [Troubleshooting](troubleshooting.md)

LoRA Dataset Studio ships two Compose stacks:

| Stack | Compose file | Includes | Default data |
|---|---|---|---|
| **API-only** | `docker-compose.yml` | Core app only | `./data-docker` |
| **GPU + ComfyUI** | `docker-compose.gpu.yml` | Core app, ML extras and ComfyUI in one NVIDIA container | `./data-docker-gpu` plus repo-local ComfyUI folders |

The stacks use separate Compose project names and separate application data. Do not point both at the same `LDS_DATA`: two Flask processes must not share one SQLite database and `config.json`.

## Beginner Windows GPU install

1. On GitHub, choose **Code → Download ZIP** and extract the complete folder.
2. Start **Docker Desktop** and wait for it to report that Docker is running.
3. Double-click **`start-docker-gpu.bat`**.
4. Keep the first build/start open. The browser opens when LoRA Dataset Studio is ready.

The launcher **always forces** the four repo-local folders `./run`, `./basedir`, `./data-docker-gpu` and `./bank-images` for ComfyUI, models, app data and bank sources. It does this even when `.env` contains custom `LDS_COMFY_RUN`, `LDS_COMFY_BASEDIR`, `LDS_DATA` or `LDS_BANK_SOURCES` values, specifically so a double-click can never touch an existing ComfyUI. To relocate or reuse storage, do not use the launcher; use the [advanced CLI](#advanced-gpu-cli), which respects those `.env` values.

An existing ComfyUI is never mounted or modified, but stop it first if it uses port `8188` (and stop another LDS on `5050`): two processes cannot share those ports.

On Windows, the double-click launcher also sets `LDS_UID=0` and `LDS_GID=0` **for its Docker process only** because Docker Desktop presents these bind mounts as owner `0:0`; it does not edit `.env`. Advanced CLI and Linux launches keep the UID/GID values from `.env`, which should match the host folders as described under [Linux, Unraid and permissions](#linux-unraid-and-permissions).

Windows needs Docker Desktop's WSL2 backend, GPU support and a compatible NVIDIA driver. Linux hosts need the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). For CUDA 12.x compatibility the minimum driver is 525.60.13 on Linux or 528.33 on Windows/WSL; the default CUDA 12.9 image is best paired with 575.51.03+ on Linux or 576.02+ on Windows. The optional CUDA 13 build requires an R580-series driver.

## Advanced GPU CLI

From the repository root:

```bash
cp .env.example .env
mkdir -p run basedir data-docker-gpu bank-images
docker compose -f docker-compose.gpu.yml up --build
```

Create bind-mount source folders before the first CLI start. Docker may otherwise create them as root while the container runs as your configured user, leaving the app unable to write.

When healthy:

- Studio: `http://127.0.0.1:5050/`
- ComfyUI: `http://127.0.0.1:8188/`

Compose publishes both ports on host interfaces by default. Before using the LAN addresses, read [Security](../../SECURITY.md#the-default-threat-model) and enable the app's access token or place the service behind an authenticated boundary. ComfyUI's own port has no LDS token gate.

## Persistent storage

In **advanced CLI mode only**, the GPU stack has four primary host paths, all relocatable in `.env`. The double-click launcher overrides these four variables with the repo-local defaults on every run:

| Default host folder | Holds | Variable |
|---|---|---|
| `./run` | ComfyUI checkout/environment and Hugging Face cache | `LDS_COMFY_RUN` |
| `./basedir` | Parent containing ComfyUI `models/`, `input/`, `output/` and `custom_nodes/` | `LDS_COMFY_BASEDIR` |
| `./data-docker-gpu` | LDS datasets, database, Trash and `config.json` | `LDS_DATA` |
| `./bank-images` | Host image sources exposed to the Image Bank at `/images` | `LDS_BANK_SOURCES` |

Back up `LDS_DATA` for the app state and `LDS_COMFY_BASEDIR` for model/input/output assets. The large, reproducible ComfyUI environment and cache under `LDS_COMFY_RUN` can also be backed up when avoiding a rebuild matters.

### Use another disk

This is an **advanced CLI-only** setup. Set absolute host paths in the Docker block of `.env`, then start with `docker compose` directly rather than `start-docker-gpu.bat`:

```dotenv
LDS_COMFY_RUN=D:/LDS-Docker/run
LDS_COMFY_BASEDIR=D:/LDS-Docker/basedir
LDS_DATA=D:/LDS-Docker/data
LDS_BANK_SOURCES=E:/ImageDumps
```

On Linux, use native absolute paths. Docker Desktop must have permission to share Windows drives used by a bind mount.

### Deliberately reuse existing ComfyUI data

The double-click launcher cannot opt in to an existing ComfyUI. In **advanced CLI mode**, point `LDS_COMFY_BASEDIR` at the **parent** that directly contains the existing `models/`, `input/`, `output/` and `custom_nodes/` directories — never at `models/` itself. Pointing only the model tree at the wrong level makes path discovery look successful while file-based generation cannot share input/output.

Keep `LDS_COMFY_RUN` separate unless you intentionally want the container to own and update that ComfyUI runtime too. Mounting existing model/data folders is a filesystem operation: verify the resolved host paths before launch and make a backup if another application also writes there.

For a bank, enter the **container path** in the UI: `/images` or a subfolder such as `/images/telegram-export`. The host path stored in `LDS_BANK_SOURCES` does not exist inside the container. Mount the source read-only if originals must be untouchable; **Delete rejected** is the bank action that otherwise removes source files.

## Linux, Unraid and permissions

Set the container identity to the owner of the mounted folders:

```dotenv
LDS_UID=1000
LDS_GID=1000
```

Use `id -u` and `id -g` on Linux. Unraid commonly uses 99/100. A mismatch usually appears as a writable-folder failure in the container log.

`LDS_FORCE_CHOWN=true` is a last-resort adoption switch. It recursively changes ownership of the **LDS data mount only**; it does not adopt the ComfyUI runtime, model tree or Image Bank sources. Do not enable it casually on a large or shared directory.

If an Unraid restart leaves the container with no network, inspect its attached networks and recreate the stack:

```bash
docker inspect -f '{{json .NetworkSettings.Networks}}' lora-dataset-studio-gpu
docker compose -f docker-compose.gpu.yml down
docker compose -f docker-compose.gpu.yml up -d --force-recreate
```

An empty `{}` network result means the old Compose network disappeared. On Unraid, **Settings → Docker → Preserve user defined networks = Yes** prevents the array stop/start cycle from removing it.

## DNS and startup networking

The upstream GPU image checks/updates its installer and ComfyUI at startup, so working DNS is required on every start rather than only during the first build. The Compose file uses:

```dotenv
LDS_DNS=1.1.1.1
```

Change that to a reachable router, corporate resolver or Pi-hole when Cloudflare DNS is inappropriate. A log sequence such as `Could not resolve host: astral.sh` followed by `uv not found after installation` is a DNS/network failure, not a GPU or LDS failure.

There is currently no offline startup mode for this image. A boot race can recover automatically under `restart: unless-stopped`; a container detached from every network cannot.

## CPU and memory limits

The stack is unlimited unless these optional variables are set:

```dotenv
LDS_MEM_LIMIT=32g
LDS_MEMSWAP_LIMIT=40g
LDS_CPUS=12
```

`LDS_MEMSWAP_LIMIT` is memory and swap combined, so it must be greater than or equal to `LDS_MEM_LIMIT`. Leave generous headroom during first boot: torch and the ComfyUI dependency tree are installed there, and an overly tight cap looks like a killed installer rather than a slow one.

GPU allocation is controlled by Docker/NVIDIA rather than these CPU/RAM variables. ComfyUI and LDS scoring share the selected GPU inside this container, so avoid overlapping large passes when VRAM is tight.

## Updates and restarts

An in-app restart exits back to the container supervisor, which respawns LDS on the fixed container bind. It does not create a second loopback-only process.

The in-app source updater cannot replace the immutable `/app` files in an image. From a git checkout, update and recreate instead:

```bash
git pull
docker compose -f docker-compose.gpu.yml up -d --build
```

For a GitHub ZIP, download/extract the newer ZIP into a new folder, stop the old stack, and either keep the new repo-local defaults or carry over explicit `.env` storage paths. Do not copy an old SQLite database while either stack is running.

Rebuilding the image does not delete bind-mounted data. `docker compose down` removes containers and the Compose network, not the host folders listed above.

## Costs and current boundaries

- Budget roughly **20 GB before model downloads**: the CUDA image and its persistent ComfyUI environment are both large. Model stacks add substantially more.
- Building the LDS-side torch dependencies from the CPU wheel index can save several gigabytes, but Image Bank Score then runs on CPU; ComfyUI still owns the GPU.
- **Ollama is not included.** Captioning, framing, head-crop and watermark detection need an Ollama on the host or another reachable machine. Configure `LDS_OLLAMA_URL`/the Compose host mapping, then test it in Settings.
- **Local training is not included.** Connect ai-toolkit on the host where its filesystem is visible, or use cloud training. ComfyUI inside this image is for generation, Studio, Canvas generation and deployment.
- Watermark inpainting is currently listed as unsupported in this Docker lane; model-free crop remains available. Track this and other boundaries in [Known limitations](known-limitations.md).
- Immediately after a container recreate, an extra install can briefly fail while the launcher adopts the large internal virtual environment. The image already ships the ML extras; wait for the adoption message in the log before repairing an optional package.
- Mounting your own `/userscripts_dir` shadows the LDS launcher shipped there. If only ComfyUI starts, remove that override.
- Both published ports are unauthenticated at the container layer. LDS can gate its own remote UI; ComfyUI needs its own firewall, VPN or authenticated proxy if it is exposed.

## API-only stack

The smaller stack is useful for imports, API generation, manual curation/captions, scraping installed after launch, cloud training, backup and publishing:

```bash
cp .env.example .env
mkdir -p data-docker
docker compose up --build
```

It installs `backend/requirements.txt` only. Scraping and ML extras can be installed from the app, but container recreation removes packages installed only into the container layer. ComfyUI features and local ai-toolkit training remain outside this stack.

## Quick diagnostics

```bash
docker compose -f docker-compose.gpu.yml ps
docker compose -f docker-compose.gpu.yml logs --tail=200
nvidia-smi
```

Use the app's **Guide → Getting help** report for LDS capability state. For model/path failures, continue with [Troubleshooting](troubleshooting.md); for every Docker variable and default, read the Docker block in `.env.example`.
