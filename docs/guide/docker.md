# Docker guide

[← Documentation index](../README.md) · [Quick install](../../README.md#setup--install) · [Troubleshooting](troubleshooting.md)

LoRA Dataset Studio provides two beginner Windows launchers:

| Launcher | Use it when | ComfyUI |
|---|---|---|
| **`start-docker.bat`** | You already have ComfyUI on this computer | Keeps your normal host ComfyUI; asks for its folder once |
| **`start-docker-gpu.bat`** | You want a completely fresh NVIDIA Docker setup | Creates an isolated ComfyUI and isolated model/data folders |

Each launcher remembers its own stack and data. Do not run two LDS containers against the same data folder: two Flask processes must not share one SQLite database and `config.json`.

## Beginner Windows install with an existing ComfyUI

1. On GitHub, choose **Code → Download ZIP** and extract the complete folder.
2. Start **Docker Desktop** and wait until it reports that Docker is running.
3. Double-click **`start-docker.bat`**.
4. On first launch, choose either the ComfyUI root containing `main.py` and `models/`, or the portable parent containing `ComfyUI\main.py`.
5. Start ComfyUI with your usual launcher. LDS opens automatically on the first free Studio port.

The selected folder is validated, mounted read/write at `/external-comfyui`, and remembered in a generated local override. If it moves, run **`configure-docker.bat`**. From Docker, LDS connects to the host API at `http://host.docker.internal:8188`.

On **Docker Desktop for Windows** a ComfyUI listening only on `127.0.0.1` is reachable as-is — Docker Desktop proxies that name from the host side, so the default portable launcher needs no change and no firewall hole. This was measured, not assumed. On a **Linux host**, `host.docker.internal` resolves to the host gateway and a loopback-only ComfyUI is genuinely unreachable: start it with `--listen 0.0.0.0` and restrict port 8188 to Docker or your private network. If Studio reports it cannot reach ComfyUI, the launcher prints that same guidance.

## Beginner Windows GPU install

1. On GitHub, choose **Code → Download ZIP** and extract the complete folder.
2. Start **Docker Desktop** and wait for it to report that Docker is running.
3. Double-click **`start-docker-gpu.bat`**.
4. Keep the first build/start open. The launcher prints both actual addresses and opens Studio as soon as Studio responds. Its batch window keeps working until ComfyUI becomes healthy; no second ComfyUI window needs to appear.

The launcher **always forces** the four repo-local folders `./run`, `./basedir`, `./data-docker-gpu` and `./bank-images` for ComfyUI, models, app data and bank sources. It does this even when `.env` contains custom `LDS_COMFY_RUN`, `LDS_COMFY_BASEDIR`, `LDS_DATA` or `LDS_BANK_SOURCES` values, specifically so a double-click can never touch an existing ComfyUI. To relocate or reuse storage, do not use the launcher; use the [advanced CLI](#advanced-gpu-cli), which respects those `.env` values.

The double-click launcher picks both host ports itself, by testing them on the host: Studio between `5050` and `5149`, ComfyUI between `8188` and `8287`, taking the first that is genuinely free. The usual addresses therefore remain `http://127.0.0.1:5050/` and `http://127.0.0.1:8188/`, but an existing LDS, ComfyUI or unrelated service can keep either default port: the launcher leaves it running and selects another one automatically. It prints both final addresses and opens the mapped Studio URL.

The launcher probes rather than handing Docker a port range, because Docker's allocator only tracks the ports it assigned itself: given a range it will still choose a port another program already holds and then fail to start with `ports are not available`. Each resolved port is published as a single fixed port, so `docker stop` followed by `docker start` keeps the address you bookmarked instead of moving it.

Re-running the launcher from the **same checkout** while its container is already running reuses the existing Docker mappings and opens Studio without recreating the container. If the fixed Compose container identity belongs to a different checkout, the launcher stops with a clear collision error rather than modifying or replacing it. It never stops an existing service and does not write the selected ports to `.env`; the allocation applies only to that launcher run.

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

When healthy, the advanced CLI defaults to:

- Studio: `http://127.0.0.1:5050/`
- ComfyUI: `http://127.0.0.1:8188/`

Advanced CLI mode does not scan the launcher ranges. Set `LDS_HOST_PORT` and `LDS_COMFY_HOST_PORT` in `.env` when those defaults are unavailable; Compose continues to respect those explicit values.

Compose binds published ports to **127.0.0.1 by default**. LAN access is an explicit opt-in: set `LDS_BIND_ADDRESS=0.0.0.0`, enable the LDS access token, and restrict the ports with Windows Firewall or another trusted-network boundary. ComfyUI's own port has no LDS token gate, so never expose it publicly.

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

Use **`start-docker.bat`** to opt in to an existing ComfyUI. Select the ComfyUI root containing `main.py` and `models/`, or its portable parent; never select `models/` itself. The launcher validates the path before generating its local Compose override.

Use **`start-docker-gpu.bat`** when you do not want Docker to touch that existing installation. Its `run/` and `basedir/` folders stay isolated beside the checkout.

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

The in-app source updater cannot replace the immutable `/app` files in an image. For a GitHub ZIP installation, double-click **`update-docker.bat`**: it downloads the latest stable Release, keeps the previously selected launcher, rebuilds the image, and preserves `.env`, app data, ComfyUI folders, bank sources, `ollama-data/` and the generated external-ComfyUI override. Pass `main` only when you explicitly want the preview branch.

The code swap is transactional. The updater keeps the previous code aside, then calls the launcher with `--update-rebuild`, which returns only once Docker reports the Studio container healthy. On that confirmation the update is committed and the backup is removed; on any failure the previous code is put back and its launcher is restarted, so a failed update leaves you on the version you already had.

A git checkout is never reset or merged by the updater. Run `git pull --ff-only` yourself, then rerun the same Docker launcher with `--update-rebuild`. Rebuilding replaces containers and code, not bind-mounted user data.

Rebuilding the image does not delete bind-mounted data. `docker compose down` removes containers and the Compose network, not the host folders listed above.

## Costs and current boundaries

- Budget roughly **20 GB before model downloads**: the CUDA image and its persistent ComfyUI environment are both large. Model stacks add substantially more.
- Building the LDS-side torch dependencies from the CPU wheel index can save several gigabytes, but Image Bank Score then runs on CPU; ComfyUI still owns the GPU.
- **Ollama is optional and selected inside LDS Setup.** Choose no Ollama, an existing host Ollama, or the isolated official Docker companion. No model is pulled automatically; start the explicit model download in LDS to see progress or cancel it. The companion publishes no host port and keeps models in `./ollama-data`.
- **Local training is not included.** Connect ai-toolkit on the host where its filesystem is visible, or use cloud training. ComfyUI inside this image is for generation, Studio, Canvas generation and deployment.
- Watermark inpainting is currently listed as unsupported in this Docker lane; model-free crop remains available. Track this and other boundaries in [Known limitations](known-limitations.md).
- Immediately after a container recreate, an extra install can briefly fail while the launcher adopts the large internal virtual environment. The image already ships the ML extras; wait for the adoption message in the log before repairing an optional package.
- Mounting your own `/userscripts_dir` shadows the LDS launcher shipped there. If only ComfyUI starts, remove that override.
- Both published ports are unauthenticated at the container layer. LDS can gate its own remote UI; ComfyUI needs its own firewall, VPN or authenticated proxy if it is exposed.

## Existing-host ComfyUI stack

The smaller `start-docker.bat` stack runs LDS in Docker and connects to the ComfyUI you already maintain on the host. It does not install or launch a second ComfyUI. Its first-run folder picker creates the strict local mount override required for model discovery and shared input/output files.

The image installs `backend/requirements.txt` only. Optional scraping and ML extras can be installed from LDS; local ai-toolkit training remains a separate host/cloud tool.

## Quick diagnostics

```bash
docker compose -f docker-compose.gpu.yml ps
docker compose -f docker-compose.gpu.yml logs --tail=200
nvidia-smi
```

Use the app's **Guide → Getting help** report for LDS capability state. For model/path failures, continue with [Troubleshooting](troubleshooting.md); for every Docker variable and default, read the Docker block in `.env.example`.
