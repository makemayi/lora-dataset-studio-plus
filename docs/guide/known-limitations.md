# Known limitations

[← Documentation index](../README.md) · [Settings reference](settings-reference.md) · [Troubleshooting](troubleshooting.md)

These are current boundaries, not setup failures.

| Area | Current boundary | Practical path |
|---|---|---|
| **Test Studio families** | Studio workflows currently cover Z-Image, SDXL and Krea 2; train/manage support is broader | Use family-native external inference for families that do not yet have a Studio workflow |
| **Krea 2 img2img Studio mode** | `backend/workflows/krea2_turbo_img2img.json` exists but is not wired to a separate Studio mode | Krea 2 Edit remains available for dataset/reference generation; Studio uses its reachable text-to-image path |
| **Canvas comparison generation** | A single launch requires checkpoints from one model family | Run separate same-family comparisons; cross-family workflows do not share one base graph |
| **Dual captions** | Local training only; Krea 2 and Anima cache text embeddings and use the long caption alone | Keep the long caption complete for every family |
| **Cloud training** | Support is family- and variant-specific, and full-state continuation is local-only | Trust the launch UI's live compatibility reason before renting a pod |
| **GPU Docker** | Ollama and local ai-toolkit training are not bundled; watermark inpainting is not currently available in this lane | Connect external Ollama/ai-toolkit, use cloud training, and use model-free watermark crop where suitable |
| **Remote access** | The app has no user accounts and ComfyUI's published port is not protected by the LDS access token | Keep loopback defaults or use tokens, firewall/VPN and an authenticated reverse proxy as described in [Security](../../SECURITY.md) |
| **Browser-local preferences** | Some convenience choices, such as the last generator, are remembered per browser | Explicitly select a configured engine after moving to a new browser/profile |

ComfyUI-dependent paths are covered extensively against a mocked API, but not every model/custom-node combination has been exercised on live third-party installations. A failed preflight should name the missing asset; use **Settings → Local tools → Test** and attach the diagnostic report when a supported layout is not detected.

Provider policies, moderation and service availability are outside this project's control. See the direct [Gemini](settings-reference.md#what-the-gemini-engine-will-and-will-not-do), [ChatGPT subscription](settings-reference.md#chatgpt-subscription-experimental), [OpenRouter/image-engine](settings-reference.md#image-engines) and [Pexels](workflow.md#the-built-in-web-scraper) notes before depending on those lanes.
