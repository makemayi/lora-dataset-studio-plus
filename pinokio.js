/* Pinokio launcher UI (https://pinokio.computer).

   This repository IS the launcher: Pinokio's "Download from URL" clones it into
   its own `api/` folder, reads this file for the menu, and runs the sibling
   install.js / start.js / update.js / reset.js scripts IN PLACE. There is no
   second clone into an `app/` subfolder (the usual launcher shape) because the
   app being launched is this checkout — one tree means Pinokio's Update and the
   in-app updater fast-forward the exact same working copy.

   The venv lives at `env/` (Pinokio's convention, gitignored). `data/` and
   `config.json` stay where every other install shape puts them: next to the
   sources, so a Pinokio install carries its datasets across updates untouched.

   Menu shape follows the current Pinokio launchers (pinokiofactory/comfy):
   `info.exists` for the installed state, `info.running` per script, and
   `info.local("start.js").url` for the "Open Web UI" tab, which start.js sets
   from the address the server actually bound (port 5050 may be taken). */
module.exports = {
  version: "1.0",
  title: "LoRA Dataset Studio",
  description:
    "Build, curate, caption and clean LoRA training datasets in your browser, then train, compare and iterate — local Flask app, your files stay on your disk. https://github.com/perfectgf/lora-dataset-studio",
  icon: "icon.png",
  menu: async (kernel, info) => {
    let installed = info.exists("env")
    let running = {
      install: info.running("install.js"),
      start: info.running("start.js"),
      update: info.running("update.js"),
      reset: info.running("reset.js")
    }
    if (running.install) {
      return [{
        default: true,
        icon: "fa-solid fa-plug",
        text: "Installing",
        href: "install.js"
      }]
    }
    if (!installed) {
      return [{
        default: true,
        icon: "fa-solid fa-plug",
        text: "Install",
        href: "install.js"
      }]
    }
    if (running.start) {
      let local = info.local("start.js")
      if (local && local.url) {
        return [{
          default: true,
          icon: "fa-solid fa-rocket",
          text: "Open Web UI",
          href: local.url
        }, {
          icon: "fa-solid fa-terminal",
          text: "Terminal",
          href: "start.js"
        }]
      }
      return [{
        default: true,
        icon: "fa-solid fa-terminal",
        text: "Terminal",
        href: "start.js"
      }]
    }
    if (running.update) {
      return [{
        default: true,
        icon: "fa-solid fa-terminal",
        text: "Updating",
        href: "update.js"
      }]
    }
    if (running.reset) {
      return [{
        default: true,
        icon: "fa-solid fa-terminal",
        text: "Resetting",
        href: "reset.js"
      }]
    }
    return [{
      default: true,
      icon: "fa-solid fa-power-off",
      text: "Start",
      href: "start.js"
    }, {
      icon: "fa-solid fa-rotate",
      text: "Update",
      href: "update.js"
    }, {
      icon: "fa-solid fa-plug",
      text: "Install",
      href: "install.js"
    }, {
      icon: "fa-regular fa-circle-xmark",
      text: "Reset",
      href: "reset.js",
      confirm: "This deletes the Python environment (env/) and rebuilds it on the next Install. Your datasets, config and Image Bank are NOT touched. Continue?"
    }]
  }
}
