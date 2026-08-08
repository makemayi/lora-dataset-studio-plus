/* Pinokio update script.

   `--ff-only` on purpose: it is the exact command the in-app updater runs, so
   whichever surface the user reaches for, the checkout moves the same way and a
   local edit is reported as a refusal instead of being buried under a merge
   commit. The pip run afterwards is what picks up a new or bumped dependency;
   it is a no-op when nothing changed.

   The launcher's Update tab is offered only while the server is stopped (see
   pinokio.js) — updating a running tree then restarting from Pinokio is the
   clean order. */
module.exports = {
  run: [
    {
      method: "shell.run",
      params: {
        message: "git pull --ff-only"
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: "uv pip install -r backend/requirements.txt"
      }
    },
    {
      method: "notify",
      params: {
        html: "Up to date. Click <b>Start</b> to run the new version."
      }
    }
  ]
}
