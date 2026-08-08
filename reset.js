/* Pinokio reset script.

   Removes ONLY the Python environment. The usual launcher reset (`fs.rm app`)
   deletes the cloned application folder, which here is the same folder that
   holds `data/`, `config.json` and the Image Bank — a reset that silently
   destroys months of curation is not a repair, it is a data loss. A broken
   install is a broken venv; that is what this rebuilds. */
module.exports = {
  run: [
    {
      method: "fs.rm",
      params: {
        path: "env"
      }
    },
    {
      method: "notify",
      params: {
        html: "Python environment removed. Click <b>Install</b> to rebuild it — your datasets, config and Image Bank were left untouched."
      }
    }
  ]
}
