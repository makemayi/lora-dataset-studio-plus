"""Give a test dataset the FILES its kept rows claim to have.

A dataset is two things: rows in the database and images on disk under
``dataset_images_root()/<dataset_id>/``. A fixture that creates only the rows
builds a dataset that cannot exist — and the launch path says so, on purpose:

    checkpoint_registry.prepare_launch()
      -> run_snapshot.build()      reads every kept file to hash its BYTES
      -> prepared_generation_identity()  returns None when a row has no hash
      -> lora_training: "could not freeze the Dataset provenance for training"

That guard is the whole point of the run snapshot: two runs may only be called
identical when their pixels were proven identical. A fixture cannot be excused
from it, because a real launch on files that are not there cannot be excused
from it either — ``export_dataset_to_aitoolkit`` skips every missing source and
would train on whatever is left.

So the fixtures write real, decodable images instead. One per kept row, each a
different colour, so every row gets its OWN content signature and a test that
one day asserts "these two images differ" is not quietly comparing a file to
itself.
"""
from __future__ import annotations

import os


def _colour(index: int) -> tuple:
    """A distinct, deterministic colour per row — distinct BYTES per row."""
    return (37 * (index + 1) % 256, 91 * (index + 3) % 256, 151 * (index + 7) % 256)


def write_kept_image_files(dataset_id, size=(64, 64)) -> list:
    """Write one small image for every KEPT row of ``dataset_id``.

    Call it after the rows are committed (and after any later batch of rows):
    it reads the rows back, so the files and the manifest can never disagree
    about which images the dataset holds. Returns the paths written.

    Must run inside an app context — it resolves the dataset folder through the
    same helper the production code uses, so tests land exactly where a launch
    will look.
    """
    from PIL import Image

    from app.models import FaceDatasetImage
    from app.services.dataset_storage import ensure_dataset_dir

    folder = ensure_dataset_dir(dataset_id)
    rows = (FaceDatasetImage.query
            .filter_by(dataset_id=dataset_id, status='keep')
            .order_by(FaceDatasetImage.id.asc()).all())
    written = []
    for index, row in enumerate(rows):
        if not row.filename:
            continue
        path = os.path.join(folder, row.filename)
        Image.new('RGB', size, _colour(index)).save(path)
        written.append(path)
    return written
