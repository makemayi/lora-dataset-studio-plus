"""Makes `backend/tests` a REGULAR package, which is the only thing that keeps
`import tests.<module>` pointing here.

Three test modules import a sibling by its full dotted name rather than by a
bare `import test_x` — test_blend_provenance borrows a fixture builder from
test_blend_sweep, test_instagram_scan borrows from test_scraping_service, and
test_no_personal_data imports ITSELF to stub its own seams. Without this file
all three died at COLLECTION on any machine that has `ultralytics` installed:

    ModuleNotFoundError: No module named 'tests.test_blend_sweep'

...because ultralytics (8.4.84 here) ships its own top-level `tests/` package
into site-packages. Python's path scan treats a directory WITHOUT `__init__.py`
as a namespace portion and keeps looking, so a regular `tests` package further
down sys.path wins over this directory even though this one comes first. A
regular package here stops that scan at the first entry, where it belongs.

`pythonpath`/`PYTHONPATH` cannot fix it: backend was already on the path first.
The problem was never a missing entry, it was which kind of package wins.
"""
