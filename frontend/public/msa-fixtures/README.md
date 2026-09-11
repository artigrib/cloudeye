# MSA fixtures

`02_modular_home.glb` / `02_modular_home_gaps.json` are `scripts/msa/bootstrap.py`'s
real output for the `02_modular_home` golden fixture
(`tests/fixtures/msa/02_modular_home/`), regenerated 2026-09-06 against the current
(post-Stage-E0) `scripts/msa/{bootstrap,export_glb}.py` so the GLB's node naming
includes the `<obj>/visual`, `<obj>/collision`, and `Plan/*` groups `MsaSceneViewer`
expects (an earlier copy of this fixture, taken from MSA Stage A1's batch run before
E0 landed, had none of those - a stale-fixture bug caught while screenshotting the
collision/plan layers, see `docs/DECISIONS.md`'s Stage E1 entry). Regenerate with:

```
uv run python3 -c "
from pathlib import Path
from scripts.msa.bootstrap import run_bootstrap, load_object_inputs_from_hulls_json
scene_dir = Path('tests/fixtures/msa/02_modular_home')
out_dir = Path('/tmp/msa_e1_regen')
out_dir.mkdir(parents=True, exist_ok=True)
objects = load_object_inputs_from_hulls_json(scene_dir / 'scene_objects_hulls.json')
run_bootstrap(scene_dir, out_dir, object_inputs=objects)
"
```

Used only by the dev-only `/_msa-viewer` route (`src/pages/MsaViewerPage.tsx`) to
exercise `MsaSceneViewer` without a live backend - MSA has no API route yet
(script-based, writes to disk).
