"""Thin wrapper around TRELLIS.2 (microsoft/TRELLIS.2-4B) for Stage B object mesh
generation. Import-guarded: the trellis2/o_voxel packages only exist on a GPU
instance with the Stage B0 setup applied (see docs/GPU_SETUP_LOG.md, 2026-09-06
entry) - this module is safe to import on the VPS, it just can't run there.

Usage (on the GPU instance, after Stage B0 setup):
    from scripts.msa.trellis_client import Trellis2Client
    client = Trellis2Client()
    mesh = client.generate(image)  # PIL.Image (RGBA, real per-pixel alpha) -> TrellisMesh
    client.export_glb(mesh, "object.glb")

`image` MUST be RGBA with a real (non-fully-opaque) alpha channel - a genuine object
mask, e.g. from SAM 3 (gpu/stage_objects.py already produces these per keyframe).
TRELLIS.2's own `preprocess_image` uses that alpha directly and skips its background-
removal model entirely when it sees one (see `_ensure_loaded`'s rembg note below) -
passing a plain RGB image would attempt to load `briaai/RMBG-2.0` instead, which is
both a second gated dependency and unnecessary given Stage B always has a real mask
already. A *sparse/dotted* alpha (e.g. a mask reconstructed by splatting individual
3D points rather than a dense 2D segmentation mask) will produce a fragmented,
holey mesh - TRELLIS treats the alpha as ground truth for object shape, not just a
crop hint. Always pass a dense, filled mask.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

_MODEL_ID = "microsoft/TRELLIS.2-4B"


class Trellis2Client:
    """Lazily loads the TRELLIS.2 pipeline on first use (GPU only)."""

    def __init__(self, model_id: str = _MODEL_ID) -> None:
        self.model_id = model_id
        self._pipeline = None

    def _ensure_loaded(self) -> None:
        if self._pipeline is not None:
            return
        self._ensure_trellis_src_on_path()
        try:
            from trellis2.pipelines import Trellis2ImageTo3DPipeline
        except ImportError as e:
            raise RuntimeError(
                "trellis2 is not installed - this must run on the GPU instance "
                "with Stage B0 setup applied, not the VPS. See docs/GPU_SETUP_LOG.md."
            ) from e

        # Two real, confirmed-on-GPU workarounds for this exact pinned TRELLIS.2
        # checkout + transformers==5.16.1 pairing (2026-09-06) - see
        # docs/DECISIONS.md's "Stage B0 unblocked" entry for the full story and
        # exact error text each one fixes.
        self._patch_rembg_bypass()
        self._patch_dinov3_layer_path()

        self._pipeline = Trellis2ImageTo3DPipeline.from_pretrained(self.model_id)
        self._pipeline.cuda()

    @staticmethod
    def _ensure_trellis_src_on_path() -> None:
        """`trellis2` (and `o_voxel`) were never `pip install`-ed on the GPU
        instance - Stage B0's setup only `git clone`-d them to `trellis-src/`
        (confirmed via `pip show trellis2` returning nothing, 2026-09-06) and
        every prior working invocation added that directory to `sys.path` by
        hand (`smoke_test.py`'s `sys.path.insert(0, "/workspace/trellis-src")`)
        before this client existed. Without this, `from trellis2.pipelines
        import ...` fails with `ModuleNotFoundError` from any script that
        doesn't happen to do the same manual insert - which Stage B1's batch
        driver didn't, and hit exactly that. Override with `TRELLIS_SRC_DIR`
        if a future re-provision installs it elsewhere."""
        import os
        import sys

        src_dir = os.environ.get("TRELLIS_SRC_DIR", "/workspace/trellis-src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)

    @staticmethod
    def _patch_rembg_bypass() -> None:
        """`Trellis2ImageTo3DPipeline.from_pretrained` unconditionally instantiates
        `rembg_model` (`BiRefNet`, config points at the gated `briaai/RMBG-2.0`) even
        though it's never called when the input image already has a real alpha
        channel (`preprocess_image`'s `has_alpha` branch) - which is always true for
        Stage B's real inputs (SAM 3 masks). Stub the class so construction succeeds
        without that second gated download; the stub raises if it's ever actually
        invoked, so a caller passing a flat-alpha/RGB image fails loudly instead of
        silently downloading nothing and producing garbage.
        """
        import trellis2.pipelines.rembg as rembg_mod

        class _NoRembgStub:
            def __init__(self, *a, **kw) -> None:
                pass

            def to(self, device):
                return self

            def cpu(self):
                return self

            def __call__(self, img):
                raise RuntimeError(
                    "rembg_model invoked but Trellis2Client.generate() requires "
                    "RGBA input with a real (non-fully-opaque) alpha channel - "
                    "pass a real object mask (e.g. from gpu/stage_objects.py's "
                    "SAM 3 output), not a plain RGB image."
                )

        rembg_mod.BiRefNet = _NoRembgStub

    @staticmethod
    def _patch_dinov3_layer_path() -> None:
        """`trellis2.modules.image_feature_extractor.DinoV3FeatureExtractor.
        extract_features` accesses `self.model.layer` directly - correct against
        the transformers version TRELLIS.2 was originally written for, but
        transformers==5.16.1 (what Stage B0 actually installs, 2026-09-06) nests
        the encoder's layer list one level deeper: `DINOv3ViTModel.model` is a
        `DINOv3ViTEncoder` with its own `.layer` (top-level `.embeddings` /
        `.rope_embeddings` are unaffected). Confirmed by inspecting the actual
        loaded model's `named_children()`, not by guessing. Monkeypatch the method
        rather than editing vendored `trellis-src` in place.
        """
        import torch.nn.functional as F
        import trellis2.modules.image_feature_extractor as ife

        def _patched_extract_features(self, image):
            image = image.to(self.model.embeddings.patch_embeddings.weight.dtype)
            hidden_states = self.model.embeddings(image, bool_masked_pos=None)
            position_embeddings = self.model.rope_embeddings(image)
            for layer_module in self.model.model.layer:
                hidden_states = layer_module(hidden_states, position_embeddings=position_embeddings)
            return F.layer_norm(hidden_states, hidden_states.shape[-1:])

        ife.DinoV3FeatureExtractor.extract_features = _patched_extract_features

    def generate(self, image: "Image.Image"):
        """Run image-to-3D generation. `image` must be RGBA with a real, dense
        (not sparse/point-splatted) alpha mask - see module docstring. Returns a
        TRELLIS mesh object (has .vertices, .faces, .attrs, .coords, .layout,
        .voxel_size - see o_voxel.postprocess.to_glb for the export contract)."""
        self._ensure_loaded()
        outputs = self._pipeline.run(image)
        mesh = outputs[0]
        mesh.simplify(16_777_216)  # nvdiffrast vertex-count limit
        return mesh

    def export_glb(
        self,
        mesh,
        out_path: str | Path,
        *,
        decimation_target: int = 200_000,
        texture_size: int = 1024,
    ) -> Path:
        import o_voxel

        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=mesh.layout,
            voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=decimation_target,
            texture_size=texture_size,
            remesh=True,
            remesh_band=1,
            remesh_project=0,
            verbose=False,
        )
        out_path = Path(out_path)
        glb.export(str(out_path))
        return out_path
