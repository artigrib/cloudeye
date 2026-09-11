# Third-party components

CloudEye is licensed under Apache-2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)). This
file lists everything third-party that the project **bundles** or **depends on by name**,
with where it came from and under what terms.

Rule used throughout: where a licence could not be verified from inside this repository or
from the source the asset was taken from, it is written **UNKNOWN**. Nothing is guessed.

---

## Models — depended on by name, never redistributed

No model weights are in this repository, in the working tree or in its history. The code
downloads them by name at runtime, under **the operator's own account and credentials**, or
calls them through an HTTP API under the operator's own key. Anyone deploying this project
is responsible for reading and accepting those terms. This file summarises them; it does not
substitute for the licence text at the links, and it is not legal advice.
Per-model detail, including what each one costs and where it runs, is in
[docs/MODELS.md](docs/MODELS.md).

### nvblox / nvblox_torch

- **Used for:** TSDF + ESDF integration, the band layer, floor plane, room mesh.
- **Version:** `nvblox_torch` **0.0.10**, the prebuilt wheel
  `nvblox_torch-0.0.10+cu12ubuntu24-py3-none-linux_x86_64.whl`.
- **Source:** <https://github.com/nvidia-isaac/nvblox> — installed from that project's own
  GitHub release by `pipeline/box/onstart_nvblox.sh:80,95`. **Not redistributed here:** this
  repository contains no nvblox source, no wheel, and no binary. Only the install command.
- **Licence:** **UNKNOWN from this repository.** Nothing here states nvblox's licence. The
  build patches under `patches/nvblox-cuda128/` carry an Apache-style header tail in their
  diff context, which is indicative and not verification. Read the upstream `LICENSE`.
- **Also:** `patches/nvblox-cuda128/` contains four patches this project wrote to build
  nvblox under CUDA 12.8 / GCC 13, plus the draft upstream issues. The patch *diffs* quote
  short fragments of upstream nvblox and stdgpu source as diff context, as any patch must.

### MapAnything

- **Used for:** monocular 3D reconstruction (the `infer` stage).
- **Version:** weights `facebook/map-anything-apache`, no revision pin; code
  `mapanything==1.1.4` from a shallow clone with no commit recorded.
- **Source:** <https://huggingface.co/facebook/map-anything-apache> ·
  <https://github.com/facebookresearch/map-anything>
- **Licence:** **Apache-2.0** for both the `-apache` weights (HF model card tag) and the
  repository code. The un-suffixed `facebook/map-anything` weights are **CC-BY-NC-4.0**
  (non-commercial) and are deliberately not used; `gpu/stage_infer.py:74` allowlists exactly
  one repo id so a per-job override cannot substitute them.

### SAM 3

- **Used for:** open-vocabulary object segmentation (the `objects` stage).
- **Version:** weights `facebook/sam3` (**gated**), no revision pin; code `sam3==0.1.0`.
- **Source:** <https://github.com/facebookresearch/sam3> ·
  <https://huggingface.co/facebook/sam3>
- **Licence:** **"SAM License"** — Meta's own terms, tagged `other` on HuggingFace, **not an
  OSI licence**. Quoted from the public GitHub `LICENSE`, since the model card sits behind a
  login and agreement wall. It permits commercial use and derivative works but adds terms
  Apache-2.0 does not: a litigation-termination clause, an indemnification obligation, a
  publication-attribution requirement, and prohibited-use categories (military, nuclear,
  weapons, espionage, ITAR).
- **Access is per HuggingFace account.** You must request and be granted access yourself
  before `gpu/stage_objects.py` can run at all.

### Hosted models called by API

`z-ai/glm-5.3-flash` (MIT), `nvidia/nemotron-3-nano-30b-a3b` (NVIDIA Open Model License),
`google/gemma-4-31b-it` and `google/gemma-4-26b-a4b-it-maas` (**UNKNOWN from this
repository** — Google's own Gemma and Vertex AI terms apply),
`qwen/qwen3-vl-30b-a3b-instruct` (Apache-2.0) and `microsoft/TRELLIS.2-4B` (MIT) for the MSA
scripts, plus `facebook/dinov3-vitl16-pretrain-lvd1689m` (**UNKNOWN** — Meta's custom DINOv3
licence, gated) which TRELLIS.2 loads internally. No weights for any of these are in this
repository.

---

## Bundled 3D assets — robot platforms

Eight robot models ship under `frontend/public/models/`, one `.glb` each, used only to draw
the selected platform in the 3D view. Every one is a **static, non-articulated** conversion:
the upstream meshes assembled at the URDF's own joint offsets, decimated, re-exported.
Each `.glb` has a sibling `<name>.LICENSE.txt` naming the upstream repository, the exact
commit, the exact mesh paths, and **reproducing the full licence text**. Read those files —
this table is a summary.

| File | Upstream | Commit / branch | Licence | Copyright |
|---|---|---|---|---|
| `turtlebot3_burger.glb` | [ROBOTIS-GIT/turtlebot3](https://github.com/ROBOTIS-GIT/turtlebot3) | `turtlebot3_description/meshes/` | Apache-2.0 | ROBOTIS CO., LTD. |
| `waffle_pi.glb` | [ROBOTIS-GIT/turtlebot3](https://github.com/ROBOTIS-GIT/turtlebot3) | `fc817ce`, fetched 2026-08-31 | Apache-2.0 | ROBOTIS CO., LTD. |
| `turtlebot4.glb` | [turtlebot/turtlebot4](https://github.com/turtlebot/turtlebot4) | `7fd29fb` (jazzy) + `a6ee13b` (humble) | Apache-2.0 | Clearpath Robotics / the `turtlebot4_description` authors |
| `rosbot_xl_arm.glb` | [husarion/rosbot_ros](https://github.com/husarion/rosbot_ros) + a second upstream for the arm | `41fad02` (main) | Apache-2.0 (both parts, two copyright holders) | Husarion and the arm's upstream |
| `husky.glb` | [husky/husky](https://github.com/husky/husky) | `41e15d2`, noetic-devel, 2025-10-01 | BSD-3-Clause | 2021 Clearpath Robotics Inc. |
| `jackal.glb` | [jackal/jackal](https://github.com/jackal/jackal) | `4ddf9b5`, 2024-05-24 | BSD-3-Clause | 2021 Clearpath Robotics Inc. |
| `limo.glb` | [agilexrobotics/limo_ros](https://github.com/agilexrobotics/limo_ros) | `4c78efc`, 2025-01-23 | BSD-3-Clause | 2021 Agilex Robotics |
| `go2.glb` | [unitreerobotics/unitree_ros](https://github.com/unitreerobotics/unitree_ros) | `7d6075f`, fetched 2026-08-31 | BSD-3-Clause | HangZhou YuShu TECHNOLOGY CO., LTD. ("Unitree Robotics"), 2016-2022 |

BSD-3-Clause requires the copyright notice and licence text to travel with redistributions;
that requirement is met by the per-model `.LICENSE.txt` files, which ship alongside each
`.glb`. No trademark or endorsement by any of these vendors is implied.

`go2.LICENSE.txt` records the diligence explicitly: no divergent per-file licence, NC term,
or ND term was found under `robots/go2_description/`, so the repository-root BSD-3-Clause
applies to those meshes.

---

## Bundled fonts

`frontend/public/fonts/` holds subset `.woff2` files for **Inter** and **JetBrains Mono**
(latin and cyrillic, weights 400–600).

- **Licence: SIL Open Font License 1.1**, for both families. The OFL requires its text to
  accompany redistribution, so the upstream licence text for each family ships beside the
  subsets, fetched verbatim on 2026-09-11 from the projects' own repositories:

  | Family | Licence file | Fetched from |
  |---|---|---|
  | Inter | `frontend/public/fonts/OFL-Inter.txt` | <https://raw.githubusercontent.com/rsms/inter/master/LICENSE.txt> (repo: <https://github.com/rsms/inter>) |
  | JetBrains Mono | `frontend/public/fonts/OFL-JetBrainsMono.txt` | <https://raw.githubusercontent.com/JetBrains/JetBrainsMono/master/OFL.txt> (repo: <https://github.com/JetBrains/JetBrainsMono>) |

  Copyright lines, as carried in those files: *Copyright (c) 2016 The Inter Project
  Authors* and *Copyright 2020 The JetBrains Mono Project Authors*.

- **Provenance of the subsets** is recorded in `frontend/public/fonts/README.md`: the Inter
  faces were copied verbatim from `@fontsource/inter@5.3.0` (whose own bundled `LICENSE` is
  the same OFL 1.1 text), and the JetBrains Mono faces were fetched from Google Fonts,
  which serves that family under the same licence. Neither family's outlines were modified
  or renamed — only subset — so no Reserved Font Name question arises.

---

## Furniture and small-object asset packs

Used by the MSA scripts (`scripts/msa/`), **not committed** — they live outside the
repository under `var/assets/`. Full provenance, including download dates, archived pages and
per-model lists, is in [docs/ASSET_ATTRIBUTION.md](docs/ASSET_ATTRIBUTION.md).

| Pack | Licence | Attribution required? |
|---|---|---|
| Kenney Furniture Kit | CC0 1.0 | No (credited anyway) |
| mastjie, "Low poly household goods" | CC0 (confirmed from the archived page text) | No (credited anyway) |
| Poly Haven furniture models | CC0 1.0 | No (credited anyway) |
| Google Scanned Objects (small-object tier) | **CC BY 4.0 — attribution REQUIRED** | **Yes**, wherever the models or renders containing them appear |

---

## Other bundled media

`frontend/public/landing/` (`hero-orbit.webm`, `isaac-demo.gif`),
`frontend/public/msa-fixtures/02_modular_home.glb` and `docs/evidence/floor-fix/*.png` are
this project's own output — renders and screenshots from its own pipeline — and are covered
by this repository's Apache-2.0 licence.

---

## Python and JavaScript dependencies

Declared in `pyproject.toml` / `uv.lock` and `frontend/package.json` /
`frontend/package-lock.json`, resolved from PyPI and npm at install time. None is vendored
into this repository. Their licences are whatever those lockfiles resolve to; run your own
audit (`uv pip list`, `npm ls --all`) if you need a complete SBOM — this project does not
ship one.
