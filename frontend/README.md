## Regenerating the robot model

`public/models/turtlebot3_burger.glb` is a static, non-articulated TurtleBot3 Burger model
used by the 3D scene view (`src/components/SceneMap3D.tsx`). It's built once, offline, and
committed - the app never parses URDF or loads individual meshes at runtime. See
`public/models/turtlebot3_burger.LICENSE.txt` for the source and Apache 2.0 attribution.

To regenerate it (e.g. after an upstream mesh update):

1. Clone [ROBOTIS-GIT/turtlebot3](https://github.com/ROBOTIS-GIT/turtlebot3) and locate
   `turtlebot3_description/` (it contains `meshes/` and `urdf/turtlebot3_burger.urdf`).
2. From `frontend/`, run:
   ```sh
   node scripts/build-turtlebot-glb.mjs <path-to-turtlebot3_description>
   ```
   This assembles the base, both wheels, and the LiDAR puck (STL, millimetres) into one
   scene-ready mesh: scaled to metres, positioned per the URDF's part offsets, and rotated
   from ROS's Z-up into the scene's Y-up. It asserts the resulting bounding box against the
   real robot's known dimensions (137.5 x 178.2 x 191.1 mm) before writing
   `scripts/out/raw.glb`, and fails loudly if a regeneration silently drops the mm->m scale
   or the axis convention.
3. Decimate and quantize it (raw STL geometry is far more tessellated than a ~14cm object
   needs at a few hundred screen pixels; quantization needs no runtime decoder because
   three's `GLTFLoader` supports `KHR_mesh_quantization` natively - do **not** Draco/meshopt
   compress this model, no decoder is wired into the app):
   ```sh
   cd scripts/out
   npx --yes @gltf-transform/cli@4 weld     raw.glb  w.glb
   npx --yes @gltf-transform/cli@4 simplify w.glb    s.glb --ratio 0.10 --error 0.005
   npx --yes @gltf-transform/cli@4 quantize s.glb    q.glb
   npx --yes @gltf-transform/cli@4 dedup    q.glb    turtlebot3_burger.glb
   ```
   This produced a ~29k-triangle, ~400KB `.glb` from the source's 154k triangles, with the
   bounding box unchanged to within a millimetre.
4. Copy the result to `public/models/turtlebot3_burger.glb`, confirm it still loads (`npm run
   dev`, `?view=3d`), and commit it.

`scripts/out/` is gitignored - only the final `.glb` in `public/models/` is checked in.

# React + TypeScript + Vite

This template provides a minimal setup to get React working in Vite with HMR and some Oxlint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Oxc](https://oxc.rs)
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/)

## React Compiler

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the Oxlint configuration

If you are developing a production application, we recommend enabling type-aware lint rules by installing `oxlint-tsgolint` and editing `.oxlintrc.json`:

```json
{
  "$schema": "./node_modules/oxlint/configuration_schema.json",
  "plugins": ["react", "typescript", "oxc"],
  "options": {
    "typeAware": true
  },
  "rules": {
    "react/rules-of-hooks": "error",
    "react/only-export-components": ["warn", { "allowConstantExport": true }]
  }
}
```

See the [Oxlint rules documentation](https://oxc.rs/docs/guide/usage/linter/rules) for the full list of rules and categories.
