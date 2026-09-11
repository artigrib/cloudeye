// Maps the backend's raw (English) scene.error_message to a human, readable explanation
// with a next step. Substring matching against the actual exception text raised by the
// GPU pipeline (see gpu/stage_keyframes.py, gpu/stage_infer.py, gpu/stage_align.py,
// app/services/scene_validator.py, app/services/gpu_client.py on the backend) - the
// backend itself is not localized, so this stays purely a frontend concern.

interface ErrorRule {
  patterns: string[]
  message: string
}

const RULES: ErrorRule[] = [
  {
    // stage_keyframes.py: "only N usable keyframes extracted ... too few for a 3D
    // reconstruction"; stage_infer.py: "only N views loaded - too few for reconstruction"
    patterns: ['usable keyframes extracted', 'too few for a 3d reconstruction', 'too few for reconstruction', 'views loaded'],
    message: "All frames were rejected as blurry or too uniform. Try filming slower and closer to the furniture.",
  },
  {
    // scene_validator.py checks that depend on where the floor/ceiling landed, plus
    // stage_align.py's "no cluster found" (upstream symptom of the same problem).
    patterns: [
      'floor_below_cameras',
      'ceiling_above_cameras',
      'height_prior_majority',
      'relative_order_counter_below_wall',
      'dbscan found no cluster',
    ],
    message:
      "Couldn't reliably determine where the floor is. This often happens with glossy floors or empty rooms.",
  },
  {
    // scene_validator.py's room_dimensions check; stage_occupancy.py's empty-cloud case.
    patterns: ['room_dimensions', 'zero points - nothing to grid'],
    message: "The resulting geometry looks implausible. Try reshooting while walking the room's perimeter.",
  },
  {
    // gpu_client.py (SSH/GPU-host errors), pipeline_orchestrator.py's timeout, and
    // missing-artifact/infra errors that have nothing to do with how the video was shot.
    patterns: [
      'gpu unreachable',
      'command exited',
      'job_timeout_sec',
      'ssh',
      'artifact missing',
      'video record missing',
      'could not read status.json',
    ],
    message: 'Processing did not finish. Please try again.',
  },
]

const FALLBACK = 'Reconstruction failed.'

/** Human-readable explanation for a scene's raw error_message, with a fallback to the
 * original text for causes we don't yet recognize. */
export function translateSceneError(raw: string | null | undefined): string {
  if (!raw) return FALLBACK
  const lower = raw.toLowerCase()
  for (const rule of RULES) {
    if (rule.patterns.some((p) => lower.includes(p))) return rule.message
  }
  return raw
}
