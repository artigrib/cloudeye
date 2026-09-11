import { describe, expect, it } from 'vitest'
import { translateSceneError } from './sceneErrorMessage'

describe('translateSceneError', () => {
  const cases: Array<[string, string]> = [
    [
      'pipeline failed at stage \'keyframes\': only 2 usable keyframes extracted from the input video - too few for a 3D reconstruction. Check the source video isn\'t corrupt, too short, or uniformly blurry.',
      'All frames were rejected as blurry or too uniform. Try filming slower and closer to the furniture.',
    ],
    [
      "pipeline failed at stage 'infer': only 2 views loaded - too few for reconstruction",
      'All frames were rejected as blurry or too uniform. Try filming slower and closer to the furniture.',
    ],
    [
      'scene failed validation: height_prior_majority: 4/6 typed objects violate their height prior (67%)',
      "Couldn't reliably determine where the floor is. This often happens with glossy floors or empty rooms.",
    ],
    [
      "pipeline failed at stage 'align': DBSCAN found no cluster at all in the pooled point cloud - reconstruction likely failed upstream (stage_infer), not something to work around here",
      "Couldn't reliably determine where the floor is. This often happens with glossy floors or empty rooms.",
    ],
    [
      'scene failed validation: room_dimensions: footprint 0.90x0.80m (expect 1.5-30.0m each), ceiling 2.40m (expect 2.0-5.0m)',
      "The resulting geometry looks implausible. Try reshooting while walking the room's perimeter.",
    ],
    [
      'GPU unreachable after 3 attempts',
      'Processing did not finish. Please try again.',
    ],
    [
      "job exceeded JOB_TIMEOUT_SEC (1800s) at stage 'infer'",
      'Processing did not finish. Please try again.',
    ],
    [
      'command exited 1: ssh gpu run.sh\nCUDA out of memory',
      'Processing did not finish. Please try again.',
    ],
  ]

  it.each(cases)('maps %s', (raw, expected) => {
    expect(translateSceneError(raw)).toBe(expected)
  })

  it('falls back to the original text for an unrecognized error', () => {
    const raw = 'something completely unexpected happened'
    expect(translateSceneError(raw)).toBe(raw)
  })

  it('falls back to a generic message when there is no error text at all', () => {
    expect(translateSceneError(null)).toBe('Reconstruction failed.')
    expect(translateSceneError(undefined)).toBe('Reconstruction failed.')
    expect(translateSceneError('')).toBe('Reconstruction failed.')
  })
})
