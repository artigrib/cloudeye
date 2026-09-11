// Pure lookup backing the video-player sync half of the camera-path feature - kept
// separate from the rendering components so the "nearest keyframe" rule is
// unit-testable without mounting anything.

import type { CameraPoseResponse } from '../api/types'

/**
 * Index into `poses` of whichever has a timestamp closest to `timeSec` - nearest-match,
 * never interpolated, per spec (keyframes are discrete moments, not a continuous track).
 * Ties break toward the earlier pose. Returns null when `poses` is empty or none of them
 * have a known timestamp (see CameraPoseResponse.timestamp_sec) - nothing to sync to.
 */
export function findNearestPoseIndex(poses: CameraPoseResponse[], timeSec: number): number | null {
  let bestIndex: number | null = null
  let bestDist = Infinity
  for (let i = 0; i < poses.length; i++) {
    const ts = poses[i].timestamp_sec
    if (ts === null) continue
    const dist = Math.abs(ts - timeSec)
    if (dist < bestDist) {
      bestDist = dist
      bestIndex = i
    }
  }
  return bestIndex
}
