import { describe, expect, it } from 'vitest'
import { findNearestPoseIndex } from './cameraTrack'
import type { CameraPoseResponse } from '../api/types'

function pose(timestampSec: number | null): CameraPoseResponse {
  return { x: 0, y: 0, z: 0, timestamp_sec: timestampSec }
}

describe('findNearestPoseIndex', () => {
  it('picks the closest timestamp', () => {
    const poses = [pose(0), pose(1.5), pose(4)]
    expect(findNearestPoseIndex(poses, 1.6)).toBe(1)
    expect(findNearestPoseIndex(poses, 0.2)).toBe(0)
    expect(findNearestPoseIndex(poses, 10)).toBe(2)
  })

  it('breaks an exact tie toward the earlier pose', () => {
    const poses = [pose(1), pose(3)]
    expect(findNearestPoseIndex(poses, 2)).toBe(0)
  })

  it('skips poses with no known timestamp', () => {
    const poses = [pose(null), pose(5), pose(null)]
    expect(findNearestPoseIndex(poses, 5.1)).toBe(1)
  })

  it('returns null when no pose has a timestamp', () => {
    const poses = [pose(null), pose(null)]
    expect(findNearestPoseIndex(poses, 1)).toBeNull()
  })

  it('returns null for an empty track', () => {
    expect(findNearestPoseIndex([], 1)).toBeNull()
  })
})
