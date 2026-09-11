import { act, renderHook } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { useCommandAnimation } from './useCommandAnimation'
import type { CommandResponse } from '../api/types'

// Named explicitly (rather than left to inference) so a literal `undefined`/`null`
// initialProps value below doesn't narrow the shared `renderHook` Props type parameter
// down to exactly `undefined`/`null` instead of this wider, correct shape.
type AnimHookProps = { sceneKey: string | undefined; pos: { x: number; z: number } | null }

// Regression test for a real bug: on first mount, the scene (and therefore the
// robot's start position) hasn't loaded yet, so this hook was always called once with
// `initialRobotPosition: null`. useState's initializer only runs on that first call,
// so `robotPosition` stayed null forever once the real coordinates arrived - the robot
// was invisible until the user clicked "reset robot to start" (the only other place
// that ever pushed a real position into state). See ScenePage.tsx's usage.

describe('useCommandAnimation', () => {
  it('starts with a null robotPosition when nothing is known yet', () => {
    const { result } = renderHook(() => useCommandAnimation(undefined, null))
    expect(result.current.state.robotPosition).toBeNull()
  })

  it('seeds robotPosition once real coordinates arrive after mount (the bug)', () => {
    const { result, rerender } = renderHook<ReturnType<typeof useCommandAnimation>, AnimHookProps>(
      ({ sceneKey, pos }) => useCommandAnimation(sceneKey, pos),
      { initialProps: { sceneKey: undefined, pos: null } },
    )
    expect(result.current.state.robotPosition).toBeNull()

    // Scene finishes loading - same async gap that exists in ScenePage.tsx between
    // first render and the getScene() response arriving.
    rerender({ sceneKey: 'scene-1', pos: { x: 1.5, z: -2.5 } })

    expect(result.current.state.robotPosition).toEqual({ x: 1.5, z: -2.5 })
  })

  it('does not clobber an in-progress position on a same-scene re-render (e.g. polling)', () => {
    const { result, rerender } = renderHook<ReturnType<typeof useCommandAnimation>, AnimHookProps>(
      ({ sceneKey, pos }) => useCommandAnimation(sceneKey, pos),
      { initialProps: { sceneKey: 'scene-1', pos: { x: 1, z: 1 } } },
    )
    expect(result.current.state.robotPosition).toEqual({ x: 1, z: 1 })

    // Simulate the robot having moved away from its start (a command finished).
    act(() => result.current.reset({ x: 9, z: 9 }))

    // A poll refresh for the SAME scene re-renders with a brand-new object reference
    // holding the identical start coordinates - must not snap the robot back.
    rerender({ sceneKey: 'scene-1', pos: { x: 1, z: 1 } })
    expect(result.current.state.robotPosition).toEqual({ x: 9, z: 9 })
  })

  it('re-seeds when navigating to a different scene without a full page reload', () => {
    const { result, rerender } = renderHook<ReturnType<typeof useCommandAnimation>, AnimHookProps>(
      ({ sceneKey, pos }) => useCommandAnimation(sceneKey, pos),
      { initialProps: { sceneKey: 'scene-1', pos: { x: 1, z: 1 } } },
    )
    act(() => result.current.reset({ x: 9, z: 9 }))
    expect(result.current.state.robotPosition).toEqual({ x: 9, z: 9 })

    // ScenePage isn't remounted on a sceneId route-param change - a different scene's
    // data lands as a normal re-render, not a fresh mount.
    rerender({ sceneKey: 'scene-2', pos: { x: 3, z: 4 } })
    expect(result.current.state.robotPosition).toEqual({ x: 3, z: 4 })
  })

  // The status badge says what the robot is doing WHILE it does it. It used to end on a
  // permanent "Done.", which then sat in the left column for the rest of the session and
  // was wrong the moment anything else happened.
  it('finishes with no label at all, so the status badge retires', async () => {
    const { result } = renderHook(() => useCommandAnimation('scene-1', { x: 0, z: 0 }))
    const command = {
      command_id: 'c1',
      scene_id: 'scene-1',
      user_text: 'goto desk',
      action: 'goto',
      status: 'done',
      steps: [
        { type: 'move', path: [[0, 0], [1, 0]], duration_sec: 0, object: null, position: null, at: null, length_m: null },
      ],
    } as unknown as CommandResponse
    await act(async () => {
      await result.current.play(command, [])
    })
    expect(result.current.state.label).toBeNull()
    expect(result.current.state.playing).toBe(false)
  })
})
