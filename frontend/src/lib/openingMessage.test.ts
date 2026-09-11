import { describe, expect, it } from 'vitest'
import { openingMessage } from './openingMessage'
import type { OccupancyGridResponse, ReachabilityResponse } from '../api/types'

const grid = {
  resolution: 0.05,
  width: 111,
  height: 142,
  layer_available: true,
} as OccupancyGridResponse

const reach = (over: Partial<ReachabilityResponse>) =>
  ({ start_status: 'original', layer_available: true, ...over }) as ReachabilityResponse

// Templated from the scene's own numbers - no model behind it, no delay in front of it.
// Every branch is a state the screen can really be in, and none of them is written as an
// error, because none of them is one.
describe('openingMessage', () => {
  it('says the room size, what was found and what the robot can reach', () => {
    expect(
      openingMessage({
        grid,
        objectCount: 17,
        reachableCount: 13,
        platformName: 'TurtleBot3 Burger',
        reachability: reach({}),
      }),
    ).toBe(
      'This room is 5.6 × 7.1 m and 17 objects were detected in it. ' +
        'TurtleBot3 Burger can reach 13 of 17 - double-click one to drive there.',
    )
  })

  it('does not say "13 of 17" when it can reach all of them', () => {
    const msg = openingMessage({
      grid, objectCount: 17, reachableCount: 17,
      platformName: 'TurtleBot3 Burger', reachability: reach({}),
    })
    expect(msg).toContain('can reach all of them')
    expect(msg).not.toContain('17 of 17')
  })

  it('distinguishes "no layer" from "fits nowhere" - both are all-zero otherwise', () => {
    const noLayer = openingMessage({
      grid: { ...grid, layer_available: false, width: 0, height: 0 },
      objectCount: 17, reachableCount: null, platformName: 'Husky A200',
      reachability: reach({ layer_available: false }),
    })
    expect(noLayer).toContain('no navigation layer')

    const noStart = openingMessage({
      grid, objectCount: 17, reachableCount: 0, platformName: 'Husky A200',
      reachability: reach({ start_status: 'none' }),
    })
    expect(noStart).toContain('nowhere to stand')
    expect(noStart).not.toContain('no navigation layer')
  })

  it('says an empty room is empty rather than "0 of 0 reachable"', () => {
    const msg = openingMessage({
      grid, objectCount: 0, reachableCount: 0, platformName: 'TurtleBot3 Burger',
      reachability: reach({}),
    })
    expect(msg).toContain('nothing was detected in it')
    expect(msg).toContain('nothing to drive to')
  })

  it('says it is still working rather than claiming a count it does not have', () => {
    expect(
      openingMessage({
        grid, objectCount: 17, reachableCount: null,
        platformName: 'Unitree Go2', reachability: reach({}),
      }),
    ).toContain('Working out what Unitree Go2 can reach')
  })

  it('names the robot generically when no platform is selected yet', () => {
    expect(
      openingMessage({ grid, objectCount: 5, reachableCount: 5, platformName: null, reachability: reach({}) }),
    ).toContain('This robot can reach all of them')
  })

  it('omits a size it does not have rather than printing 0.0 × 0.0 m', () => {
    expect(
      openingMessage({
        grid: null, objectCount: 3, reachableCount: 3,
        platformName: 'TurtleBot3 Burger', reachability: reach({}),
      }),
    ).toBe('This room and 3 objects were detected in it. TurtleBot3 Burger can reach all of them - double-click one to drive there.')
  })
})
