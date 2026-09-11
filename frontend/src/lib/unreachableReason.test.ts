import { describe, expect, it } from 'vitest'
import { hasGloss, unreachableReasonLabel, unreachableReasonTitle } from './unreachableReason'

// The exact strings app/services/pathfinding.py can put on the wire - see
// compute_reachability and app/schemas.py's ReachabilityResponse.unreachable_reasons.
const BACKEND_LABELS = [
  'outside_grid',
  'disconnected',
  'no_free_cell_within_2m',
  'no_visible_cell_within_2m',
  'robot_does_not_fit',
]

describe('unreachableReasonTitle', () => {
  it('explains every label the backend can send', () => {
    for (const label of BACKEND_LABELS) {
      expect(hasGloss(label)).toBe(true)
      expect(unreachableReasonTitle(label)).not.toBe(label)
    }
  })

  it('never invents a measurement', () => {
    for (const label of BACKEND_LABELS) {
      // The only number any gloss may carry is the 2 m the label itself names.
      const numbers = unreachableReasonTitle(label).match(/\d+(\.\d+)?/g) ?? []
      expect(numbers.every((n) => n === '2')).toBe(true)
    }
  })

  it('falls through to the raw label for one it does not know', () => {
    expect(hasGloss('some_future_label')).toBe(false)
    expect(unreachableReasonTitle('some_future_label')).toBe('some_future_label')
  })
})

// The one display relabel (2026-09-10): `disconnected` reads as a network fault on a
// screen where it only ever means "the robot cannot get there". Narrow on purpose - the
// raw word still reaches `data-object-reason` and the 2D map's hover label.
describe('unreachableReasonLabel', () => {
  it('shows `disconnected` as `unreachable`', () => {
    expect(unreachableReasonLabel('disconnected')).toBe('unreachable')
  })

  it('leaves every other backend label exactly as it came', () => {
    for (const label of BACKEND_LABELS) {
      if (label === 'disconnected') continue
      expect(unreachableReasonLabel(label)).toBe(label)
    }
  })

  it('passes an unknown label through rather than swallowing it', () => {
    expect(unreachableReasonLabel('some_new_backend_reason')).toBe('some_new_backend_reason')
  })
})
