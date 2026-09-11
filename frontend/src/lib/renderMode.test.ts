import { beforeEach, describe, expect, it } from 'vitest'
import { DEFAULT_RENDER_MODE, loadRenderMode, saveRenderMode } from './renderMode'

beforeEach(() => {
  localStorage.clear()
})

describe('renderMode persistence', () => {
  it('defaults to clearance when nothing is stored', () => {
    expect(loadRenderMode('scene-a')).toBe('clearance')
    expect(DEFAULT_RENDER_MODE).toBe('clearance')
  })

  it('round-trips a saved mode', () => {
    saveRenderMode('scene-a', 'occupancy')
    expect(loadRenderMode('scene-a')).toBe('occupancy')
  })

  it('is scoped per scene id', () => {
    saveRenderMode('scene-a', 'height')
    expect(loadRenderMode('scene-b')).toBe('clearance')
  })

  it('rejects an unrecognized mode rather than trusting stale/tampered storage', () => {
    localStorage.setItem('cloudeye:renderMode:scene-a', 'thermal')
    expect(loadRenderMode('scene-a')).toBe('clearance')
  })
})
