import { beforeEach, describe, expect, it } from 'vitest'
import { DEFAULT_POINT_CLOUD_VIEW_SETTINGS, loadPointCloudViewSettings, savePointCloudViewSettings } from './pointcloudViewSettings'

beforeEach(() => {
  localStorage.clear()
})

describe('loadPointCloudViewSettings', () => {
  it('returns defaults when nothing is stored', () => {
    expect(loadPointCloudViewSettings('scene-a')).toEqual(DEFAULT_POINT_CLOUD_VIEW_SETTINGS)
  })

  it('round-trips a saved value', () => {
    savePointCloudViewSettings('scene-a', { ceilingMode: 'fade', heightCutoff: 1.5 })
    expect(loadPointCloudViewSettings('scene-a')).toEqual({ ceilingMode: 'fade', heightCutoff: 1.5 })
  })

  it('keeps settings scoped per scene id - a different scene sees defaults', () => {
    savePointCloudViewSettings('scene-a', { ceilingMode: 'hidden', heightCutoff: 1.2 })
    expect(loadPointCloudViewSettings('scene-b')).toEqual(DEFAULT_POINT_CLOUD_VIEW_SETTINGS)
  })

  it('falls back to null heightCutoff when stored value is not a number', () => {
    localStorage.setItem('cloudeye:pointcloudView:scene-a', JSON.stringify({ ceilingMode: 'hidden', heightCutoff: 'nope' }))
    expect(loadPointCloudViewSettings('scene-a').heightCutoff).toBeNull()
  })

  it('falls back to defaults on corrupt JSON rather than throwing', () => {
    localStorage.setItem('cloudeye:pointcloudView:scene-a', '{not json')
    expect(loadPointCloudViewSettings('scene-a')).toEqual(DEFAULT_POINT_CLOUD_VIEW_SETTINGS)
  })

  it('rejects an unrecognized ceilingMode rather than trusting stale/tampered storage', () => {
    localStorage.setItem('cloudeye:pointcloudView:scene-a', JSON.stringify({ ceilingMode: 'gone' }))
    expect(loadPointCloudViewSettings('scene-a').ceilingMode).toBe('visible')
  })
})
