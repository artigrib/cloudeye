/// <reference types="node" />
// tsconfig.app.json restricts ambient "types" to vite/client (the rest of src/ is
// browser-only code) - the reference above pulls in @types/node for this one test file
// so `node:fs` resolves under `tsc -b`, without loosening the app-wide config.
//
// A `?raw` import (`import css from '../index.css?raw'`) would be the more idiomatic
// Vite way to pull this file in, but under this project's vitest setup CSS `?raw`
// imports resolve to an empty string in the test (SSR-transform) module graph - true
// for any .css file, not just this Tailwind entry - so plain node:fs it is.
import { readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'
import {
  BG,
  BORDER,
  BORDER_STRONG,
  CANVAS,
  DATA_OBJECT,
  DATA_ROBOT,
  DATA_TRAJECTORY,
  DATA_UNREACHABLE,
  FAINT,
  FG,
  H_0,
  H_1,
  H_2,
  H_3,
  H_4,
  MUTED,
  OCC_FREE,
  OCC_OBSTACLE,
  OCC_UNKNOWN,
  SUBTLE,
  SURFACE,
  SURFACE_RAISED,
  oklchToSrgb,
  toHex,
  toRgba,
  type OklchColor,
} from './tokens'

const __dirname = dirname(fileURLToPath(import.meta.url))
const css = readFileSync(resolve(__dirname, '../index.css'), 'utf-8')

/** Pulls `--name: oklch(L% C H ...)` out of index.css's :root block - tolerant of the
 * `none` hue-less form (h=0) so OCC_OBSTACLE/OCC_UNKNOWN/DATA_ROBOT still match. */
function cssOklch(name: string): OklchColor {
  const re = new RegExp(`--${name}:\\s*oklch\\(([\\d.]+)%\\s+([\\d.]+)\\s+(none|[\\d.]+)`)
  const match = css.match(re)
  if (!match) throw new Error(`--${name} not found in index.css`)
  const [, l, c, h] = match
  return { l: Number(l) / 100, c: Number(c), h: h === 'none' ? 0 : Number(h) }
}

describe('data tokens match index.css', () => {
  it.each([
    ['data-object', DATA_OBJECT],
    ['data-trajectory', DATA_TRAJECTORY],
    ['data-robot', DATA_ROBOT],
    ['data-unreachable', DATA_UNREACHABLE],
    ['h-0', H_0],
    ['h-1', H_1],
    ['h-2', H_2],
    ['h-3', H_3],
    ['h-4', H_4],
    ['occ-free', OCC_FREE],
    ['occ-obstacle', OCC_OBSTACLE],
    ['occ-unknown', OCC_UNKNOWN],
    ['bg', BG],
    ['surface', SURFACE],
    ['surface-raised', SURFACE_RAISED],
    ['canvas', CANVAS],
    ['fg', FG],
    ['subtle', SUBTLE],
    ['muted', MUTED],
    ['faint', FAINT],
    ['border', BORDER],
    ['border-strong', BORDER_STRONG],
  ] as const)('%s', (name, token) => {
    expect(token).toEqual(cssOklch(name))
  })
})

describe('oklchToSrgb', () => {
  it('maps white (L=1, C=0) to full-white sRGB', () => {
    const [r, g, b] = oklchToSrgb({ l: 1, c: 0, h: 0 })
    expect(r).toBeCloseTo(1, 2)
    expect(g).toBeCloseTo(1, 2)
    expect(b).toBeCloseTo(1, 2)
  })

  it('maps black (L=0, C=0) to full-black sRGB', () => {
    const [r, g, b] = oklchToSrgb({ l: 0, c: 0, h: 0 })
    expect(r).toBeCloseTo(0, 2)
    expect(g).toBeCloseTo(0, 2)
    expect(b).toBeCloseTo(0, 2)
  })

  it('clamps out-of-gamut linear values instead of wrapping/NaN-ing', () => {
    const [r, g, b] = oklchToSrgb({ l: 0.7, c: 0.4, h: 30 })
    for (const channel of [r, g, b]) {
      expect(channel).toBeGreaterThanOrEqual(0)
      expect(channel).toBeLessThanOrEqual(1)
      expect(Number.isNaN(channel)).toBe(false)
    }
  })
})

describe('toRgba / toHex', () => {
  it('toRgba carries alpha through, defaulting to opaque', () => {
    expect(toRgba({ l: 0, c: 0, h: 0 })).toEqual([0, 0, 0, 255])
    expect(toRgba({ l: 0, c: 0, h: 0, a: 0.5 })).toEqual([0, 0, 0, 128])
  })

  it('toHex packs the same channels toRgba does', () => {
    const token: OklchColor = { l: 0.8, c: 0.16, h: 85 }
    const [r, g, b] = toRgba(token)
    expect(toHex(token)).toBe((r << 16) | (g << 8) | b)
  })
})
