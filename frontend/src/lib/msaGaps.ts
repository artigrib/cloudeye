/** Shape of scripts/msa/gaps.py's `gaps.json` output (SPEC.md §4.A5) - one entry per
 * measured passage between two obstacles (wall or object). Field names are kept
 * exactly as written on disk (including the `_xy` suffix on what is actually an
 * (x, z) floor-plane point in this project's Y-up world frame - see gaps.py:111). */
export interface MsaGapPlatform {
  platform_id: string
  fits: boolean
  required_clearance_m: number
}

export interface MsaGap {
  a: string
  a_kind: 'wall' | 'object'
  b: string
  b_kind: 'wall' | 'object'
  width_m: number
  measurement_point_xy: [number, number]
  platforms: MsaGapPlatform[]
}

/** A gap "passes" for the gaps layer's color/label if at least one registered
 * platform fits through it - matches bootstrap.py's own per-platform verdict model
 * (SPEC §4.A5: "pass/fail per platform"), so the layer reads as "usable by someone"
 * rather than an arbitrary single-platform pass/fail. */
export function gapPasses(gap: MsaGap): boolean {
  return gap.platforms.some((p) => p.fits)
}

export async function loadMsaGaps(url: string): Promise<MsaGap[]> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`Failed to load gaps.json: ${res.status}`)
  return (await res.json()) as MsaGap[]
}
