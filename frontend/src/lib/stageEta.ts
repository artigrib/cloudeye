// Stage-by-stage ETA for a scene still `processing`, using the real per-stage
// duration shares measured from the 2026-09-05 nightly regression run (see
// docs/DECISIONS.md's "Job-timeline UI" entry and
// var/scratch/autopilot-20260905/isaac/regression_20260905_report/results.json,
// 6 scenes that reached `done`, each carrying a full backend `stage_timings` list).
// These are NOT illustrative numbers - they're each stage's share of total measured
// wall-clock time, summed across those 6 runs' stage durations and normalized to 1.
//
// `finalize` has no measurable duration in that data (it's always the last stage
// *observed*, with no next transition to compute a delta against - see
// app/services/stage_timings.py's parse_stage_timings docstring), so it carries no
// fraction of its own here; reaching it means the job is essentially done.
import type { StageTimingResponse } from '../api/types'

export const STAGE_ORDER = [
  'keyframes',
  'vocab',
  'infer',
  'align',
  'objects',
  'occupancy',
  'export_glb',
  'finalize',
] as const

export type PipelineStage = (typeof STAGE_ORDER)[number]

/** Each stage's share of total measured wall-clock time (sums to 1 across the 7
 * stages with measured duration). Source: regression_20260905_report/results.json. */
export const STAGE_FRACTIONS: Readonly<Record<string, number>> = Object.freeze({
  keyframes: 0.0449,
  vocab: 0.0301,
  infer: 0.3882,
  align: 0.075,
  objects: 0.3945,
  occupancy: 0.0298,
  export_glb: 0.0375,
  finalize: 0,
})

/** Mean total pipeline duration (seconds) across the same 6 done scenes - used only
 * as a rough fallback ETA before any stage has fully finished (so the UI isn't blank
 * for the ~15s of the first stage). Real scenes vary a lot around this (276s-568s in
 * that run), so it's clearly labeled `isRough` by estimateEta below. */
export const MEAN_TOTAL_DURATION_SEC = 358

export interface EtaEstimate {
  /** Seconds since the job's first observed stage started. */
  elapsedSec: number
  /** Name of the currently running (last observed) stage. */
  currentStage: string
  /** Sum of STAGE_FRACTIONS for every stage that has already finished (duration_sec
   * set) - does not credit any partial progress within the still-running stage. */
  completedFraction: number
  /** Estimated remaining seconds, or null if not enough data yet (still in the very
   * first stage, nothing finished to extrapolate from - falls back to
   * MEAN_TOTAL_DURATION_SEC instead, with isRough=true). */
  remainingSec: number | null
  /** True when remainingSec comes from the MEAN_TOTAL_DURATION_SEC fallback rather
   * than this job's own observed progress. */
  isRough: boolean
}

/** Estimate remaining time from a scene's live `stage_timings` (see
 * SceneStatusResponse). Returns null if there's no timing data at all yet (job just
 * started, no stage line written).
 *
 * Method: sum the measured fraction-of-total-time for every stage that has already
 * fully finished in this run, then extrapolate: estimated_total = elapsed /
 * completed_fraction. This only uses *this job's own* elapsed time plus the
 * historical fractions - it doesn't assume this job runs at the same absolute speed
 * as the regression run, so it stays accurate for scenes that are faster/slower than
 * average once at least one stage has completed. */
export function estimateEta(stageTimings: StageTimingResponse[], now: Date = new Date()): EtaEstimate | null {
  if (stageTimings.length === 0) return null

  const firstStartedAt = new Date(stageTimings[0].started_at).getTime()
  const elapsedSec = Math.max(0, (now.getTime() - firstStartedAt) / 1000)
  const current = stageTimings[stageTimings.length - 1]

  const completedFraction = stageTimings.reduce((sum, t) => {
    if (t.duration_sec == null) return sum // the still-running stage - not counted
    return sum + (STAGE_FRACTIONS[t.stage] ?? 0)
  }, 0)

  if (completedFraction > 0) {
    const estimatedTotalSec = elapsedSec / completedFraction
    return {
      elapsedSec,
      currentStage: current.stage,
      completedFraction,
      remainingSec: Math.max(0, estimatedTotalSec - elapsedSec),
      isRough: false,
    }
  }

  // Still in the first stage - nothing finished to extrapolate from yet. Fall back to
  // the historical mean total duration so the UI has *something* to show.
  return {
    elapsedSec,
    currentStage: current.stage,
    completedFraction: 0,
    remainingSec: Math.max(0, MEAN_TOTAL_DURATION_SEC - elapsedSec),
    isRough: true,
  }
}

/** "~2m 30s" / "~45s" style compact duration, for the ETA/elapsed display. */
export function formatDurationCompact(sec: number): string {
  const total = Math.round(Math.max(0, sec))
  const m = Math.floor(total / 60)
  const s = total % 60
  if (m === 0) return `${s}s`
  return `${m}m ${s}s`
}

/** A human label for a pipeline stage name, falling back to the raw string for any
 * stage not in STAGE_ORDER (forward-compatible with a pipeline stage added later). */
const STAGE_LABELS: Readonly<Record<string, string>> = Object.freeze({
  keyframes: 'Extracting keyframes',
  vocab: 'Detecting object vocabulary',
  infer: 'Reconstructing geometry',
  align: 'Aligning to floor',
  objects: 'Segmenting objects',
  occupancy: 'Building occupancy map',
  export_glb: 'Exporting mesh',
  finalize: 'Finishing up',
})

export function stageLabel(stage: string): string {
  return STAGE_LABELS[stage] ?? stage
}
