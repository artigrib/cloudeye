import type { ProcessingStageResponse, ProcessingStatusResponse } from '../api/types'

/** `substage`'s closed vocabulary, from pipeline-v1's `run_pipeline.SUBSTAGES`. Used to
 * decide presentation, never to filter: a value not in here is still shown, as itself. */
export const SUBSTAGES = [
  'searching',
  'creating',
  'provisioning',
  'waiting_capacity',
  'transfer_gate',
  'refused',
  'waiting_manual',
  'skipped',
] as const

/** The two that get a banner instead of a row: one says the run was turned away, the
 * other says a human has to do something. Neither may scroll past unnoticed. */
export const SUBSTAGES_NEEDING_ATTENTION = ['refused', 'waiting_manual'] as const

/** Local ISO-8601 with an offset - what pipeline-v1 writes (docs/JOB_SPEC.md section 2).
 * Returns null rather than NaN for anything unparseable, so a caller can omit the row
 * instead of rendering "Invalid Date". */
export function parseStamp(iso: string | null | undefined): number | null {
  if (!iso) return null
  const ms = new Date(iso).getTime()
  return Number.isNaN(ms) ? null : ms
}

/** "4s", "1m 12s", "2h 05m". Whole seconds under a minute: this counts a wait, and a
 * tenth of a second on a four-minute wait is noise pretending to be precision. */
export function formatElapsed(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000))
  if (total < 60) return `${total}s`
  if (total < 3600) return `${Math.floor(total / 60)}m ${String(total % 60).padStart(2, '0')}s`
  return `${Math.floor(total / 3600)}h ${String(Math.floor((total % 3600) / 60)).padStart(2, '0')}m`
}

/** How long the CURRENT SUBSTAGE has been running, from `since` - not from `started`.
 * The two differ exactly when a step is waiting, which is the case worth showing.
 * null when the writer recorded no `since`, or when it is unparseable, or when it is in
 * the future: a clock that has not reached its own start is a fact about the clock, and
 * "-3s" would be the screen guessing which of the two is wrong. */
export function elapsedSince(since: string | null | undefined, nowMs: number): string | null {
  const started = parseStamp(since)
  if (started === null || started > nowMs) return null
  return formatElapsed(nowMs - started)
}

/** The substage line: "waiting_capacity · no box available on attempt 4". One line, one
 * separator, and each half is optional - a substage with no detail is still worth
 * showing, and a detail with no substage is what an older writer produces. null when
 * neither is present, and the caller then draws nothing at all. */
export function substageLine(row: ProcessingStageResponse): string | null {
  const parts = [row.substage, row.detail].filter((p): p is string => !!p && p.trim() !== '')
  return parts.length ? parts.join(' · ') : null
}

/** "attempt 4", plus "· retry at 13:51:10" when a retry time is known.
 *
 * null for attempt 0 and for attempt absent alike, though they arrive differently:
 * `Status.enter` writes 0 before any retry loop starts counting, and 0 attempts is not an
 * attempt. This is the edge that decides that - the server passes the zero through
 * deliberately (see its own comment) so the decision is made once, here, where it is
 * about rendering. */
export function attemptLine(row: ProcessingStageResponse): string | null {
  if (typeof row.attempt !== 'number' || row.attempt <= 0) return null
  const retryAt = parseStamp(row.next_retry)
  if (retryAt === null) return `attempt ${row.attempt}`
  const clock = new Date(retryAt).toLocaleTimeString(undefined, {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })
  return `attempt ${row.attempt} · retry at ${clock}`
}

export interface AttentionBanner {
  kind: 'refused' | 'waiting_manual' | 'fallback'
  title: string
  /** The writer's own words, verbatim. Never a paraphrase and never a placeholder: a
   * banner with nothing to say is not rendered. */
  detail: string | null
  /** The stage this is about, for the two that come from a step. */
  stage?: string
}

/** Every banner the screen owes the user, in the order it should show them.
 *
 * A banner and not a row, for three cases: the run was refused, the run is waiting on a
 * person, or the provider that ran is not the provider that was asked for. Each is a
 * thing someone has to act on, and a row in a fifteen-line feed is not where that
 * belongs. */
export function attentionBanners(status: ProcessingStatusResponse): AttentionBanner[] {
  const banners: AttentionBanner[] = []

  for (const row of status.stages) {
    if (row.substage === 'refused') {
      banners.push({ kind: 'refused', title: 'Refused', detail: row.detail ?? null, stage: row.stage })
    } else if (row.substage === 'waiting_manual') {
      banners.push({
        kind: 'waiting_manual',
        title: 'Waiting for you',
        detail: row.detail ?? null,
        stage: row.stage,
      })
    }
  }

  // The provider notice fires only when a provider is REPORTED and differs from the one
  // the spec asked for - JOB_SPEC section 2's rule, and the reason nothing fires on a run
  // that reports no provider at all, which is every run today.
  if (status.provider_used && status.requested_provider && status.provider_used !== status.requested_provider) {
    banners.push({
      kind: 'fallback',
      title: `Ran on ${status.provider_used}, not ${status.requested_provider}`,
      detail: status.fallback_reason ?? null,
    })
  }

  return banners
}

/** Which stage the feed should highlight: the one the run is in.
 *
 * Taken from the top-level `state`, which pipeline-v1 sets on entering a step, rather
 * than by hunting for a row whose state is "running" - on a terminal document there is no
 * running row, and on a failed one the failing step's state is the failure, not
 * "running". null once the run has reached a terminal state: DONE and FAILED are not
 * stages and there is no row to highlight. */
export function currentStage(status: ProcessingStatusResponse): string | null {
  if (!status.state || status.is_done || status.is_failed) return null
  return status.stages.some((s) => s.stage === status.state) ? status.state : null
}
