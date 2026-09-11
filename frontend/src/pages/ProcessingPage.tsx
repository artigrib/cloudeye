import { useEffect, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { getProcessingStatus } from '../api/workspaces'
import type { ProcessingStatusResponse } from '../api/types'
import {
  attemptLine,
  attentionBanners,
  currentStage,
  elapsedSince,
  substageLine,
} from '../lib/progressRows'

/** How often the screen asks the server what is happening. Spec: 5 s. */
const POLL_MS = 5000

/** How often the elapsed clocks re-render. A waiting step's "3m 41s" has to advance
 * between polls or it reads as a frozen screen; the value itself still comes from the
 * server's `since`, this only re-reads the local clock against it. */
const TICK_MS = 1000

/** Local ISO-8601 with an offset (what pipeline-v1 writes), or worker.log's naive local
 * timestamps - both parse; anything else is shown verbatim rather than as "Invalid Date". */
function clock(iso: string | null | undefined): string {
  if (!iso) return '—'
  const d = new Date(iso)
  return Number.isNaN(d.getTime())
    ? iso
    : d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

function duration(sec: number | null | undefined): string {
  if (sec == null) return '—'
  return sec < 60 ? `${sec.toFixed(1)}s` : `${Math.floor(sec / 60)}m ${Math.round(sec % 60)}s`
}

/** Live pipeline progress for one workspace.
 *
 * Every value on screen comes from the server on each poll and NOTHING is kept here
 * between them - no localStorage, no accumulated log buffer, no optimistic rows. That is
 * deliberate and it is what the acceptance probe checks: a reload at any moment, and a
 * second tab on the same URL, must render the identical feed, which can only be true if
 * the feed is a pure function of what the server just returned.
 *
 * It also means the screen never shows a stage nobody observed. When neither status.json
 * nor worker.log has a record it says so, in words.
 */
export default function ProcessingPage() {
  const { workspaceId } = useParams<{ workspaceId: string }>()
  const [status, setStatus] = useState<ProcessingStatusResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  // Read on every tick so the elapsed clocks advance. Held in state rather than read
  // during render so the numbers are the same for every row of one paint.
  const [nowMs, setNowMs] = useState(() => Date.now())
  const navigate = useNavigate()

  useEffect(() => {
    const timer = setInterval(() => setNowMs(Date.now()), TICK_MS)
    return () => clearInterval(timer)
  }, [])

  useEffect(() => {
    if (!workspaceId) return
    let cancelled = false

    async function poll() {
      try {
        const next = await getProcessingStatus(workspaceId!)
        if (!cancelled) {
          setStatus(next)
          setError(null)
        }
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : 'Could not reach the server.')
      }
    }

    void poll()
    const timer = setInterval(poll, POLL_MS)
    return () => {
      cancelled = true
      clearInterval(timer)
    }
  }, [workspaceId])

  // DONE -> the scene. Done as an effect off server state rather than inside the poll, so
  // it fires identically on a fresh load of an already-finished job.
  useEffect(() => {
    if (status?.is_done && status.scene_id) {
      navigate(`/workspaces/${status.workspace_id}/scenes/${status.scene_id}`, { replace: true })
    }
  }, [status?.is_done, status?.scene_id, status?.workspace_id, navigate, status])

  const stages = status?.stages ?? []
  const current = status ? currentStage(status) : null
  const banners = status ? attentionBanners(status) : []

  return (
    <div className="mx-auto max-w-3xl px-8 py-8" data-testid="processing-page">
      <header className="mb-5">
        <h1 className="text-[19px] font-semibold tracking-tight text-fg">
          {status?.workspace_name ?? 'Workspace'}
        </h1>
        <p className="mt-1 text-sm text-muted">Processing</p>
      </header>

      {error && (
        <p className="mb-4 text-sm text-data-unreachable" data-testid="processing-error">
          {error}
        </p>
      )}

      {/* The status line. Never a bare spinner: with no records at all, or none for a
         minute, the honest answer is that nothing is running, and saying "working…"
         would be a claim nothing on this box supports. */}
      <div
        className={`mb-5 rounded-md border-hair px-4 py-3 text-sm ${
          status?.is_failed
            ? 'border-status-blocked/40 bg-status-blocked/10 text-status-blocked'
            : status?.worker_connected
              ? 'border-border bg-surface text-subtle'
              : 'border-status-partial/40 bg-status-partial/10 text-status-partial'
        }`}
        data-testid="processing-status"
      >
        {status === null
          ? 'Loading…'
          : status.is_failed
            ? `Failed at ${status.state ?? 'an unknown stage'}`
            : status.worker_connected
              ? `Running · ${status.state ?? 'starting'}`
              : 'Worker not connected — no pipeline activity has been recorded for this workspace'}
      </div>

      {/* Banners, not rows: a refusal, a run waiting on a person, and a provider that is
         not the one that was asked for are each something somebody has to act on, and a
         line in a fifteen-row feed is not where that belongs. Each shows the writer's own
         words; one with nothing to say shows its title alone rather than a placeholder. */}
      {banners.map((banner) => (
        <div
          key={`${banner.kind}:${banner.stage ?? ''}`}
          data-testid={`banner-${banner.kind}`}
          data-banner-stage={banner.stage}
          className={`mb-s3 rounded-card border-hair px-s4 py-s2 text-bodyux ${
            banner.kind === 'refused'
              ? 'border-status-blocked/40 bg-status-blocked/10 text-status-blocked'
              : 'border-status-partial/40 bg-status-partial/10 text-status-partial'
          }`}
        >
          <span className="font-medium">
            {banner.title}
            {banner.stage ? ` · ${banner.stage}` : ''}
          </span>
          {banner.detail && <span className="ml-s2 opacity-80">{banner.detail}</span>}
        </div>
      ))}

      {status?.is_failed && status.error && (
        <pre
          className="mb-4 max-h-64 overflow-auto whitespace-pre-wrap rounded-md border-hair border-border bg-canvas px-4 py-3 font-mono text-[11px] text-data-unreachable"
          data-testid="processing-stderr"
        >
          {status.error}
        </pre>
      )}

      <div className="overflow-hidden rounded-md border-hair border-border bg-surface" data-testid="stage-feed">
        <div className="flex items-center justify-between border-b-hair border-border px-4 py-2">
          <span className="font-mono text-[11px] uppercase tracking-[0.06em] text-muted">Stages</span>
          {/* Which file these rows came from, so a degraded feed is never mistaken for
             the authoritative one. */}
          <span className="font-mono text-[11px] text-faint" data-testid="feed-source">
            {status?.source ?? '—'}
          </span>
        </div>

        {stages.length === 0 ? (
          <p className="px-4 py-6 text-sm text-muted" data-testid="stage-feed-empty">
            No stages recorded yet. Nothing is invented here — this list stays empty until
            the pipeline writes status.json or logs a stage transition.
          </p>
        ) : (
          <ul>
            {stages.map((s) => {
              // Every one of these is null when the writer recorded nothing, and null
              // means the line is not drawn. No "—", no "n/a": a placeholder claims the
              // pipeline said something it did not.
              const sub = substageLine(s)
              const attempt = attemptLine(s)
              const waiting = elapsedSince(s.since, nowMs)
              const isCurrent = s.stage === current
              return (
                <li
                  key={s.stage}
                  data-stage-row={s.stage}
                  data-stage-state={s.state}
                  data-stage-current={isCurrent ? 'true' : undefined}
                  className={`border-b-hair border-border px-s4 py-s2 last:border-b-0 ${
                    isCurrent ? 'bg-accent-dim' : ''
                  }`}
                >
                  <div className="flex items-baseline gap-s3">
                    <span
                      className={`h-1.5 w-1.5 shrink-0 rounded-full ${
                        isCurrent
                          ? 'animate-pulse bg-accent'
                          : s.error
                            ? 'bg-status-blocked'
                            : 'bg-data-object'
                      }`}
                    />
                    <span
                      className={`min-w-[140px] font-mono text-bodyux ${isCurrent ? 'text-fg' : 'text-subtle'}`}
                    >
                      {s.stage}
                    </span>
                    <span className="min-w-[64px] font-mono text-caption text-muted">{s.state}</span>
                    <span className="min-w-[84px] font-mono text-caption text-muted">{clock(s.started_at)}</span>
                    <span className="font-mono text-caption text-faint">{duration(s.duration_sec)}</span>
                  </div>

                  {sub && (
                    <p
                      data-stage-substage={s.substage ?? undefined}
                      className="mt-s1 pl-s6 font-mono text-caption text-subtle"
                    >
                      {sub}
                      {waiting && <span className="text-muted"> · {waiting}</span>}
                    </p>
                  )}

                  {attempt && (
                    <p data-stage-attempt className="mt-s1 pl-s6 font-mono text-caption text-muted">
                      {attempt}
                    </p>
                  )}

                  {s.error && (
                    <p className="mt-s1 pl-s6 font-mono text-caption text-status-blocked">{s.error}</p>
                  )}
                </li>
              )
            })}
          </ul>
        )}
      </div>

      <p className="mt-3 font-mono text-[11px] text-faint">
        polling every {POLL_MS / 1000}s
        {status?.seconds_since_last_record != null &&
          ` · last record ${Math.round(status.seconds_since_last_record)}s ago`}
      </p>
    </div>
  )
}
