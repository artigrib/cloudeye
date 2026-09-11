import { useEffect, useState } from 'react'
import { NotFoundError } from '../api/client'
import { getSceneCritic } from '../api/scenes'
import type { CriticReportResponse } from '../api/types'

interface Props {
  sceneId: string
}

/** Read-only warnings badge for scripts/audit/critic_rules.py + critic_vlm.py's
 * merged report (`GET /api/scenes/{id}/critic`, see app.routers.scenes.get_scene_critic
 * and scripts/audit/critic_merge.py) - lists what the scene critic flagged, nothing
 * more. **No auto-fix**: clicking a finding does nothing but show its text - this
 * component never calls a mutating endpoint and never edits scene data, same
 * guarantee as the rest of the critic pipeline (critic_rules.py's module docstring).
 *
 * PM decision (2026-09-07): **the rules engine is the gate; the VLM is advisory
 * only.** The badge's count and colour are driven ENTIRELY by `rules.findings` -
 * `vlm.findings` never affect whether the badge shows, its count, or its colour.
 * Hidden entirely when the scene has no critic report on disk (404, the default -
 * the critic is opt-in, run manually per scene) or when there are zero RULE
 * findings, even if the VLM raised something on its own (an unverified,
 * non-gating read of a render is not, by itself, a reason to warn). The expanded
 * list shows rule findings first, then any VLM findings each explicitly labelled
 * "unverified (vlm)". */
export default function CriticBadge({ sceneId }: Props) {
  const [report, setReport] = useState<CriticReportResponse | null>(null)
  const [open, setOpen] = useState(false)

  useEffect(() => {
    let cancelled = false
    setReport(null)
    setOpen(false)
    getSceneCritic(sceneId)
      .then((r) => {
        if (!cancelled) setReport(r)
      })
      .catch((err) => {
        if (err instanceof NotFoundError) return // no critic report for this scene - badge stays hidden
        console.error('Failed to fetch scene critic report for scene %s', sceneId, err)
      })
    return () => {
      cancelled = true
    }
  }, [sceneId])

  if (!report) return null
  const ruleFindings = report.rules?.findings ?? []
  const vlmFindings = report.vlm?.findings ?? []
  // Gate: rule findings only. Zero rule findings -> hidden, regardless of what the
  // (non-gating, advisory-only) VLM pass found.
  if (ruleFindings.length === 0) return null

  return (
    <div className="relative shrink-0">
      <button
        type="button"
        onClick={() => setOpen((prev) => !prev)}
        className="hit-target flex items-center rounded-sm border-hair border-status-partial/40 bg-status-partial/10 px-s2 text-caption text-status-partial hover:bg-status-partial/20"
        data-testid="critic-badge"
        aria-expanded={open}
      >
        ⚠ {ruleFindings.length} critic finding{ruleFindings.length === 1 ? '' : 's'}
      </button>
      {open && (
        <div
          className="absolute right-0 top-full z-10 mt-1 max-h-80 w-96 overflow-y-auto rounded-md border-hair border-border bg-surface-raised p-2 text-left shadow-lg"
          data-testid="critic-findings-list"
        >
          <ul className="flex flex-col gap-1.5">
            {ruleFindings.map((f, i) => (
              <li key={`rule-${i}`} className="border-b-hair border-border pb-1.5 last:border-b-0 last:pb-0">
                <div className="flex items-center gap-1.5">
                  <span className="font-mono text-[10px] uppercase text-muted">{f.severity}</span>
                  <span className="truncate font-medium text-fg">{f.object_id}</span>
                </div>
                <p className="text-muted">{f.detail}</p>
              </li>
            ))}
            {vlmFindings.map((f, i) => (
              <li key={`vlm-${i}`} className="border-b-hair border-border pb-1.5 last:border-b-0 last:pb-0" data-testid="critic-vlm-finding">
                <div className="flex items-center gap-1.5">
                  <span className="font-mono text-[10px] uppercase text-muted">{f.severity}</span>
                  <span
                    className="rounded bg-muted/15 px-1 font-mono text-[9px] uppercase tracking-wide text-muted"
                    title="Advisory only - a VLM's free-text read of a render, not independently verified, never gating"
                  >
                    unverified (vlm)
                  </span>
                  <span className="truncate font-medium text-fg">{f.object_id_or_region}</span>
                </div>
                <p className="text-muted">{f.issue}</p>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
