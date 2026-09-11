import { useEffect, useState } from 'react'
import type { SceneStatusResponse } from '../api/types'
import { translateSceneError } from '../lib/sceneErrorMessage'
import { estimateEta, formatDurationCompact, stageLabel } from '../lib/stageEta'

const SPINNER_CLASS: Record<'compact' | 'full', string> = {
  compact: 'h-3 w-3 animate-spin rounded-full border-2 border-accent/30 border-t-accent',
  full: 'mx-auto mb-4 h-8 w-8 animate-spin rounded-full border-2 border-accent/30 border-t-accent',
}

interface Props {
  /** Either poller's shape - a SceneStatusResponse or a SceneResponse, both of which
   * carry `stage`/`stage_timings` (see app/schemas.py). */
  scene: Pick<SceneStatusResponse, 'status' | 'error_message' | 'stage_timings'> | null
  /** `full`: ScenePage's centered panel, the only caller today. `compact` was VideoRow's
   * small trailing status column; VideoRow went with the project page, so that variant
   * currently has no caller - kept because it is one branch of a layout switch, not dead
   * machinery, and the processing screen is the obvious next user of it. Only affects
   * layout/text size, not which states are handled. */
  variant?: 'compact' | 'full'
}

/** Stage-by-stage progress + ETA for a scene that isn't `done` yet, and the
 * validator's actual failure message (not a bare "failed") once it is `failed`.
 * Renders nothing once `status === 'done'` - callers render their own "open scene"
 * link/content for that case. */
export default function JobTimeline({ scene, variant = 'compact' }: Props) {
  // Ticks once a second so the "~Xs left" / elapsed display counts down smoothly
  // between polls (ScenePage polls every 6s) - purely local, no network.
  const [, setTick] = useState(0)
  const status = scene?.status
  useEffect(() => {
    if (status !== 'processing' && status !== 'queued') return
    const id = setInterval(() => setTick((n) => n + 1), 1000)
    return () => clearInterval(id)
  }, [status])

  if (!scene || scene.status === 'queued') {
    if (variant === 'compact') {
      return (
        <div className="flex items-center gap-2 text-xs text-muted">
          <span className={SPINNER_CLASS.compact} />
          <span>Queued</span>
        </div>
      )
    }
    return (
      <>
        <span className={SPINNER_CLASS.full} />
        <p className="text-sm text-subtle">Queued for reconstruction…</p>
        <p className="mt-1 text-xs text-muted">Usually takes ~5 minutes.</p>
      </>
    )
  }

  if (scene.status === 'failed') {
    const message = translateSceneError(scene.error_message)
    if (variant === 'compact') {
      return (
        <p className="max-w-xs text-xs text-data-unreachable" title={scene.error_message ?? undefined}>
          Failed: {message}
        </p>
      )
    }
    return (
      <p className="mt-1 text-xs text-muted" title={scene.error_message ?? undefined}>
        {message}
      </p>
    )
  }

  if (scene.status === 'processing') {
    const eta = estimateEta(scene.stage_timings)
    const label = eta ? stageLabel(eta.currentStage) : 'Processing'
    const remaining =
      eta?.remainingSec != null
        ? `~${formatDurationCompact(eta.remainingSec)} left${eta.isRough ? ' (estimate)' : ''}`
        : null

    if (variant === 'compact') {
      return (
        <div className="flex items-center gap-2 text-xs text-muted">
          <span className={SPINNER_CLASS.compact} />
          <span>
            {label}
            {remaining ? ` · ${remaining}` : ''}
          </span>
        </div>
      )
    }
    return (
      <>
        <span className={SPINNER_CLASS.full} />
        <p className="text-sm text-subtle">{label}…</p>
        <p className="mt-1 text-xs text-muted">{remaining ?? 'Usually takes ~5 minutes.'}</p>
      </>
    )
  }

  return null
}
