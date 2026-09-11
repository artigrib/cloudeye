import { useEffect, useState } from 'react'
import { getProject } from '../api/projects'
import type { SceneStatus } from '../api/types'

/** Project-card status derived from its most-recently-created scene - see
 * useLatestSceneStatus below. `null` covers both "still loading" and "load failed";
 * either way the card falls back to rendering no badge rather than a wrong one. */
export type LatestSceneStatus = SceneStatus | 'none' | null

/** `ProjectResponse.primary_scene_id` means "which scene is featured," not "is this
 * project done processing" - it's null for plenty of projects whose scenes finished or
 * failed hours ago, so HomePage can't use it alone to render an accurate status badge.
 * This fetches the project detail (which does carry each scene's `status`) and picks
 * the most recently created scene's status, refetching whenever `projectId` changes.
 * Modeled on useScenePreviewUrl.ts: local state, cancels on unmount/change, swallows
 * fetch errors (a stale/missing badge beats a page-breaking error here). */
export function useLatestSceneStatus(projectId: string): LatestSceneStatus {
  const [status, setStatus] = useState<LatestSceneStatus>(null)

  useEffect(() => {
    setStatus(null)
    let cancelled = false

    getProject(projectId)
      .then((detail) => {
        if (cancelled) return
        if (detail.scenes.length === 0) {
          setStatus('none')
          return
        }
        const latest = detail.scenes.reduce((a, b) =>
          new Date(b.created_at).getTime() > new Date(a.created_at).getTime() ? b : a,
        )
        setStatus(latest.status)
      })
      .catch(() => {})

    return () => {
      cancelled = true
    }
  }, [projectId])

  return status
}
