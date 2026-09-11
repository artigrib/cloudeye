import { useCallback, useEffect, useState } from 'react'
import { Link, Navigate, useParams } from 'react-router-dom'
import { NotFoundError } from '../api/client'
import { getProject } from '../api/projects'
import type { ProjectDetailResponse } from '../api/types'
import WorkspaceHeader from '../components/WorkspaceHeader'
import NewWorkspacePage from './NewWorkspacePage'

/** Which scene this workspace opens to, or null while it has none finished.
 *
 * `primary_scene_id` FIRST, and this is not cosmetic: WorkspaceSwitcher already navigates
 * by that field, so picking "newest done" here instead would open two different scenes for
 * the same workspace depending on whether you came from the dropdown or from a link. The
 * fallback covers a workspace whose primary was never set, and self-heals - ScenePage
 * writes `primary_scene_id` on the first successful open. */
function resolveSceneId(detail: ProjectDetailResponse): string | null {
  const done = detail.scenes.filter((s) => s.status === 'done')
  if (detail.project.primary_scene_id && done.some((s) => s.id === detail.project.primary_scene_id)) {
    return detail.project.primary_scene_id
  }
  if (!done.length) return null
  return done.reduce((a, b) => (new Date(b.created_at) > new Date(a.created_at) ? b : a)).id
}

/** `/workspaces/:workspaceId` is not a screen - it is the one place that answers "what is
 * this workspace right now" and sends you to the canonical URL for that answer:
 *
 *   no video      -> the wizard, with this workspace preselected
 *   scene done    -> the scene
 *   otherwise     -> the processing screen
 *
 * It asks `GET /api/projects/{id}`, NOT the processing poll. That is deliberate:
 * ProcessingStatusResponse has `scene_id: string | null` and nothing else, so it cannot
 * tell "this workspace has no video" from "the pipeline has not made a scene yet" - the
 * two states that decide between the wizard and the progress screen. The project detail
 * carries `videos` and each scene's `status`, which answers it in one call, and it is a
 * different endpoint from the one ProcessingPage polls, so nothing is fetched twice.
 *
 * The redirects are render-time `<Navigate>`, not an effect: under StrictMode an
 * effect-based navigate fires twice, and there is no ordering hazard to reason about.
 * ProcessingPage keeps its own DONE redirect - it is a deep-linkable URL reached straight
 * from the wizard - and the two never run at once, because this hands off by changing the
 * URL rather than rendering ProcessingPage inline. */
export default function WorkspacePage() {
  const { workspaceId } = useParams<{ workspaceId: string }>()
  const [detail, setDetail] = useState<ProjectDetailResponse | null>(null)
  const [notFound, setNotFound] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(async () => {
    if (!workspaceId) return
    try {
      setDetail(await getProject(workspaceId))
      setError(null)
    } catch (err) {
      if (err instanceof NotFoundError) setNotFound(true)
      // Not re-thrown: the old page threw from inside an async effect, which becomes an
      // unhandled rejection and a blank screen rather than a message.
      else setError(err instanceof Error ? err.message : 'Could not reach the server.')
    }
  }, [workspaceId])

  useEffect(() => {
    // Clear first: a workspaceId change must not render the previous workspace's answer
    // for a frame while the new one is in flight.
    setDetail(null)
    setNotFound(false)
    void load()
  }, [load])

  if (notFound) {
    return (
      <div className="mx-auto max-w-3xl px-8 py-10" data-testid="workspace-not-found">
        <p className="text-sm text-muted">Workspace not found.</p>
        <Link to="/workspaces" className="mt-3 inline-block text-sm text-accent hover:underline">
          Back to workspaces
        </Link>
      </div>
    )
  }

  if (error) {
    return (
      <div className="mx-auto max-w-3xl px-8 py-10">
        <p className="text-sm text-data-unreachable">{error}</p>
      </div>
    )
  }

  // Same wording as ProcessingPage's own loading state, so the hand-off reads as one
  // continuous load rather than two screens.
  if (!detail) {
    return <div className="mx-auto max-w-3xl px-8 py-10 text-sm text-muted">Loading…</div>
  }

  if (detail.videos.length === 0) {
    return (
      <>
        <WorkspaceHeader workspace={detail.project} videoCount={0} onChanged={load} />
        <NewWorkspacePage workspaceId={detail.project.id} />
      </>
    )
  }

  const sceneId = resolveSceneId(detail)
  return (
    <Navigate
      replace
      to={
        sceneId
          ? `/workspaces/${detail.project.id}/scenes/${sceneId}`
          : `/workspaces/${detail.project.id}/processing`
      }
    />
  )
}
