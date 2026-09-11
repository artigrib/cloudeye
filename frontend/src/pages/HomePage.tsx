import { type CSSProperties } from 'react'
import { Link } from 'react-router-dom'
import type { ProjectResponse } from '../api/types'
import Badge from '../components/ui/Badge'
import { useLatestSceneStatus } from '../lib/useLatestSceneStatus'
import { useScenePreviewUrl } from '../lib/useScenePreviewUrl'
import { useProjects } from '../state/projects-context'

function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' })
}

/** Label + Badge variant for a project card's status badge, derived from its latest
 * scene's status (see useLatestSceneStatus) - null means "render no badge", for a
 * scene that finished successfully or a status we're still waiting on. */
function statusBadge(status: ReturnType<typeof useLatestSceneStatus>): { label: string; variant: 'unreachable' | 'neutral' } | null {
  switch (status) {
    case 'queued':
      return { label: 'Queued', variant: 'neutral' }
    case 'processing':
      return { label: 'Processing', variant: 'neutral' }
    case 'failed':
      return { label: 'Failed', variant: 'unreachable' }
    case 'none':
      return { label: 'No video yet', variant: 'neutral' }
    case 'done':
    case null:
      return null
    default:
      return null
  }
}

/** Faint grid texture standing in for the occupancy map an unopened/unfinished
 * scene doesn't have a thumbnail for yet - a plain flat card reads as "empty",
 * this reads as "a map that hasn't loaded", which is what's actually true. */
const SKELETON_STYLE: CSSProperties = {
  backgroundColor: 'var(--canvas)',
  backgroundImage:
    'linear-gradient(var(--border) 1px, transparent 1px), linear-gradient(90deg, var(--border) 1px, transparent 1px)',
  backgroundSize: '18px 18px',
}

function WorkspaceCard({ project }: { project: ProjectResponse }) {
  const previewUrl = useScenePreviewUrl(project.primary_scene_id)
  const latestStatus = useLatestSceneStatus(project.id)
  const badge = statusBadge(latestStatus)

  return (
    <Link
      to={`/workspaces/${project.id}`}
      className="group relative flex h-[168px] flex-col justify-end overflow-hidden rounded-md border-hair border-border-strong transition-colors duration-100 hover:border-fg/30"
    >
      <div className="absolute inset-0" style={previewUrl ? undefined : SKELETON_STYLE}>
        {previewUrl && (
          <img src={previewUrl} alt="" className="h-full w-full object-cover" />
        )}
      </div>
      <div className="absolute inset-0 bg-gradient-to-t from-surface from-15% via-surface/40 via-60% to-transparent" />
      {badge && (
        <Badge variant={badge.variant} className="absolute right-2 top-2">
          {badge.label}
        </Badge>
      )}
      <div className="relative z-10 p-3">
        <h2 className="truncate text-sm font-medium text-fg">{project.name}</h2>
        {project.description && <p className="mt-0.5 line-clamp-1 text-xs text-muted">{project.description}</p>}
        <p className="mt-1 font-mono text-xs text-faint">{formatDate(project.created_at)}</p>
      </div>
    </Link>
  )
}

/** The gallery's entry into the wizard. It used to be an inline "name it and create an
 * empty project" form, which produced a workspace with no video - a state that now has no
 * meaning, since a workspace IS one video and one scene. Creating one goes through the
 * wizard, which is the only way a video gets in. */
function NewWorkspaceCard() {
  return (
    <Link
      to="/workspaces/new"
      data-testid="new-workspace-card"
      className="flex h-[168px] flex-col items-center justify-center gap-1 rounded-md border-2 border-dashed border-border-strong text-muted transition-colors duration-100 hover:border-accent/40 hover:text-accent"
    >
      <span className="text-2xl leading-none">+</span>
      <span className="text-sm">New workspace</span>
    </Link>
  )
}

export default function HomePage() {
  const { projects, loading, error } = useProjects()

  return (
    <div className="mx-auto max-w-5xl px-8 py-10">
      <h1 className="text-h2 font-medium text-fg">Workspaces</h1>
      <p className="mt-1 text-sm text-muted">
        A workspace is one room: one video, one reconstructed scene. Create one to get a
        3D scene, an occupancy map and an object set out of a walkthrough.
      </p>

      {error && <p className="mt-6 text-sm text-data-unreachable">{error}</p>}
      {loading && <p className="mt-6 text-sm text-muted">Loading…</p>}

      <div className="mt-6 grid grid-cols-[repeat(auto-fill,minmax(280px,1fr))] gap-4">
        <NewWorkspaceCard />
        {projects.map((p) => (
          <WorkspaceCard key={p.id} project={p} />
        ))}
      </div>
    </div>
  )
}
