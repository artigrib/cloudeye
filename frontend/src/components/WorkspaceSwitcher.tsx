import { useMatch, useNavigate } from 'react-router-dom'
import { useProjects } from '../state/projects-context'
import Select from './ui/Select'

/** Sentinel option value - not a project id, so it can never collide with one. */
const ALL_PROJECTS = '__all__'
const NEW_WORKSPACE = '__new__'

/** Which workspace the URL is currently about, or null on the routes that are about none
 * (`/workspaces`, `/workspaces/new`).
 *
 * Derived here rather than passed in. It used to be a prop, and the Sidebar passed a
 * hard-coded `null` - so on every route except the scene screen the trigger fell through
 * to its placeholder and read "Loading…" forever, long after the list had arrived. Making
 * the component answer its own question removes the class of bug, not just the instance.
 *
 * `/workspaces/new` is excluded explicitly: `:workspaceId` happily matches the literal
 * "new", and the wizard is not a workspace. */
function useCurrentWorkspaceId(): string | null {
  const scene = useMatch('/workspaces/:workspaceId/scenes/:sceneId')
  const processing = useMatch('/workspaces/:workspaceId/processing')
  const workspace = useMatch('/workspaces/:workspaceId')
  const id =
    scene?.params.workspaceId ?? processing?.params.workspaceId ?? workspace?.params.workspaceId ?? null
  return id === 'new' ? null : id
}

/** The workspace picker: which workspace is open, and how to get to another one, in the
 * space the expanded project list used to occupy in the global sidebar.
 *
 * It lists WORKSPACES, not scenes, because that is what the app's navigation is keyed on -
 * a workspace owns one `primary_scene_id`. Selecting one goes straight to its scene, or to
 * the workspace itself when it has none yet, so nothing that used to be one click away
 * became two. "All workspaces" is the last entry and replaces the header's old `← back`
 * link.
 *
 * `project` survives inside this file only as the API's own vocabulary - the `projects`
 * table, `/api/projects`, `ProjectResponse`, `primary_scene_id`. None of that is renamed:
 * the DB is shared with production and this change is UI and routes only.
 *
 * The FIRST entry is "+ New workspace", and it is the only way into the upload flow: the
 * sidebar's old "+ UPLOAD VIDEO" button and its capture-guide "?" both moved into the
 * wizard (the button became the page, the "?" became the help inside its upload step).
 * That is why this component is mounted in the global sidebar as well as in the scene
 * screen's left column - the single entry point has to exist on every route, not only on
 * the one screen that happens to have a left column of its own.
 */
interface Props {
  /** The wrapper's own chrome. Defaults to the global sidebar's - a bottom rule and 12px
   * of padding of its own. The scene screen passes "" and lets its own section body
   * (`px-s4`) place the control, so the workspace select shares one left edge and one
   * width with the robot select, the view toggle and the video card above and below it.
   * It used to be 12px further in and 24px narrower than all three, being the only
   * control in that column that padded itself. */
  className?: string
}

export default function WorkspaceSwitcher({
  className = 'border-b-hair border-border px-3 py-2.5',
}: Props = {}) {
  const { projects, loading } = useProjects()
  const navigate = useNavigate()
  const currentWorkspaceId = useCurrentWorkspaceId()

  const options = [
    // First, per spec: this is the ONLY way into the upload flow now.
    { value: NEW_WORKSPACE, label: '+ New workspace', sublabel: 'upload a video' },
    ...projects.map((p) => ({
      value: p.id,
      label: p.name,
      sublabel: p.primary_scene_id ? undefined : 'no scene yet',
    })),
    { value: ALL_PROJECTS, label: 'All workspaces', sublabel: 'browse every workspace' },
  ]

  function handleChange(value: string) {
    if (value === NEW_WORKSPACE) {
      navigate('/workspaces/new')
      return
    }
    if (value === ALL_PROJECTS) {
      navigate('/workspaces')
      return
    }
    const project = projects.find((p) => p.id === value)
    if (!project) return
    navigate(
      project.primary_scene_id
        ? `/workspaces/${project.id}/scenes/${project.primary_scene_id}`
        : `/workspaces/${project.id}`,
    )
  }

  return (
    <div className={className} data-testid="workspace-switcher">
      {/* The placeholder says "Loading…" only while the list genuinely is loading. Once
          it has arrived the trigger shows the current workspace's name, or "Select
          workspace" on the routes that are about no particular one - never "Loading…". */}
      <Select
        value={currentWorkspaceId}
        data-testid="workspace-select"
        onChange={handleChange}
        options={options}
        placeholder={loading ? 'Loading…' : 'Select workspace'}
        aria-label="Workspace"
        className="w-full"
      />
    </div>
  )
}
