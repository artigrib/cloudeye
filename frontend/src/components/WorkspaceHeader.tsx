import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { deleteProject, updateProject } from '../api/projects'
import type { ProjectResponse } from '../api/types'
import { useProjects } from '../state/projects-context'
import Button from './ui/Button'
import Input from './ui/Input'
import Modal from './ui/Modal'

interface Props {
  workspace: ProjectResponse
  /** How many videos this workspace actually holds. A workspace is one video and one
   * scene, but rows created before that rule exists can hold up to nine - see the note
   * this renders when the count is above one. */
  videoCount: number
  /** Re-read the workspace after a rename/archive. Not called after delete - the page is
   * gone by then. */
  onChanged: () => void | Promise<void>
}

/** The workspace's own controls, above whichever state `/workspaces/:id` is showing.
 *
 * These used to live on the project page, which no longer exists as a screen: name and
 * archive are carried over from it verbatim, delete is new. Deliberately NOT rendered on
 * the scene route - `ScenePage` polls a scene, and offering to delete its files from
 * under it is a trap. */
export default function WorkspaceHeader({ workspace, videoCount, onChanged }: Props) {
  const [renaming, setRenaming] = useState(false)
  const [nameDraft, setNameDraft] = useState('')
  const [renameError, setRenameError] = useState<string | null>(null)
  const [archiving, setArchiving] = useState(false)
  const [busyError, setBusyError] = useState<string | null>(null)
  const [confirmDelete, setConfirmDelete] = useState(false)
  const [deleting, setDeleting] = useState(false)
  const navigate = useNavigate()
  const { refresh: refreshWorkspaces } = useProjects()

  async function saveRename(e: React.FormEvent) {
    e.preventDefault()
    const name = nameDraft.trim()
    if (!name) return
    try {
      await updateProject(workspace.id, { name })
      setRenaming(false)
      await Promise.all([onChanged(), refreshWorkspaces()])
    } catch {
      setRenameError('Failed to rename workspace.')
    }
  }

  // Archiving hides the workspace from the default listing (list_projects'
  // include_archived) and touches neither its video nor its scene, so it needs no
  // confirmation. It DOES make the workspace vanish from the sidebar while you are still
  // standing on its URL, so this leaves for the gallery afterwards rather than stranding
  // you on a page nothing links to any more.
  async function toggleArchived() {
    setArchiving(true)
    setBusyError(null)
    try {
      const nowArchived = !workspace.archived
      await updateProject(workspace.id, { archived: nowArchived })
      await refreshWorkspaces()
      if (nowArchived) navigate('/workspaces')
      else await onChanged()
    } catch {
      setBusyError('Failed to update.')
    } finally {
      setArchiving(false)
    }
  }

  // DELETE /api/projects/{id} is a hard delete: project_service.delete_project rmtree's
  // each scene directory and unlinks each video file before removing the row, and Postgres
  // cascades videos/scenes/scene_objects/commands. Nothing here is recoverable, which is
  // the whole reason for the confirm step.
  async function reallyDelete() {
    setDeleting(true)
    setBusyError(null)
    try {
      await deleteProject(workspace.id)
      await refreshWorkspaces()
      navigate('/workspaces', { replace: true })
    } catch {
      setBusyError('Failed to delete.')
      setDeleting(false)
      setConfirmDelete(false)
    }
  }

  return (
    <div className="mx-auto max-w-3xl px-8 pt-8" data-testid="workspace-header">
      <div className="flex items-start justify-between gap-4">
        {renaming ? (
          <form onSubmit={saveRename} className="flex min-w-0 flex-1 items-center gap-2">
            <Input
              autoFocus
              value={nameDraft}
              onChange={(e) => setNameDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Escape') setRenaming(false)
              }}
              className="text-lg font-semibold"
              aria-label="Workspace name"
            />
            <Button variant="ghost" type="submit">
              Save
            </Button>
            <Button variant="ghost" onClick={() => setRenaming(false)}>
              Cancel
            </Button>
          </form>
        ) : (
          <h1 className="group flex min-w-0 items-center gap-2 text-[19px] font-semibold tracking-tight text-fg">
            <span className="truncate" data-testid="workspace-name">
              {workspace.name}
            </span>
            {workspace.archived && (
              <span className="shrink-0 rounded border-hair border-border bg-surface-raised px-1.5 py-0.5 text-[10px] font-normal text-faint">
                archived
              </span>
            )}
            <button
              type="button"
              onClick={() => {
                setNameDraft(workspace.name)
                setRenameError(null)
                setRenaming(true)
              }}
              data-testid="rename-workspace"
              aria-label="Rename workspace"
              className="shrink-0 text-xs text-faint opacity-0 transition-opacity duration-100 group-hover:opacity-100 hover:text-muted"
            >
              ✎
            </button>
          </h1>
        )}

        <div className="flex shrink-0 items-center gap-2">
          <Button
            variant="ghost"
            onClick={toggleArchived}
            disabled={archiving}
            data-testid="archive-workspace"
            aria-label={workspace.archived ? 'Unarchive workspace' : 'Archive workspace'}
          >
            {workspace.archived ? 'Unarchive' : 'Archive'}
          </Button>
          <Button
            variant="ghost"
            onClick={() => setConfirmDelete(true)}
            disabled={deleting}
            data-testid="delete-workspace"
            aria-label="Delete workspace"
            className="text-data-unreachable"
          >
            Delete
          </Button>
        </div>
      </div>

      {renameError && <p className="mt-1 text-xs text-data-unreachable">{renameError}</p>}
      {busyError && <p className="mt-1 text-xs text-data-unreachable">{busyError}</p>}

      {/* A workspace is one video and one scene. Rows created before that rule can hold
          more, and the screens below only ever show ONE of them - said out loud here so
          the others are not silently invisible. Everything is still reachable by URL. */}
      {videoCount > 1 && (
        <p className="mt-2 text-xs text-muted" data-testid="multi-video-note">
          This workspace holds {videoCount} videos, from before a workspace meant exactly
          one. Only the most recent is shown.
        </p>
      )}

      {confirmDelete && (
        <Modal onClose={() => setConfirmDelete(false)}>
          <div className="flex flex-col gap-3" data-testid="delete-confirm">
            <h2 className="text-[15px] font-medium text-fg">Delete “{workspace.name}”?</h2>
            <p className="text-sm text-muted">
              This removes the workspace, its video and its reconstructed scene, including
              the files on disk. It cannot be undone. Archive instead if you only want it
              out of the list.
            </p>
            <div className="flex justify-end gap-2">
              <Button variant="ghost" onClick={() => setConfirmDelete(false)}>
                Cancel
              </Button>
              <Button
                variant="primary"
                onClick={reallyDelete}
                disabled={deleting}
                data-testid="confirm-delete-workspace"
              >
                {deleting ? 'Deleting…' : 'Delete'}
              </Button>
            </div>
          </div>
        </Modal>
      )}
    </div>
  )
}
