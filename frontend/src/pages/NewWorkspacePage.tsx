import { useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { uploadVideo } from '../api/videos'
import { useProjects } from '../state/projects-context'
import { useCaptureGuide } from '../state/capture-guide-context'
import Button from '../components/ui/Button'
import {
  CHAT_LLM_CHOICES,
  DEFAULT_JOB_SPEC,
  FPS_CHOICES,
  SEMANTICS_CHOICES,
  type ChatLlm,
  type JobSpecRequest,
  type Semantics,
} from '../lib/jobSpec'

/** One row of the wizard. Deliberately not a stepper: every choice has a default that is
 * already correct, so the operator should be able to see all of them at once, change the
 * one they care about, and drop in a file - not click Next four times past three screens
 * they were going to accept. The spec calls this "one screen, top to bottom". */
function Step({
  n,
  title,
  hint,
  children,
  testId,
}: {
  n: number
  title: string
  hint?: string
  children: React.ReactNode
  testId: string
}) {
  return (
    <section className="border-b-hair border-border px-6 py-5" data-testid={testId}>
      <div className="mb-3 flex items-baseline gap-3">
        <span className="font-mono text-[11px] text-faint">{n}</span>
        <h2 className="text-[15px] font-medium text-fg">{title}</h2>
        {hint && <span className="text-xs text-muted">{hint}</span>}
      </div>
      {children}
    </section>
  )
}

/** A radio row rendered as buttons - the same visual language as SegmentedControl, but
 * with a second line per option, which the segmented control has nowhere to put. */
function Choice<T extends string | number>({
  value,
  onChange,
  options,
  name,
}: {
  value: T
  onChange: (v: T) => void
  options: { value: T; label: string; note: string }[]
  name: string
}) {
  return (
    <div className="flex flex-wrap gap-2" role="radiogroup" aria-label={name}>
      {options.map((o) => (
        <button
          key={String(o.value)}
          type="button"
          role="radio"
          aria-checked={o.value === value}
          data-choice={`${name}:${o.value}`}
          onClick={() => onChange(o.value)}
          className={`min-w-[168px] rounded-md border-hair px-3 py-2 text-left transition-colors duration-100 ${
            o.value === value
              ? 'border-accent bg-accent-dim text-fg'
              : 'border-border text-subtle hover:bg-fg/5'
          }`}
        >
          <span className="block text-sm">{o.label}</span>
          <span className="block font-mono text-[11px] text-muted">{o.note}</span>
        </button>
      ))}
    </div>
  )
}

/** "+ New workspace" - the single entry point for getting a video into CloudEye. The
 * sidebar's old "+ UPLOAD VIDEO" button and its capture-guide "?" both live here now: the
 * button became this page, and the "?" is the help inside the upload step, which is the
 * only step it was ever about. */
interface Props {
  /** Fill THIS existing, empty workspace instead of letting the upload auto-create one.
   * Set by WorkspacePage for `/workspaces/:id` when the workspace holds no video yet;
   * undefined on `/workspaces/new`. The backend already takes both - POST
   * /api/videos/upload accepts `project_id` alongside `job_spec` - so this needs no new
   * endpoint. Deliberately a prop rather than a useParams() read inside this component:
   * it would then behave differently depending on which route mounted it, invisibly, and
   * could not be tested outside a Router. */
  workspaceId?: string
}

export default function NewWorkspacePage({ workspaceId }: Props = {}) {
  const [spec, setSpec] = useState<JobSpecRequest>(DEFAULT_JOB_SPEC)
  const [file, setFile] = useState<File | null>(null)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const fileInput = useRef<HTMLInputElement>(null)
  const navigate = useNavigate()
  const { refresh } = useProjects()
  const { openGuide } = useCaptureGuide()

  async function start() {
    if (!file || uploading) return
    setUploading(true)
    setError(null)
    try {
      const video = await uploadVideo(file, workspaceId, { jobSpec: spec })
      await refresh()
      // The upload auto-creates the workspace when none is given, so the workspace id IS
      // that project id - see video_service.handle_upload.
      const id = video.project_id ?? workspaceId
      // `replace` is required in the embedded case, not stylistic: the path was
      // /workspaces/:id, which now has a video and would immediately re-dispatch forward -
      // without it, Back is a trap. It is also right on /workspaces/new (Back goes to the
      // gallery you came from).
      if (id) navigate(`/workspaces/${id}/processing`, { replace: true })
      else setError('Upload succeeded but the server returned no workspace to open.')
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Upload failed.')
    } finally {
      setUploading(false)
    }
  }

  return (
    <div className="mx-auto max-w-3xl px-8 py-8" data-testid="new-workspace">
      {/* Embedded in /workspaces/:id, WorkspaceHeader is already above this with the
          workspace's own name and controls - a second title and a Cancel that leaves the
          workspace entirely would both be wrong there. */}
      {!workspaceId && (
        <header className="mb-6 flex items-baseline justify-between gap-4">
          <h1 className="text-[19px] font-semibold tracking-tight text-fg">New workspace</h1>
          <button
            type="button"
            onClick={() => navigate('/workspaces')}
            className="text-xs text-muted hover:text-subtle"
          >
            Cancel
          </button>
        </header>
      )}

      <div className="overflow-hidden rounded-md border-hair border-border bg-surface">
        <Step n={1} title="Frames" hint="how densely the walk is sampled" testId="step-frames">
          <Choice
            name="frames_fps"
            value={spec.frames_fps}
            onChange={(v) => setSpec((s) => ({ ...s, frames_fps: v }))}
            options={FPS_CHOICES}
          />
        </Step>

        <Step n={2} title="Reconstruction" testId="step-mapping">
          {/* No control: one backend exists. Shown anyway so the pipeline is not a black
              box - and so the day a second one appears, this row is where it goes. */}
          <div className="inline-flex flex-col rounded-md border-hair border-border px-3 py-2">
            <span className="text-sm text-subtle">MapAnything</span>
            <span className="font-mono text-[11px] text-muted">
              facebook/map-anything-apache · the only option
            </span>
          </div>
        </Step>

        <Step n={3} title="Semantics" hint="names the objects in the room" testId="step-semantics">
          <Choice
            name="semantics"
            value={spec.semantics}
            onChange={(v) => setSpec((s) => ({ ...s, semantics: v as Semantics }))}
            options={SEMANTICS_CHOICES}
          />
        </Step>

        <Step n={4} title="Chat LLM" hint="parses typed robot commands" testId="step-chat-llm">
          <Choice
            name="chat_llm"
            value={spec.chat_llm}
            onChange={(v) => setSpec((s) => ({ ...s, chat_llm: v as ChatLlm }))}
            options={CHAT_LLM_CHOICES}
          />
        </Step>

        <Step n={5} title="Video" testId="step-video">
          <div className="flex flex-wrap items-center gap-3">
            <input
              ref={fileInput}
              type="file"
              accept="video/mp4,video/quicktime,.mp4,.mov"
              data-testid="video-file-input"
              onChange={(e) => {
                setFile(e.target.files?.[0] ?? null)
                setError(null)
              }}
              className="text-sm text-subtle file:mr-3 file:rounded-sm file:border-0 file:bg-surface-raised file:px-3 file:py-1.5 file:text-sm file:text-fg"
            />
            <button
              type="button"
              onClick={openGuide}
              data-testid="capture-guide"
              title="Capture guide"
              aria-label="Capture guide"
              className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full border-hair border-border-strong text-[11px] text-muted transition-colors duration-100 hover:border-fg/30 hover:text-subtle"
            >
              ?
            </button>
          </div>
          {file && (
            <p className="mt-2 font-mono text-[11px] text-muted" data-testid="chosen-file">
              {file.name} · {(file.size / 1_000_000).toFixed(1)} MB
            </p>
          )}
        </Step>
      </div>

      {error && (
        <p className="mt-4 text-sm text-data-unreachable" data-testid="wizard-error">
          {error}
        </p>
      )}

      <div className="mt-5 flex items-center gap-3">
        <Button variant="primary" onClick={start} disabled={!file || uploading} data-testid="create-workspace">
          {uploading ? 'Uploading…' : 'Create workspace'}
        </Button>
        <span className="font-mono text-[11px] text-muted" data-testid="spec-preview">
          {spec.frames_fps} fps · {spec.mapping} · {spec.semantics} · {spec.chat_llm}
        </span>
      </div>
    </div>
  )
}
