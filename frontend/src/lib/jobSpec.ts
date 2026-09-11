// The New-workspace wizard's job spec, client side. The contract, the defaults and the
// reasons live in docs/JOB_SPEC.md and docs/job_spec.schema.json; this file must not
// disagree with them. Pure data + labels, no React - the wizard renders it, the upload
// client serialises it.

export const FRAMES_FPS_DEFAULT = 15

export type Semantics = 'openrouter-glm' | 'vertex-gemma'
export type ChatLlm = 'openrouter-nemotron' | 'vertex-gemma'

/** What the client sends. `video_id` is deliberately absent: the server fills it in from
 * the row the upload creates, so a request cannot point its spec at another video. */
export interface JobSpecRequest {
  spec_version: 1
  frames_fps: number
  mapping: 'mapanything'
  semantics: Semantics
  chat_llm: ChatLlm
  robots: string[]
}

export const DEFAULT_JOB_SPEC: JobSpecRequest = {
  spec_version: 1,
  frames_fps: FRAMES_FPS_DEFAULT,
  mapping: 'mapanything',
  semantics: 'openrouter-glm',
  chat_llm: 'openrouter-nemotron',
  robots: [],
}

/** Frame rates the wizard offers. 15 is the measured knee on the hero footage (HANDOFF
 * section 4: 574 -> 1148 views buys +4.70 points, 1148 -> 2295 only +1.81), which is why
 * it is the default - but the operator picking a rate does not need that argument, so the
 * option just says "default" and the reasoning stays here. The other three keep their view
 * counts: those ARE decision-relevant, being what the choice costs. */
export const FPS_CHOICES: { value: number; label: string; note: string }[] = [
  { value: 7.5, label: '7.5 fps', note: 'fastest, ~574 views on a 76 s walk' },
  { value: 10, label: '10 fps', note: '~765 views' },
  { value: 15, label: '15 fps', note: 'default' },
  { value: 30, label: '30 fps', note: 'every frame; +1.81 pts over 15 fps, and twice the GPU time' },
]

export const SEMANTICS_CHOICES: { value: Semantics; label: string; note: string }[] = [
  { value: 'openrouter-glm', label: 'OpenRouter', note: 'z-ai/glm-5.3-flash' },
  { value: 'vertex-gemma', label: 'Vertex AI', note: 'google/gemma-4-26b-a4b-it-maas' },
]

/** What the wizard offers for the typed-command parser.
 *
 * ONE ENTRY, on purpose. `vertex-gemma` is still a valid `ChatLlm` - the type keeps it so
 * job specs already recorded with it still validate, and `vlm_client` can still dispatch
 * to it - but it is not offered, because on this deployment it cannot answer: chat through
 * Vertex needs `gcp_project_id`, which is unset, and the Vertex direction for this field
 * was closed on 2026-09-10. The onboarding worktree verified read-only that
 * `nvidia/nemotron-3-nano-30b-a3b` is not in the Vertex catalogue at all, and that the
 * Nemotron entries which do exist are deploy-only against zero GPU quota
 * (`docs/NEMOTRON_VERTEX.md` on that branch).
 *
 * A picker entry that cannot be honoured is worse than a short list: it reads as a
 * capability, and the only way to find out otherwise is to ship a workspace with it. */
export const CHAT_LLM_CHOICES: { value: ChatLlm; label: string; note: string }[] = [
  { value: 'openrouter-nemotron', label: 'OpenRouter', note: 'nvidia/nemotron-3-nano-30b-a3b' },
]
