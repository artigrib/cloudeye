import { apiDelete, apiGet, apiUpload } from './client'
import type { SceneStatusResponse, VideoResponse } from './types'

import type { JobSpecRequest } from '../lib/jobSpec'

export interface UploadOptions {
  /** Override the keyframe stage's blur filter for this video only (default: 100.0). */
  blurThreshold?: number
  /** Override the keyframe stage's similarity filter for this video only (default: 0.98). */
  similarityThreshold?: number
  /** Which scene-vocabulary provider to use ("openrouter"/"vertex"). Omit for the default. */
  vocabProvider?: string
  /** The New-workspace wizard's choices - see lib/jobSpec.ts and docs/JOB_SPEC.md.
   * Serialised into the `job_spec` form field; the server validates it, fills in
   * `video_id`, and saves it beside the video as `<video_uuid>.job.json`. */
  jobSpec?: JobSpecRequest
}

export function uploadVideo(file: File, projectId?: string, opts?: UploadOptions): Promise<VideoResponse> {
  const form = new FormData()
  form.append('file', file)
  if (projectId) form.append('project_id', projectId)
  if (opts?.blurThreshold !== undefined) form.append('blur_threshold', String(opts.blurThreshold))
  if (opts?.similarityThreshold !== undefined)
    form.append('similarity_threshold', String(opts.similarityThreshold))
  if (opts?.vocabProvider) form.append('vocab_provider', opts.vocabProvider)
  // The wizard's job spec, verbatim JSON (docs/JOB_SPEC.md). The server fills in
  // `video_id` from the row it creates, so it is deliberately absent here.
  if (opts?.jobSpec) form.append('job_spec', JSON.stringify(opts.jobSpec))
  return apiUpload('/videos/upload', form)
}

export function getVideo(id: string): Promise<VideoResponse> {
  return apiGet(`/videos/${id}`)
}

export function getVideoScene(videoId: string): Promise<SceneStatusResponse> {
  return apiGet(`/videos/${videoId}/scene`)
}

export function deleteVideo(id: string): Promise<void> {
  return apiDelete(`/videos/${id}`)
}

export function videoFileUrl(id: string): string {
  return `/api/videos/${id}/file`
}
