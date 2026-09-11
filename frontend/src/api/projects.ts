import { apiDelete, apiGet, apiPatch } from './client'
import type { ProjectDetailResponse, ProjectListResponse, ProjectResponse } from './types'

export function listProjects(skip = 0, limit = 50, includeArchived = false): Promise<ProjectListResponse> {
  return apiGet(`/projects?skip=${skip}&limit=${limit}&include_archived=${includeArchived}`)
}

export function getProject(id: string): Promise<ProjectDetailResponse> {
  return apiGet(`/projects/${id}`)
}

export function updateProject(
  id: string,
  patch: { name?: string; description?: string; primary_scene_id?: string; archived?: boolean },
): Promise<ProjectResponse> {
  return apiPatch(`/projects/${id}`, patch)
}

export function deleteProject(id: string): Promise<void> {
  return apiDelete(`/projects/${id}`)
}

