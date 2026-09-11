// Thin fetch wrapper for the CloudEye backend. All requests go through the Vite dev
// proxy at /api (see vite.config.ts) so no CORS setup is needed on the backend.

export class ApiError extends Error {
  status: number
  body: unknown

  constructor(status: number, body: unknown, message?: string) {
    super(message ?? `Request failed with status ${status}`)
    this.status = status
    this.body = body
  }
}

/** Thrown by scene endpoints (409) while a scene is still queued/processing. */
export class SceneNotReadyError extends ApiError {}

/** Thrown when a resource genuinely doesn't exist (404). */
export class NotFoundError extends ApiError {}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, {
    ...init,
    headers: {
      ...(init?.body && !(init.body instanceof FormData)
        ? { 'Content-Type': 'application/json' }
        : {}),
      ...init?.headers,
    },
  })

  if (!res.ok) {
    let body: unknown = null
    try {
      body = await res.json()
    } catch {
      // no JSON body
    }
    const detail =
      body && typeof body === 'object' && 'detail' in body
        ? String((body as { detail: unknown }).detail)
        : undefined
    if (res.status === 409) throw new SceneNotReadyError(res.status, body, detail)
    if (res.status === 404) throw new NotFoundError(res.status, body, detail)
    throw new ApiError(res.status, body, detail)
  }

  if (res.status === 204) return undefined as T
  return (await res.json()) as T
}

export function apiGet<T>(path: string): Promise<T> {
  return request<T>(path)
}

export function apiPost<T>(path: string, body?: unknown): Promise<T> {
  return request<T>(path, {
    method: 'POST',
    body: body !== undefined ? JSON.stringify(body) : undefined,
  })
}

export function apiPatch<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, { method: 'PATCH', body: JSON.stringify(body) })
}

export function apiDelete(path: string): Promise<void> {
  return request<void>(path, { method: 'DELETE' })
}

export function apiUpload<T>(path: string, formData: FormData): Promise<T> {
  return request<T>(path, { method: 'POST', body: formData })
}
