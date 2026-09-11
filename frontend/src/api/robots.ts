import { apiGet } from './client'
import type { RobotListResponse } from './types'

/** Every registered robot platform (see app/robots.py on the backend) - backs the scene
 * page's platform selector. Static per deployment (the registry is a Python module, not
 * per-scene data), so callers fetch it once and hold it in state rather than re-polling. */
export function getRobots(): Promise<RobotListResponse> {
  return apiGet('/robots')
}
