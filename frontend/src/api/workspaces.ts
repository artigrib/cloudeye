import { apiGet } from './client'
import type { ProcessingStatusResponse } from './types'

/** Everything the processing screen renders, on every poll - see
 * app/routers/workspaces.py. The screen keeps no state of its own, so this call is the
 * whole of what it shows. */
export function getProcessingStatus(workspaceId: string): Promise<ProcessingStatusResponse> {
  return apiGet(`/workspaces/${workspaceId}/processing`)
}
