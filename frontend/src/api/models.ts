import { apiGet } from './client'
import type { ModelStageResponse, ProviderHealthResponse } from './types'

export function getModelCatalog(): Promise<ModelStageResponse[]> {
  return apiGet('/models/catalog')
}

export function getVocabHealth(force = false): Promise<Record<string, ProviderHealthResponse>> {
  return apiGet(`/models/vocab-health${force ? '?force=true' : ''}`)
}
