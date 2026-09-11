/** Human-readable label for Scene.vocab_source, for the "which vocab provider was
 * actually used" indicator on the scene detail page. A provider failure now fails the
 * whole scene rather than substituting a vocabulary (see gpu/stage_vocab.py's
 * VocabProviderError), so a "done" scene only ever has "openrouter"/"vertex"/
 * "override" going forward - the default_* cases below are retired but kept so scenes
 * built before that change still display sensibly instead of falling through to the
 * raw, un-translated source string. */
export function vocabSourceLabel(source: string | null, model: string | null): string | null {
  switch (source) {
    case 'openrouter':
      return `OpenRouter${model ? ` · ${model}` : ''}`
    case 'vertex':
      return `Vertex AI${model ? ` · ${model}` : ''}`
    case 'override':
      return 'fixed override'
    case 'default_no_key':
      return 'default vocabulary (no API key configured) - legacy scene'
    case 'default_no_frames':
      return 'default vocabulary (no keyframes) - legacy scene'
    case 'default_after_error':
      return 'default vocabulary (provider call failed) - legacy scene'
    default:
      return source
  }
}
