import { useSearchParams } from 'react-router-dom'
import MsaSceneViewer from '../components/MsaSceneViewer'

const DEFAULT_GLB = '/msa-fixtures/02_modular_home.glb'
const DEFAULT_GAPS = '/msa-fixtures/02_modular_home_gaps.json'

/** Not linked from anywhere in the app chrome - MSA scenes are file-based script
 * output (scripts/msa/*.py), not part of the DB-backed pipeline the rest of the app
 * routes through, so there is no scene id to route on yet. Defaults to the
 * `02_modular_home` bootstrap fixture checked into `frontend/public/msa-fixtures/`
 * (see that directory's README); `?glb=<url>&gaps=<url>` overrides both (a
 * same-origin static file, e.g. dropped under `frontend/public/`, or any URL the
 * browser may fetch) - how the T15c hero dollhouse screenshots were taken. Pass
 * `gaps=` empty to hide the gaps layer. A real project would take these from a
 * future MSA-aware API route. */
export default function MsaViewerPage() {
  const [params] = useSearchParams()
  const glbUrl = params.get('glb') || DEFAULT_GLB
  const gapsParam = params.get('gaps')
  const gapsUrl = gapsParam === null ? DEFAULT_GAPS : gapsParam || undefined
  return (
    <div className="flex h-screen flex-col bg-canvas p-4">
      <h1 className="mb-2 font-mono text-[11px] uppercase tracking-[0.06em] text-muted">MSA scene viewer (dev)</h1>
      <MsaSceneViewer glbUrl={glbUrl} gapsUrl={gapsUrl} />
    </div>
  )
}
