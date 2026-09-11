import { probeGlb, type GlbFacts } from './glbProbe'

/** Does this layer's endpoint answer at all?
 *
 * A one-byte Range request, not a HEAD: the API's routes are declared `GET` only and
 * Starlette does not synthesise a HEAD for them (checked - /usd answers 405 to a HEAD),
 * while every one of these is a FileResponse, which serves Range natively. One byte is
 * enough to tell 200/206 from 404 and costs nothing against a 44 MB export.
 */
export async function probeLayerAvailable(url: string | null | undefined, signal?: AbortSignal): Promise<boolean> {
  if (!url) return false
  try {
    const res = await fetch(url, { headers: { Range: 'bytes=0-0' }, signal })
    return res.ok
  } catch {
    // A transport failure is not "this scene has no layout" - but there is nothing else to
    // show for it either, and a toggle that errors on press is worse than one that is
    // off. Reported as unavailable; the console line comes from the loader if the user
    // switches it on anyway after a later probe succeeds.
    return false
  }
}

/** Which file the Points layer is actually reading, and why.
 *
 * /cloud when it answers, because that is the source of record (CEPC binary, parsed in a
 * worker). Otherwise /mesh, and ONLY if /mesh is a points primitive - a triangulated
 * export is the Mesh layer's file and drawing it as Points would be a second, wrong
 * picture of the same data. */
export type PointsSource = 'cloud' | 'mesh' | 'none'

export interface SceneLayerFacts {
  /** Where Points reads from, resolved against what the two endpoints actually contain. */
  pointsSource: PointsSource
  /** What /mesh is. `triangles` is what enables the Mesh toggle. */
  mesh: GlbFacts
  /** Whether /msa-glb answers at all - the Layout toggle needs nothing more than that. */
  layout: boolean
  /** One line per scene, logged: which source Points used and why the other toggles are
   * in the state they are in. */
  log: string
}

export async function resolveSceneLayers(
  urls: { cloud?: string | null; mesh?: string | null; msaGlb?: string | null },
  signal?: AbortSignal,
): Promise<SceneLayerFacts> {
  const [cloudOk, mesh, layout] = await Promise.all([
    probeLayerAvailable(urls.cloud, signal),
    probeGlb(urls.mesh, signal),
    probeLayerAvailable(urls.msaGlb, signal),
  ])

  const pointsSource: PointsSource = cloudOk ? 'cloud' : mesh.ok && mesh.points ? 'mesh' : 'none'
  const log =
    `points <- ${pointsSource}` +
    (cloudOk ? ' (/cloud 200)' : ' (/cloud 404)') +
    `; /mesh ${mesh.ok ? mesh.reason : `unavailable (${mesh.reason})`}` +
    `; mesh toggle ${mesh.triangles ? 'enabled' : 'disabled'}` +
    `; layout ${layout ? 'enabled' : 'disabled'}`

  return { pointsSource, mesh, layout, log }
}
