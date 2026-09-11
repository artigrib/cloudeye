/** What a GLB actually contains, read from its JSON chunk without downloading it.
 *
 * The Mesh layer exists to draw a triangulated surface. `/api/scenes/<id>/mesh` is named
 * for one and usually is not one: its own route docstring says it serves the
 * points-primitive glTF from `gpu/stage_export_glb.py`. Asking the file rather than the
 * URL is the only way to tell, and the answer decides both whether the Mesh toggle is
 * enabled and whether the Points layer may fall back to that file.
 *
 * Two Range requests, ~1 KB of a 44 MB export: the 20-byte GLB header (magic, version,
 * total length, chunk-0 length, chunk-0 type) and then chunk 0, which the spec requires
 * to be the JSON. These routes are FileResponses and serve Range natively; they are
 * GET-only, so a HEAD is not an option (checked - /usd answers 405).
 */
export interface GlbFacts {
  /** The endpoint answered and the bytes are a glTF binary. */
  ok: boolean
  /** At least one primitive draws triangles (mode 4/5/6, or an absent mode, which the
   * spec defaults to 4). */
  triangles: boolean
  /** At least one primitive draws points (mode 0). */
  points: boolean
  /** Triangle count, summed over triangle primitives: indices/3 where a primitive is
   * indexed, POSITION/3 where it is not. 0 is a real answer and is what disables the
   * Mesh toggle on a points-only export. */
  faces: number
  /** Why, in a few words - printed to the console so "the toggle is greyed out" is
   * always traceable to a fact about the file. */
  reason: string
}

const NONE: GlbFacts = { ok: false, triangles: false, points: false, faces: 0, reason: 'not fetched' }

const GLB_MAGIC = 0x46546c67 // 'glTF', little-endian
const CHUNK_JSON = 0x4e4f534a // 'JSON'
/** A glTF JSON chunk larger than this is not something we need to read to answer
 * "does it have faces" - and a header claiming one is a reason to stop, not to fetch it. */
const MAX_JSON_CHUNK = 8 * 1024 * 1024

async function range(url: string, from: number, to: number, signal?: AbortSignal): Promise<ArrayBuffer | null> {
  const res = await fetch(url, { headers: { Range: `bytes=${from}-${to}` }, signal })
  if (!res.ok) return null
  const buf = await res.arrayBuffer()
  // A server that ignores Range answers 200 with the whole file. Slice to what was asked
  // for so the offsets below mean the same thing either way.
  return buf.byteLength > to - from + 1 ? buf.slice(from, to + 1) : buf
}

export async function probeGlb(url: string | null | undefined, signal?: AbortSignal): Promise<GlbFacts> {
  if (!url) return { ...NONE, reason: 'no url' }
  try {
    const head = await range(url, 0, 19, signal)
    if (!head || head.byteLength < 20) return { ...NONE, reason: 'no response (404 or empty)' }
    const dv = new DataView(head)
    if (dv.getUint32(0, true) !== GLB_MAGIC) return { ...NONE, reason: 'not a glTF binary' }
    const jsonLen = dv.getUint32(12, true)
    if (dv.getUint32(16, true) !== CHUNK_JSON) return { ...NONE, reason: 'first chunk is not JSON' }
    if (jsonLen === 0 || jsonLen > MAX_JSON_CHUNK) return { ...NONE, reason: `implausible JSON chunk (${jsonLen} B)` }

    const jsonBuf = await range(url, 20, 20 + jsonLen - 1, signal)
    if (!jsonBuf) return { ...NONE, reason: 'JSON chunk not served' }
    const doc = JSON.parse(new TextDecoder().decode(jsonBuf)) as {
      meshes?: { primitives?: { mode?: number; indices?: number; attributes?: Record<string, number> }[] }[]
      accessors?: { count?: number }[]
    }

    const accessorCount = (i: number | undefined): number =>
      i === undefined ? 0 : (doc.accessors?.[i]?.count ?? 0)

    let triangles = false
    let points = false
    let faces = 0
    for (const mesh of doc.meshes ?? []) {
      for (const prim of mesh.primitives ?? []) {
        // "mode" is optional and defaults to 4 (TRIANGLES) per the glTF spec - an absent
        // mode is a triangle primitive, not an unknown one.
        const mode = prim.mode ?? 4
        if (mode === 0) points = true
        if (mode === 4 || mode === 5 || mode === 6) {
          triangles = true
          const verts = prim.indices !== undefined
            ? accessorCount(prim.indices)
            : accessorCount(prim.attributes?.POSITION)
          // TRIANGLES 3 per face; STRIP and FAN are n-2. Close enough for "> 0", and
          // exact for the only case that matters (an indexed TRIANGLES export).
          faces += mode === 4 ? Math.floor(verts / 3) : Math.max(0, verts - 2)
        }
      }
    }
    return {
      ok: true,
      triangles: triangles && faces > 0,
      points,
      faces,
      reason: faces > 0 ? `${faces} faces` : points ? 'points primitive, 0 faces' : 'no drawable primitives',
    }
  } catch (err) {
    // An aborted probe is an ordinary scene switch, not a fault. Anything else is a
    // transport failure, and the only honest answer for it is "cannot say", which reads
    // as unavailable - a toggle that errors on press is worse than one that is off.
    if ((err as Error)?.name === 'AbortError') return { ...NONE, reason: 'aborted' }
    return { ...NONE, reason: `probe failed: ${(err as Error)?.message ?? 'unknown'}` }
  }
}
