/// <reference lib="webworker" />
/** Fetches and parses a CEPC cloud off the main thread.
 *
 * Everything expensive happens here: the download, the length/magic checks, the copy into
 * typed arrays and the Z-up -> Y-up rotation over every point. The main thread receives two
 * already-built buffers as transferables, so building the THREE.BufferAttributes on the
 * other side is O(1) - no per-point work ever runs on the thread that renders.
 *
 * A 404 is reported as `notFound` rather than an error: scenes produced by the ordinary GPU
 * pipeline have no cloud.bin, only the points-primitive GLB, and the caller falls back to
 * that. Any other failure is a real error.
 */
import { parseCloudBin } from './cloudBin'

export interface CloudWorkerRequest {
  url: string
}

export type CloudWorkerResponse =
  | { ok: true; positions: Float32Array; colors: Uint8Array | null; count: number; bboxZUp: Float32Array }
  | { ok: false; notFound: true }
  | { ok: false; notFound: false; error: string }

self.onmessage = async (event: MessageEvent<CloudWorkerRequest>) => {
  const post = (msg: CloudWorkerResponse, transfer: Transferable[] = []) =>
    (self as unknown as Worker).postMessage(msg, transfer)
  try {
    const res = await fetch(event.data.url)
    if (res.status === 404) {
      post({ ok: false, notFound: true })
      return
    }
    if (!res.ok) {
      post({ ok: false, notFound: false, error: `HTTP ${res.status}` })
      return
    }
    const { positions, colors, count, bboxZUp } = parseCloudBin(await res.arrayBuffer())
    const transfer: Transferable[] = [positions.buffer, bboxZUp.buffer]
    if (colors) transfer.push(colors.buffer)
    post({ ok: true, positions, colors, count, bboxZUp }, transfer)
  } catch (err) {
    post({ ok: false, notFound: false, error: err instanceof Error ? err.message : String(err) })
  }
}
