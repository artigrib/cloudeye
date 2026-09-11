import { useEffect, useRef, useState } from 'react'
import { cancelSceneGlbLoad, loadSceneGlb, peekSceneGlb, type GlbLoadProgress, type SceneCloudSource } from './glbCache'

export interface UseSceneGlbState {
  gltf: SceneCloudSource | null
  progress: GlbLoadProgress | null
  error: string | null
  loading: boolean
}

/** Lazily loads a scene's point-cloud GLB - only once `enabled` (the first switch into
 * 3D view), never at scene-open time, since it's a 130-150MB file. Cached per scene id
 * (see glbCache), so a later switch back into 3D is instant. */
export function useSceneGlb(
  sceneId: string | null,
  url: string,
  enabled: boolean,
  cloudUrl?: string,
): UseSceneGlbState {
  const [state, setState] = useState<UseSceneGlbState>(() => ({
    gltf: sceneId ? peekSceneGlb(sceneId) : null,
    progress: null,
    error: null,
    loading: false,
  }))

  useEffect(() => {
    if (!enabled || !sceneId) return

    const cachedGltf = peekSceneGlb(sceneId)
    if (cachedGltf) {
      setState({ gltf: cachedGltf, progress: null, error: null, loading: false })
      return
    }

    let cancelled = false
    setState({ gltf: null, progress: null, error: null, loading: true })

    loadSceneGlb(sceneId, url, (progress) => {
      if (!cancelled) setState((s) => ({ ...s, progress }))
    }, cloudUrl)
      .then((gltf) => {
        if (!cancelled) setState({ gltf, progress: null, error: null, loading: false })
      })
      .catch(() => {
        if (!cancelled) {
          setState({ gltf: null, progress: null, loading: false, error: 'Failed to load the 3D model.' })
        }
      })

    return () => {
      cancelled = true
    }
  }, [sceneId, url, enabled, cloudUrl])

  // Leaving the scene entirely (not just toggling back to 2D) cancels a still-running
  // download, per scene id. The cancel itself is deferred to a macrotask, and skipped
  // if this effect mounts again before it fires: React StrictMode's dev-mode
  // mount->cleanup->remount happens synchronously within one tick, and three.js's
  // FileLoader dedupes concurrent loads of the same URL - so a synchronous cancel here
  // would abort the shared in-flight request the very next (StrictMode-simulated)
  // remount is about to reuse, not just this instance's interest in it.
  const pendingCancelRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  useEffect(() => {
    if (pendingCancelRef.current) {
      clearTimeout(pendingCancelRef.current)
      pendingCancelRef.current = null
    }
    return () => {
      if (sceneId) {
        pendingCancelRef.current = setTimeout(() => cancelSceneGlbLoad(sceneId), 0)
      }
    }
  }, [sceneId])

  return state
}
