import { useEffect, useState } from 'react'
import { getScenePreviewBlob } from './scenePreview'

/** Object URL for a scene's cached preview thumbnail (see scenePreview.ts), or null if
 * none is cached yet - null both for "no primary scene" (pass null as `sceneId`) and
 * for "has a primary scene but it's never been opened in 3D, so nothing captured it
 * yet". Revokes its own URL on unmount/change. */
export function useScenePreviewUrl(sceneId: string | null): string | null {
  const [url, setUrl] = useState<string | null>(null)

  useEffect(() => {
    setUrl(null)
    if (!sceneId) return

    let cancelled = false
    let objectUrl: string | null = null
    getScenePreviewBlob(sceneId)
      .then((blob) => {
        if (cancelled || !blob) return
        objectUrl = URL.createObjectURL(blob)
        setUrl(objectUrl)
      })
      .catch(() => {})

    return () => {
      cancelled = true
      if (objectUrl) URL.revokeObjectURL(objectUrl)
    }
  }, [sceneId])

  return url
}
