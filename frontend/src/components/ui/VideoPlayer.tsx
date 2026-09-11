import { useEffect, useRef, useState } from 'react'
import {
  IconMaximize,
  IconPictureInPicture,
  IconPlayerPauseFilled,
  IconPlayerPlayFilled,
  IconX,
} from '@tabler/icons-react'
import Slider from './Slider'

interface Props {
  src: string
  /** Fires on every timeupdate (during playback, and after a scrub while paused). */
  onTimeUpdate?: (seconds: number) => void
  onClose?: () => void
  className?: string
}

function formatTime(seconds: number): string {
  if (!Number.isFinite(seconds)) return '0:00'
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}

/** Replaces the native <video controls> - spec 5.10. Minimal set: play/pause, scrubber,
 * time, picture-in-picture, fullscreen, close.
 *
 * The frame's height comes from OUTSIDE - the caller's flex slot - and the picture
 * letterboxes inside it with `object-contain`. It was a fixed 16:9 box (`aspect-video`)
 * for one reason and that reason still holds: the element's own intrinsic ratio made the
 * panel below it jump the moment metadata arrived, and with `preload="none"` (see
 * `activated`) metadata does not arrive until the first press of Play, so the jump landed
 * in the middle of an interaction. A slot the parent decides keeps that - the box is the
 * same size before and after metadata - and additionally spends the left column's spare
 * height on the picture instead of on a 16:9 crop of it. Portrait footage (which is what
 * every scene here was shot on) and landscape footage both fit; neither is cropped. */
export default function VideoPlayer({ src, onTimeUpdate, onClose, className = '' }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [playing, setPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  // The `src` attribute is withheld until the viewer actually asks to play. A <video>
  // with a src fetches on its own the moment it mounts, and this player mounts on every
  // scene open: `preload="metadata"` still opened a Range request against the source
  // upload (~190 MB responses were measured on scene open, competing with the occupancy
  // grid and the point cloud for the same connection budget) for a preview most sessions
  // never play. `preload="none"` alone is a HINT browsers may ignore; not giving the
  // element a source at all is not.
  const [activated, setActivated] = useState(false)

  function togglePlay() {
    const v = videoRef.current
    if (!v) return
    if (!activated) {
      // First press: the element has no source yet, so there is nothing to play. Attach
      // it here and let the effect below start playback once React has committed the
      // attribute - calling play() now would reject on an empty element.
      setActivated(true)
      return
    }
    if (v.paused) void v.play()
    else v.pause()
  }

  // Starts the very first playback, once `activated` has actually put `src` on the
  // element. Deliberately keyed on `activated` alone: it flips false -> true exactly
  // once, so this can never re-trigger playback the viewer paused.
  useEffect(() => {
    if (!activated) return
    void videoRef.current?.play()
  }, [activated])

  function seek(t: number) {
    const v = videoRef.current
    if (!v) return
    v.currentTime = t
    setCurrentTime(t)
  }

  function toggleFullscreen() {
    containerRef.current?.requestFullscreen?.()
  }

  // Picture-in-picture: the browser's own floating window, which the viewer can put
  // anywhere and keep while scrolling - NOT a second copy of the player inside the app.
  // An in-app overlay is what commit 008e6d4 removed from over the canvas, where it
  // covered the clearance legend permanently.
  //
  // Feature-detected rather than assumed: Firefox exposes no `requestPictureInPicture` on
  // the element, and a browser can disable it entirely. Hidden when it is not there,
  // because a control that does nothing is worse than no control.
  const [pipSupported, setPipSupported] = useState(false)
  useEffect(() => {
    setPipSupported(
      typeof document !== 'undefined' &&
        document.pictureInPictureEnabled === true &&
        typeof videoRef.current?.requestPictureInPicture === 'function',
    )
  }, [])

  async function togglePip() {
    const v = videoRef.current
    if (!v) return
    try {
      if (document.pictureInPictureElement) await document.exitPictureInPicture()
      // A source is only attached on the first Play (see `activated`), and PiP on an
      // element with no source throws. Attach it, then let the effect above start it -
      // the same order togglePlay uses.
      else if (!activated) setActivated(true)
      else await v.requestPictureInPicture()
    } catch {
      // The browser refused it (no user gesture it recognised, PiP disabled by policy,
      // metadata not loaded yet). Nothing to recover: the video keeps playing where it is.
    }
  }

  useEffect(() => {
    const v = videoRef.current
    if (!v) return
    function onTime() {
      setCurrentTime(v!.currentTime)
      onTimeUpdate?.(v!.currentTime)
    }
    function onLoaded() {
      setDuration(v!.duration)
    }
    v.addEventListener('timeupdate', onTime)
    v.addEventListener('loadedmetadata', onLoaded)
    v.addEventListener('play', () => setPlaying(true))
    v.addEventListener('pause', () => setPlaying(false))
    return () => {
      v.removeEventListener('timeupdate', onTime)
      v.removeEventListener('loadedmetadata', onLoaded)
    }
  }, [onTimeUpdate])

  return (
    <div ref={containerRef} className={`relative flex min-h-0 flex-col bg-canvas ${className}`}>
      <video
        ref={videoRef}
        src={activated ? src : undefined}
        preload="none"
        className="block min-h-0 w-full flex-1 bg-black object-contain"
        onClick={togglePlay}
      />

      <div className="flex h-toolbar shrink-0 items-center gap-s2 bg-canvas/80 px-s2 backdrop-blur-sm">
        <button
          type="button"
          onClick={togglePlay}
          aria-label={playing ? 'Pause' : 'Play'}
          className="flex h-row w-row shrink-0 items-center justify-center rounded-sm text-subtle transition-colors duration-100 hover:bg-fg/[0.04] hover:text-fg focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
        >
          {playing ? <IconPlayerPauseFilled size={14} /> : <IconPlayerPlayFilled size={14} />}
        </button>

        <Slider
          value={currentTime}
          onChange={seek}
          min={0}
          max={duration || 0}
          step={0.1}
          formatValue={null}
          fillClassName="bg-fg"
          aria-label="Seek"
        />

        <span className="shrink-0 font-mono text-[11px] tabular-nums text-subtle">
          {formatTime(currentTime)} / {formatTime(duration)}
        </span>

        {pipSupported && (
          <button
            type="button"
            onClick={() => void togglePip()}
            aria-label="Picture in picture"
            data-testid="video-pip"
            className="flex h-row w-row shrink-0 items-center justify-center rounded-sm text-subtle transition-colors duration-100 hover:bg-fg/[0.04] hover:text-fg focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
          >
            <IconPictureInPicture size={14} />
          </button>
        )}

        <button
          type="button"
          onClick={toggleFullscreen}
          aria-label="Fullscreen"
          className="flex h-row w-row shrink-0 items-center justify-center rounded-sm text-subtle transition-colors duration-100 hover:bg-fg/[0.04] hover:text-fg focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
        >
          <IconMaximize size={14} />
        </button>

        {onClose && (
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="flex h-row w-row shrink-0 items-center justify-center rounded-sm text-subtle transition-colors duration-100 hover:bg-fg/[0.04] hover:text-fg focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
          >
            <IconX size={14} />
          </button>
        )}
      </div>
    </div>
  )
}
