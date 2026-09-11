import { useState } from 'react'
import { videoFileUrl } from '../api/videos'
import VideoPlayerControls from './ui/VideoPlayer'

/** The scene's source video, docked in the left column: a title bar with the filename and
 * a × to hide it, over one 16:9 letterboxed player.
 *
 * There is ONE of these on the screen and it does not float. It used to sit bottom-left
 * over the map and expand to half the canvas on a click (Esc to collapse); commit 008e6d4
 * docked it, and the `expanded` state and its Esc handler outlived their last reader by
 * some time - nothing rendered differently for either. They are gone. Floating over the
 * viewer is now the browser's job: ui/VideoPlayer's picture-in-picture button hands the
 * video to a real OS window the viewer can put anywhere, instead of an in-app overlay that
 * covered the clearance legend and could not be moved. */
export default function VideoPlayer({
  videoId,
  filename,
  onTimeUpdate,
}: {
  videoId: string
  filename?: string
  /** Fires on every native `timeupdate` (during playback, and after a scrub while
   * paused) with the video's current playback position - backs the camera-path view's
   * "highlight the pose closest to now" sync. Omit if nothing needs it. */
  onTimeUpdate?: (seconds: number) => void
}) {
  const [hidden, setHidden] = useState(false)

  if (hidden) {
    return (
      <button
        type="button"
        onClick={() => setHidden(false)}
        data-testid="show-video"
        className="hit-target mx-s4 flex items-center rounded-sm border-hair border-border bg-surface px-s2 text-caption text-subtle transition-colors duration-100 hover:bg-fg/[0.04] hover:text-fg"
      >
        Show video
      </button>
    )
  }

  return (
    <div
      // In the left panel now, not floating over the canvas. It used to sit bottom-left
      // over the map, where it covered the clearance legend and the "scroll to zoom"
      // hint - a permanent overlap that the overlap check could not see, because none of
      // what it covered was clickable. The source video is scene material, and this is
      // the scene column.
      // `flex-1` + `min-h-0`: the video is the last thing in the column and takes what is
      // left of it, so the column's spare height goes into the picture instead of into a
      // gap under a fixed stack - which is what used to make it scroll at 1440x900.
      className="mx-s4 flex min-h-0 flex-1 flex-col overflow-hidden rounded-card border-hair border-card-border"
      data-testid="video-card"
    >
      <div
        data-testid="video-titlebar"
        className="flex items-center justify-between gap-2 bg-surface-raised px-2 py-1"
      >
        <span className="truncate text-[11px] text-subtle" title={filename}>
          {filename ?? 'Video'}
        </span>
        <button
          type="button"
          onClick={() => setHidden(true)}
          className="flex h-row w-row shrink-0 items-center justify-center rounded-sm leading-none text-muted transition-colors duration-100 hover:bg-fg/[0.04] hover:text-subtle focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
          aria-label="Hide video"
          data-testid="hide-video"
        >
          ×
        </button>
      </div>
      <VideoPlayerControls
        src={videoFileUrl(videoId)}
        onTimeUpdate={onTimeUpdate}
        className="min-h-0 flex-1"
      />
    </div>
  )
}
