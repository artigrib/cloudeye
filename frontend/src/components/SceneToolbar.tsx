import { useState } from 'react'
import { sceneUsdUrl } from '../api/scenes'
import Checkbox from './ui/Checkbox'
import { buttonClassName } from './ui/Button'

interface Props {
  sceneId: string
  /** The screen's one answer, at the far left - see ScenePage's `verdict`. */
  verdict?: React.ReactNode
  compareAll: boolean
  onToggleCompareAll: () => void
  /** How many objects the agent's `move` command has shifted. Non-zero means every
   * number on this screen is about a room that does not exist, which has to be visible
   * without reading back through the chat. */
  movedCount: number
  onResetMoves: () => void
}

/** The screen's header bar: the verdict, and the one action that commits to something
 * outside the app.
 *
 * The EXPORT disclosure that used to live here - a "show/hide" that opened a panel
 * containing a checkbox and a link - is gone. Exporting is the screen's primary action;
 * putting it behind a toggle made it the only thing on the bar that could not be done in
 * one press, and the panel it opened held exactly one option. The option is a checkbox on
 * the bar, and the export is a button.
 *
 * "camera path", "show fragments" and "mesh" moved to the canvas toolbar (SceneView).
 * Each of them changes what the CANVAS draws, and they were on this bar only because
 * this bar existed. */
export default function SceneToolbar({
  sceneId,
  verdict,
  compareAll,
  onToggleCompareAll,
  movedCount,
  onResetMoves,
}: Props) {
  const [includePointcloud, setIncludePointcloud] = useState(true)

  return (
    <div
      className="flex h-12 shrink-0 items-center gap-s3 border-b-hair border-border bg-surface px-s4"
      data-testid="scene-toolbar"
    >
      <div className="min-w-0 flex-1">{verdict}</div>

      {movedCount > 0 && (
        <button
          type="button"
          onClick={onResetMoves}
          data-testid="reset-moves"
          title="Put every moved object back where it was scanned"
          className="hit-target shrink-0 rounded-sm border-hair border-status-blocked/40 bg-status-blocked/10 px-s2 text-caption text-status-blocked"
        >
          {movedCount} moved · approximate · reset
        </button>
      )}

      <button
        type="button"
        onClick={onToggleCompareAll}
        aria-pressed={compareAll}
        data-testid="compare-all"
        className={`${buttonClassName(compareAll ? 'secondary' : 'ghost')} shrink-0`}
      >
        Compare all
      </button>

      <Checkbox
        checked={includePointcloud}
        onChange={setIncludePointcloud}
        label="Include point cloud"
        className="shrink-0"
        data-testid="include-pointcloud"
      />

      <a
        href={sceneUsdUrl(sceneId, { includePointcloud })}
        className={`${buttonClassName('primary')} shrink-0`}
        data-testid="download-usd"
      >
        Export to Isaac Sim
      </a>
    </div>
  )
}
