/** How long a single click waits to see whether it is really the first half of a
 * double-click. Chosen just above a comfortable double-click interval and well below
 * the point where a single click starts to feel unacknowledged. */
export const DOUBLE_CLICK_GRACE_MS = 220

export interface DeferredClick {
  /** Schedule `run` for after the grace period, replacing any already-scheduled one. */
  click: (run: () => void) => void
  /** Drop a scheduled action - what a double-click calls before doing its own thing. */
  cancel: () => void
  /** True while an action is scheduled. For tests. */
  pending: () => boolean
}

/** Reconciles "click" and "dblclick" on the same element, so the double-click's action
 * happens INSTEAD OF the single click's, not after it.
 *
 * The DOM fires two `click`s before every `dblclick`, and the object list's single
 * click frames the 3D camera on the object. So double-clicking a row to send the robot
 * there also flew the camera into the object - literally: a demo take was thrown away
 * because the camera ended up inside the desk, the canvas 71% desk surface, and the
 * route the shot was supposed to show was not visible at all (HANDOFF 5b, trap 2). The
 * click handler also inserted the object's name into the command box, twice, which is
 * where the `desk desk` in that take's input came from.
 *
 * Deferring the single click by a fraction of a second is the price of a double-click
 * that means one thing. `cancel()` is idempotent and safe to call from an unmount. */
export function createDeferredClick(delayMs: number = DOUBLE_CLICK_GRACE_MS): DeferredClick {
  let timer: ReturnType<typeof setTimeout> | null = null

  function cancel() {
    if (timer !== null) {
      clearTimeout(timer)
      timer = null
    }
  }

  return {
    cancel,
    pending: () => timer !== null,
    click(run: () => void) {
      cancel()
      timer = setTimeout(() => {
        timer = null
        run()
      }, delayMs)
    },
  }
}
