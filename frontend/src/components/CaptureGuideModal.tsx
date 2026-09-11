import Button from './ui/Button'
import Modal from './ui/Modal'
import CaptureGuideList from './CaptureGuideList'

interface Props {
  onClose: () => void
}

/** The full-content capture-onboarding panel: auto-shown once (see CaptureGuideProvider)
 * and reopenable any time via the "?" button. Deliberately not a full-screen takeover -
 * a compact centered panel, single page, no wizard steps. */
export default function CaptureGuideModal({ onClose }: Props) {
  return (
    <Modal onClose={onClose} className="max-w-md p-5">
      <h3 className="text-sm font-medium text-fg">Capture guide</h3>
      <p className="mt-1 text-xs text-muted">
        The reconstruction pipeline is sensitive to how the video was shot. A few rules can save you a reshoot.
      </p>

      <div className="mt-4">
        <CaptureGuideList />
      </div>

      <div className="mt-4 flex justify-end">
        <Button variant="primary" onClick={onClose}>
          Got it, upload video
        </Button>
      </div>
    </Modal>
  )
}
