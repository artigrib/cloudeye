import { createContext, useContext, useEffect, useState, type ReactNode } from 'react'
import CaptureGuideModal from '../components/CaptureGuideModal'
import { hasSeenCaptureGuide, markCaptureGuideSeen } from '../lib/captureGuide'

interface CaptureGuideContextValue {
  /** Opens the capture guide modal - used by the topbar "?" button and by the
   * "Capture guide" link under a failed-scene error message. */
  openGuide: () => void
}

const CaptureGuideContext = createContext<CaptureGuideContextValue | null>(null)

/** Mounted once at the app shell (Layout). Owns the one auto-shown-on-first-visit /
 * reopenable-any-time capture guide modal, so every page can trigger it without each
 * one tracking its own open state. */
export function CaptureGuideProvider({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false)

  // Auto-show once, on first visit - after that, only explicit opens (the "?" button,
  // or the error-message link) bring it back.
  useEffect(() => {
    if (!hasSeenCaptureGuide()) setOpen(true)
  }, [])

  function close() {
    markCaptureGuideSeen()
    setOpen(false)
  }

  return (
    <CaptureGuideContext.Provider value={{ openGuide: () => setOpen(true) }}>
      {children}
      {open && <CaptureGuideModal onClose={close} />}
    </CaptureGuideContext.Provider>
  )
}

export function useCaptureGuide(): CaptureGuideContextValue {
  const ctx = useContext(CaptureGuideContext)
  if (!ctx) throw new Error('useCaptureGuide must be used within a CaptureGuideProvider')
  return ctx
}
