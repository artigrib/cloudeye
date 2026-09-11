import { useEffect } from 'react'
import type { ReactNode } from 'react'

interface Props {
  onClose: () => void
  children: ReactNode
  className?: string
}

/** The centered dialog shell, shared by CaptureGuideModal and WorkspaceHeader's delete
 * confirmation. Clicking the scrim closes; the dialog itself doesn't propagate the click.
 * Esc closes too (spec 9's "Esc - close popover" hotkey). */
export default function Modal({ onClose, children, className = '' }: Props) {
  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [onClose])

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-bg/60 p-4" onClick={onClose}>
      <div
        onClick={(e) => e.stopPropagation()}
        className={`w-full max-w-md rounded-lg border-hair border-border bg-surface-raised shadow-xl ${className}`}
      >
        {children}
      </div>
    </div>
  )
}
