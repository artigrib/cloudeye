import { CAPTURE_GUIDE_RULES, CAPTURE_GUIDE_UNSUITABLE } from '../lib/captureGuide'

/** The six-rule body shared by the auto-shown modal and the collapsed section in the
 * upload dialog - no chrome (title/buttons) of its own. */
export default function CaptureGuideList() {
  return (
    <div className="flex flex-col gap-3">
      {CAPTURE_GUIDE_RULES.map(({ icon: Icon, title, text }) => (
        <div key={title} className="flex items-start gap-3">
          <Icon size={18} className="mt-0.5 shrink-0 text-accent" stroke={1.75} />
          <p className="text-sm text-subtle">
            <span className="font-medium text-fg">{title}.</span> {text}
          </p>
        </div>
      ))}
      <p className="mt-1 border-t-hair border-border pt-3 text-xs text-muted">{CAPTURE_GUIDE_UNSUITABLE}</p>
    </div>
  )
}
