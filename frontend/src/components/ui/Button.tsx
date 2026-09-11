import { forwardRef } from 'react'
import type { ButtonHTMLAttributes, ReactNode } from 'react'

export type ButtonVariant = 'primary' | 'secondary' | 'ghost'
export type ButtonSize = 'default' | 'compact' | 'large'

interface Props extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant
  size?: ButtonSize
  /** Rendered left of the label, 16px. Never a "->"-shaped affordance - see spec 5.1. */
  icon?: ReactNode
  /** Swaps the label for a spinner WITHOUT changing the button's width - the label stays
   * in the box at zero opacity holding its own size open. A button that narrows the
   * moment it is pressed drags the whole row it sits in with it. */
  loading?: boolean
}

/** Three fills, and only three.
 *
 * primary   the accent green. Exactly one on screen: the Isaac Sim export, the one
 *           action that commits to something outside the app.
 * secondary a flat 10% white. Send, Auto tour - things done often, that should read as
 *           available without competing with the primary.
 * ghost     text only. Reset, and every small action inside a panel.
 *
 * Geometry is identical across all three (spec 5.1) so swapping a variant never moves
 * anything. */
const VARIANT_CLASSES: Record<ButtonVariant, string> = {
  primary: 'bg-accent text-accent-fg hover:bg-accent-hover',
  secondary: 'bg-fg/10 text-fg hover:bg-fg/[0.14]',
  ghost: 'bg-transparent text-subtle hover:bg-fg/[0.04] hover:text-fg',
}

/** Every size clears the 32px minimum hit area; `compact` is 32 exactly rather than the
 * 26 it used to be, because a control being visually small is not a reason for it to be
 * hard to hit. */
const SIZE_CLASSES: Record<ButtonSize, string> = {
  // h-row, not h-8: Tailwind's rem spacing on this app's 15px root makes h-8 30px,
  // two short of the 32px hit area every clickable thing is supposed to clear. The
  // token is in px for exactly that reason.
  default: 'h-row px-s3',
  compact: 'h-row px-s2',
  large: 'h-10 px-s4',
}

/** The class string alone, for callers that need Button's exact look on a non-<button>
 * element - e.g. react-router's <Link> used as the "OPEN SCENE" action (spec 6.3). */
export function buttonClassName(variant: ButtonVariant = 'secondary', size: ButtonSize = 'default'): string {
  return [
    'relative inline-flex shrink-0 items-center justify-center gap-1.5 rounded-sm',
    // Body font, not mono/uppercase: a button is a sentence the user is about to act on,
    // and the mono caps read as telemetry rather than as an offer.
    'text-bodyux font-medium transition-colors duration-100',
    'focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent',
    'disabled:pointer-events-none disabled:opacity-40',
    VARIANT_CLASSES[variant],
    SIZE_CLASSES[size],
  ].join(' ')
}

const Button = forwardRef<HTMLButtonElement, Props>(function Button(
  { variant = 'secondary', size = 'default', icon, loading = false, className = '', children, type = 'button', disabled, ...rest },
  ref,
) {
  return (
    <button
      ref={ref}
      type={type}
      disabled={disabled || loading}
      aria-busy={loading || undefined}
      className={`${buttonClassName(variant, size)} ${className}`}
      {...rest}
    >
      {/* The label keeps its box while the spinner is up - see `loading`. */}
      <span className={`inline-flex items-center gap-1.5 ${loading ? 'invisible' : ''}`}>
        {icon && <span className="flex h-4 w-4 shrink-0 items-center justify-center">{icon}</span>}
        {children}
      </span>
      {loading && (
        <span
          aria-hidden
          data-testid="button-spinner"
          className="absolute h-3.5 w-3.5 animate-spin rounded-full border-2 border-current border-t-transparent opacity-70"
        />
      )}
    </button>
  )
})

export default Button
