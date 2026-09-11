import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { IconCheck, IconChevronDown } from '@tabler/icons-react'

export interface SelectOption<T extends string> {
  value: T
  label: string
  /** Mono second line, e.g. "0.25 m · quadruped" for a robot platform (spec 5.3). */
  sublabel?: string
  /** e.g. a model provider that's currently unhealthy - shown but not selectable. */
  disabled?: boolean
}

interface Props<T extends string> {
  value: T | null
  onChange: (value: T) => void
  options: SelectOption<T>[]
  placeholder?: string
  disabled?: boolean
  'aria-label'?: string
  className?: string
  /** Declared and forwarded explicitly - TSX drops an undeclared `data-*` prop on a
   * component silently (see Checkbox/Badge, same trap). Lands on the trigger button. */
  'data-testid'?: string
}

/** Replaces the native <select> - spec 5.3. Trigger reads like a secondary button with
 * a chevron; the menu is a floating panel, not the browser's native popup, so it always
 * matches the rest of the chrome. */
export default function Select<T extends string>({
  value,
  onChange,
  options,
  placeholder = 'Select…',
  disabled,
  'aria-label': ariaLabel,
  className = '',
  'data-testid': testId,
}: Props<T>) {
  const [open, setOpen] = useState(false)
  // Whether the menu opens above the trigger instead of below - measured, not assumed,
  // each time the menu opens (see the layout effect below). Without this, a trigger
  // near the bottom of the viewport (e.g. the 3D view's Ceiling select, docked in the
  // bottom-right display-controls cluster) opens a menu that's mostly or entirely
  // off-screen, per the app's own hard 1280px-floor/fixed-viewport layout - there's no
  // page scroll to reveal it.
  const [openUpward, setOpenUpward] = useState(false)
  const rootRef = useRef<HTMLDivElement>(null)
  const menuRef = useRef<HTMLDivElement>(null)
  const selected = options.find((o) => o.value === value) ?? null

  // Runs synchronously after the menu mounts in the DOM but before the browser paints,
  // so a flip never shows as a visible jump - only the final, correct position ever
  // reaches the screen.
  useLayoutEffect(() => {
    if (!open) return
    const trigger = rootRef.current
    const menu = menuRef.current
    if (!trigger || !menu) return
    const triggerRect = trigger.getBoundingClientRect()
    const menuHeight = menu.getBoundingClientRect().height
    const spaceBelow = window.innerHeight - triggerRect.bottom
    const spaceAbove = triggerRect.top
    setOpenUpward(spaceBelow < menuHeight && spaceAbove > spaceBelow)
  }, [open])

  useEffect(() => {
    if (!open) return
    function onPointerDown(e: PointerEvent) {
      if (!rootRef.current?.contains(e.target as Node)) setOpen(false)
    }
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === 'Escape') setOpen(false)
    }
    document.addEventListener('pointerdown', onPointerDown)
    document.addEventListener('keydown', onKeyDown)
    return () => {
      document.removeEventListener('pointerdown', onPointerDown)
      document.removeEventListener('keydown', onKeyDown)
    }
  }, [open])

  return (
    <div ref={rootRef} className={`relative ${className}`}>
      <button
        type="button"
        disabled={disabled}
        aria-label={ariaLabel}
        data-testid={testId}
        aria-haspopup="listbox"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
        className="flex h-row w-full min-w-0 items-center justify-between gap-2 rounded-sm border-hair border-border-strong bg-surface px-s3 text-left text-bodyux text-fg transition-colors duration-100 hover:bg-fg/[0.04] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent disabled:pointer-events-none disabled:opacity-40"
      >
        <span className="min-w-0 flex-1 truncate">{selected?.label ?? placeholder}</span>
        <IconChevronDown size={14} className="shrink-0 text-muted" />
      </button>

      {open && (
        <div
          ref={menuRef}
          role="listbox"
          className={`absolute left-0 z-30 max-h-80 w-full min-w-max overflow-y-auto rounded-md border-hair border-border bg-surface-raised p-1 shadow-lg ${
            openUpward ? 'bottom-[calc(100%+4px)]' : 'top-[calc(100%+4px)]'
          }`}
        >
          {options.map((option) => {
            const isSelected = option.value === value
            return (
              <button
                key={option.value}
                type="button"
                role="option"
                aria-selected={isSelected}
                aria-disabled={option.disabled}
                disabled={option.disabled}
                onClick={() => {
                  onChange(option.value)
                  setOpen(false)
                }}
                className={`flex w-full items-center gap-2 rounded-[3px] px-2 text-left transition-colors duration-100 hover:bg-fg/6 disabled:pointer-events-none disabled:opacity-40 ${
                  option.sublabel ? 'py-1' : 'h-[30px]'
                } ${isSelected ? 'bg-accent-dim' : ''}`}
              >
                <span className="flex h-3.5 w-3.5 shrink-0 items-center justify-center">
                  {isSelected && <IconCheck size={14} className="text-accent" />}
                </span>
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-sm text-fg">{option.label}</span>
                  {option.sublabel && (
                    <span className="block truncate font-mono text-[11px] text-muted">{option.sublabel}</span>
                  )}
                </span>
              </button>
            )
          })}
        </div>
      )}
    </div>
  )
}
