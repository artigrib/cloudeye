import { useCallback, useRef } from 'react'

interface Props {
  value: number
  onChange: (value: number) => void
  min: number
  max: number
  step?: number
  /** Left-of-track caption, spec 5.5: sans 12px muted, min-width 84px. Omit for a
   * bare slider (e.g. inside a denser control row that labels it externally). */
  label?: string
  /** Right-of-track readout. Defaults to the raw value, 2 decimals. Pass `null` to omit
   * it entirely (e.g. the video scrubber, which shows its own combined time text). */
  formatValue?: ((value: number) => string) | null
  disabled?: boolean
  'aria-label'?: string
  /** Fill color - spec 5.5 says --border-strong (a slider is a parameter, not an
   * action), but the video scrubber (spec 5.10) is the one exception: --fg. */
  fillClassName?: string
}

function clamp(v: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, v))
}

function roundToStep(v: number, min: number, step: number): number {
  return Math.round((v - min) / step) * step + min
}

/** Custom track+thumb, not a styled native <input type="range"> - spec 5.5. The fill is
 * --border-strong, never the accent: a slider is a parameter, not an action. */
export default function Slider({
  value,
  onChange,
  min,
  max,
  step = 0.01,
  label,
  formatValue,
  disabled,
  'aria-label': ariaLabel,
  fillClassName = 'bg-border-strong',
}: Props) {
  const trackRef = useRef<HTMLDivElement>(null)
  const pct = ((clamp(value, min, max) - min) / (max - min)) * 100

  const valueFromClientX = useCallback(
    (clientX: number) => {
      const rect = trackRef.current!.getBoundingClientRect()
      const ratio = clamp((clientX - rect.left) / rect.width, 0, 1)
      return clamp(roundToStep(min + ratio * (max - min), min, step), min, max)
    },
    [min, max, step],
  )

  function onPointerDown(e: React.PointerEvent) {
    if (disabled) return
    ;(e.target as Element).setPointerCapture(e.pointerId)
    onChange(valueFromClientX(e.clientX))
  }

  function onPointerMove(e: React.PointerEvent) {
    if (disabled || e.buttons === 0) return
    onChange(valueFromClientX(e.clientX))
  }

  function onKeyDown(e: React.KeyboardEvent) {
    if (disabled) return
    const big = step * 10
    const step_ = (delta: number) => onChange(clamp(roundToStep(value + delta, min, step), min, max))
    if (e.key === 'ArrowRight' || e.key === 'ArrowUp') step_(step)
    else if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') step_(-step)
    else if (e.key === 'PageUp') step_(big)
    else if (e.key === 'PageDown') step_(-big)
    else if (e.key === 'Home') onChange(min)
    else if (e.key === 'End') onChange(max)
    else return
    e.preventDefault()
  }

  return (
    <div className={`flex items-center gap-2 ${disabled ? 'opacity-40' : ''}`}>
      {label && <span className="min-w-[84px] shrink-0 text-xs text-muted">{label}</span>}
      <div
        ref={trackRef}
        role="slider"
        tabIndex={disabled ? -1 : 0}
        aria-label={ariaLabel ?? label}
        aria-valuemin={min}
        aria-valuemax={max}
        aria-valuenow={value}
        aria-disabled={disabled}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onKeyDown={onKeyDown}
        className="group relative h-3 min-w-[64px] flex-1 cursor-pointer touch-none"
      >
        <div className="absolute left-0 top-1/2 h-[3px] w-full -translate-y-1/2 rounded-full bg-border" />
        <div
          className={`absolute left-0 top-1/2 h-[3px] -translate-y-1/2 rounded-full ${fillClassName}`}
          style={{ width: `${pct}%` }}
        />
        <div
          className="absolute top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full bg-fg transition-[width,height] duration-100 group-hover:h-3.5 group-hover:w-3.5"
          style={{ left: `${pct}%` }}
        />
      </div>
      {formatValue !== null && (
        <span className="min-w-[56px] shrink-0 text-right font-mono text-xs tabular-nums text-fg">
          {formatValue ? formatValue(value) : value.toFixed(2)}
        </span>
      )}
    </div>
  )
}
