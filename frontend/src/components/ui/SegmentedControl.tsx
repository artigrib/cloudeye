interface Segment<T extends string> {
  value: T
  label: string
  title?: string
}

interface Props<T extends string> {
  value: T
  onChange: (value: T) => void
  segments: Segment<T>[]
  'aria-label'?: string
  className?: string
  /** Skips the container's own border/background/padding - for embedding inside a
   * larger overlay panel that already supplies that chrome (e.g. the 3D view cluster,
   * spec 6.4), so the two don't nest into a double border. */
  bare?: boolean
}

/** Replaces exclusive-choice button pairs/triples (2D|3D, top|front|iso, ...) with one
 * shared shape. The active segment is a neutral fill, never the accent - spec 5.2: the
 * accent color is spent on actions only, not on state. */
export default function SegmentedControl<T extends string>({
  value,
  onChange,
  segments,
  'aria-label': ariaLabel,
  className = '',
  bare = false,
}: Props<T>) {
  return (
    <div
      role="radiogroup"
      aria-label={ariaLabel}
      className={`inline-flex items-center gap-0.5 ${bare ? '' : 'rounded-sm border-hair border-border bg-surface p-0.5'} ${className}`}
    >
      {segments.map((segment) => {
        const active = segment.value === value
        return (
          <button
            key={segment.value}
            type="button"
            role="radio"
            aria-checked={active}
            title={segment.title ?? segment.label}
            onClick={() => onChange(segment.value)}
            className={`h-row min-w-8 rounded-[2px] px-s3 text-bodyux font-medium transition-colors duration-100 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent ${
              active ? 'bg-surface-raised text-fg' : 'text-muted hover:text-subtle'
            }`}
          >
            {segment.label}
          </button>
        )
      })}
    </div>
  )
}
