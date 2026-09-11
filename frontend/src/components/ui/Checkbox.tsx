import { useId } from 'react'
import { IconCheck } from '@tabler/icons-react'

interface Props {
  checked: boolean
  onChange: (checked: boolean) => void
  label: string
  disabled?: boolean
  className?: string
  /** Forwarded to the <input>, so a checkbox with a non-unique label can still be
   * addressed by a probe. TSX accepts unknown `data-*` props on a component without
   * complaint, so this has to be declared and forwarded explicitly or it is silently
   * dropped - which is exactly what happened to "Include point cloud". */
  'data-testid'?: string
}

/** Replaces every native <input type="checkbox"> - spec 5.4. The whole row is
 * clickable, not just the 16px box. */
export default function Checkbox({
  checked,
  onChange,
  label,
  disabled,
  className = '',
  'data-testid': testId,
}: Props) {
  const id = useId()
  return (
    <label
      htmlFor={id}
      className={`flex h-row items-center gap-s2 text-bodyux text-subtle ${disabled ? 'opacity-40' : 'cursor-pointer'} ${className}`}
    >
      <span className="relative flex h-4 w-4 shrink-0 items-center justify-center">
        <input
          id={id}
          type="checkbox"
          data-testid={testId}
          checked={checked}
          disabled={disabled}
          onChange={(e) => {
            if (!disabled) onChange(e.target.checked)
          }}
          className="peer absolute inset-0 m-0 h-4 w-4 cursor-pointer appearance-none rounded-[3px] border-hair border-border-strong bg-transparent transition-colors duration-100 checked:border-accent checked:bg-accent focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent disabled:cursor-default"
        />
        <IconCheck
          size={10}
          strokeWidth={3}
          className="pointer-events-none absolute text-accent-fg opacity-0 peer-checked:opacity-100"
        />
      </span>
      {label}
    </label>
  )
}
