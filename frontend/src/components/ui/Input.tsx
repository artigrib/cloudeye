import { forwardRef } from 'react'
import type { InputHTMLAttributes } from 'react'

const Input = forwardRef<HTMLInputElement, InputHTMLAttributes<HTMLInputElement>>(function Input(
  { className = '', ...rest },
  ref,
) {
  return (
    <input
      ref={ref}
      className={`h-row min-w-0 rounded-sm border-hair border-border-strong bg-surface px-s3 text-bodyux text-fg outline-none transition-colors duration-100 placeholder:text-faint focus:border-accent ${className}`}
      {...rest}
    />
  )
})

export default Input
