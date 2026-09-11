import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import Button from './Button'

describe('Button', () => {
  it('renders its label and fires onClick', () => {
    const onClick = vi.fn()
    render(<Button onClick={onClick}>Send</Button>)
    fireEvent.click(screen.getByRole('button', { name: 'Send' }))
    expect(onClick).toHaveBeenCalledOnce()
  })

  it('does not fire onClick when disabled', () => {
    const onClick = vi.fn()
    render(
      <Button onClick={onClick} disabled>
        Send
      </Button>,
    )
    fireEvent.click(screen.getByRole('button', { name: 'Send' }))
    expect(onClick).not.toHaveBeenCalled()
  })
})
