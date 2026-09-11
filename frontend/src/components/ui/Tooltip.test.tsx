import { act, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import Tooltip from './Tooltip'

describe('Tooltip', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })
  afterEach(() => {
    vi.useRealTimers()
  })

  it('shows the tooltip after the hover delay, hides on mouse leave', () => {
    render(
      <Tooltip content="Reset view">
        <button type="button">Reset</button>
      </Tooltip>,
    )
    const trigger = screen.getByRole('button', { name: 'Reset' })
    expect(screen.queryByRole('tooltip')).not.toBeInTheDocument()

    fireEvent.mouseEnter(trigger)
    act(() => {
      vi.advanceTimersByTime(300)
    })
    expect(screen.getByRole('tooltip')).toHaveTextContent('Reset view')

    fireEvent.mouseLeave(trigger)
    expect(screen.queryByRole('tooltip')).not.toBeInTheDocument()
  })

  it('does not show while disabled', () => {
    render(
      <Tooltip content="Reset view" disabled>
        <button type="button">Reset</button>
      </Tooltip>,
    )
    fireEvent.mouseEnter(screen.getByRole('button', { name: 'Reset' }))
    act(() => {
      vi.advanceTimersByTime(300)
    })
    expect(screen.queryByRole('tooltip')).not.toBeInTheDocument()
  })
})
