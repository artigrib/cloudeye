import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import Slider from './Slider'

describe('Slider', () => {
  it('renders the formatted value', () => {
    render(
      <Slider
        value={0.25}
        onChange={() => {}}
        min={0.1}
        max={0.6}
        formatValue={(v) => `${v.toFixed(2)} m`}
        aria-label="Robot radius"
      />,
    )
    expect(screen.getByText('0.25 m')).toBeInTheDocument()
  })

  it('steps up/down and clamps to min/max on the keyboard', () => {
    const onChange = vi.fn()
    render(<Slider value={0.1} onChange={onChange} min={0.1} max={0.3} step={0.01} aria-label="Radius" />)
    const track = screen.getByRole('slider', { name: 'Radius' })

    fireEvent.keyDown(track, { key: 'ArrowRight' })
    expect(onChange.mock.lastCall![0]).toBeCloseTo(0.11, 6)

    fireEvent.keyDown(track, { key: 'Home' })
    expect(onChange).toHaveBeenLastCalledWith(0.1)

    fireEvent.keyDown(track, { key: 'End' })
    expect(onChange).toHaveBeenLastCalledWith(0.3)
  })

  it('ignores keyboard input while disabled', () => {
    const onChange = vi.fn()
    render(<Slider value={0.1} onChange={onChange} min={0.1} max={0.3} disabled aria-label="Radius" />)
    fireEvent.keyDown(screen.getByRole('slider', { name: 'Radius' }), { key: 'ArrowRight' })
    expect(onChange).not.toHaveBeenCalled()
  })
})
