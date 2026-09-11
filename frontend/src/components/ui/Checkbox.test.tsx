import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import Checkbox from './Checkbox'

describe('Checkbox', () => {
  it('reflects the checked prop and toggles via the label', () => {
    const onChange = vi.fn()
    render(<Checkbox checked={false} onChange={onChange} label="show fragments" />)
    const box = screen.getByRole('checkbox', { name: 'show fragments' })
    expect(box).not.toBeChecked()
    fireEvent.click(screen.getByText('show fragments'))
    expect(onChange).toHaveBeenCalledWith(true)
  })

  it('does not fire onChange when disabled', () => {
    const onChange = vi.fn()
    render(<Checkbox checked={false} onChange={onChange} label="robot model" disabled />)
    fireEvent.click(screen.getByRole('checkbox', { name: 'robot model' }))
    expect(onChange).not.toHaveBeenCalled()
  })
})
