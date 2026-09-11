import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import SegmentedControl from './SegmentedControl'

const SEGMENTS = [
  { value: '2d', label: '2D' },
  { value: '3d', label: '3D' },
] as const

describe('SegmentedControl', () => {
  it('marks the current value as checked', () => {
    render(<SegmentedControl value="3d" onChange={() => {}} segments={[...SEGMENTS]} />)
    expect(screen.getByRole('radio', { name: '2D' })).toHaveAttribute('aria-checked', 'false')
    expect(screen.getByRole('radio', { name: '3D' })).toHaveAttribute('aria-checked', 'true')
  })

  it('calls onChange with the clicked segment value', () => {
    const onChange = vi.fn()
    render(<SegmentedControl value="2d" onChange={onChange} segments={[...SEGMENTS]} />)
    fireEvent.click(screen.getByRole('radio', { name: '3D' }))
    expect(onChange).toHaveBeenCalledWith('3d')
  })
})
