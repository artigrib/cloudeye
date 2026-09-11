import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import Input from './Input'

describe('Input', () => {
  it('forwards value and onChange like a plain <input>', () => {
    const onChange = vi.fn()
    render(<Input value="Living Room" onChange={onChange} placeholder="Room name" />)
    const input = screen.getByPlaceholderText('Room name')
    expect(input).toHaveValue('Living Room')
    fireEvent.change(input, { target: { value: 'Kitchen' } })
    expect(onChange).toHaveBeenCalled()
  })
})
