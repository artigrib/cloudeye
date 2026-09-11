import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import Select from './Select'

const OPTIONS = [
  { value: 'burger', label: 'TurtleBot3 Burger', sublabel: '0.10 m · differential_drive' },
  { value: 'go2', label: 'Unitree Go2', sublabel: '0.25 m · legged' },
]

describe('Select', () => {
  it('shows the placeholder when nothing is selected', () => {
    render(<Select value={null} onChange={() => {}} options={OPTIONS} placeholder="Pick one" />)
    expect(screen.getByRole('button')).toHaveTextContent('Pick one')
  })

  it('shows the selected option label as the trigger text', () => {
    render(<Select value="go2" onChange={() => {}} options={OPTIONS} />)
    expect(screen.getByRole('button')).toHaveTextContent('Unitree Go2')
  })

  it('opens the menu and selects an option, closing after', () => {
    const onChange = vi.fn()
    render(<Select value="burger" onChange={onChange} options={OPTIONS} />)
    fireEvent.click(screen.getByRole('button'))
    expect(screen.getByRole('listbox')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('option', { name: /Unitree Go2/ }))
    expect(onChange).toHaveBeenCalledWith('go2')
    expect(screen.queryByRole('listbox')).not.toBeInTheDocument()
  })

  it('closes on outside pointerdown without calling onChange', () => {
    const onChange = vi.fn()
    render(<Select value="burger" onChange={onChange} options={OPTIONS} />)
    fireEvent.click(screen.getByRole('button'))
    fireEvent.pointerDown(document.body)
    expect(screen.queryByRole('listbox')).not.toBeInTheDocument()
    expect(onChange).not.toHaveBeenCalled()
  })
})
