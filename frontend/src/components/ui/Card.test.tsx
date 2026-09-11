import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import Card, { cardClassName } from './Card'

describe('Card', () => {
  it('renders its children', () => {
    render(<Card>workspace card</Card>)
    expect(screen.getByText('workspace card')).toBeInTheDocument()
  })

  it('adds hover classes only when hoverable', () => {
    expect(cardClassName(true)).toContain('hover:border-fg/30')
    expect(cardClassName(false)).not.toContain('hover:border-fg/30')
  })
})
