import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import PanelSection from './PanelSection'

describe('PanelSection', () => {
  it('renders the title with a parenthesized count', () => {
    render(
      <PanelSection title="Objects" count={40}>
        <p>body</p>
      </PanelSection>,
    )
    expect(screen.getByText('Objects')).toBeInTheDocument()
    expect(screen.getByText('(40)')).toBeInTheDocument()
    expect(screen.getByText('body')).toBeInTheDocument()
  })

  it('omits the count when not given', () => {
    render(<PanelSection title="Commands">{null}</PanelSection>)
    expect(screen.queryByText(/\(/)).not.toBeInTheDocument()
  })
})
