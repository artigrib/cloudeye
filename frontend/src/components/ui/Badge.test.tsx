import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import Badge from './Badge'

describe('Badge', () => {
  it('renders its children', () => {
    render(<Badge variant="unreachable">unreachable</Badge>)
    expect(screen.getByText('unreachable')).toBeInTheDocument()
  })

  it('defaults to the neutral variant', () => {
    render(<Badge>processing</Badge>)
    expect(screen.getByText('processing').className).toContain('text-muted')
  })
})
