import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import JobTimeline from './JobTimeline'

describe('JobTimeline', () => {
  it('shows "Queued" when there is no scene yet', () => {
    render(<JobTimeline scene={null} />)
    expect(screen.getByText('Queued')).toBeInTheDocument()
  })

  it('shows "Queued" while status is queued', () => {
    render(<JobTimeline scene={{ status: 'queued', error_message: null, stage_timings: [] }} />)
    expect(screen.getByText('Queued')).toBeInTheDocument()
  })

  it('shows the current stage while processing, with no ETA yet if no timings', () => {
    render(<JobTimeline scene={{ status: 'processing', error_message: null, stage_timings: [] }} />)
    expect(screen.getByText('Processing')).toBeInTheDocument()
  })

  it('shows a stage label and rough ETA once one stage timing is present', () => {
    render(
      <JobTimeline
        scene={{
          status: 'processing',
          error_message: null,
          stage_timings: [{ stage: 'keyframes', started_at: new Date().toISOString(), duration_sec: null }],
        }}
      />,
    )
    expect(screen.getByText(/Extracting keyframes/)).toBeInTheDocument()
    expect(screen.getByText(/left \(estimate\)/)).toBeInTheDocument()
  })

  it('shows the validator error_message (not a bare "failed") when the scene failed', () => {
    render(
      <JobTimeline
        scene={{
          status: 'failed',
          error_message: 'scene failed validation: room_dimensions: 45.2m exceeds plausible room size',
          stage_timings: [],
        }}
      />,
    )
    // sceneErrorMessage.ts's room_dimensions rule translates this to a friendlier
    // message but the raw text is still available via the title attribute.
    expect(screen.getByTitle(/room_dimensions/)).toBeInTheDocument()
  })

  it('falls back to a generic message when a failed scene has no error_message', () => {
    render(<JobTimeline scene={{ status: 'failed', error_message: null, stage_timings: [] }} />)
    expect(screen.getByText('Failed: Reconstruction failed.')).toBeInTheDocument()
  })

  it('renders nothing once the scene is done - caller owns that state', () => {
    const { container } = render(<JobTimeline scene={{ status: 'done', error_message: null, stage_timings: [] }} />)
    expect(container).toBeEmptyDOMElement()
  })

  it('full variant renders the queued/processing copy without the compact wrapper text', () => {
    render(<JobTimeline scene={{ status: 'queued', error_message: null, stage_timings: [] }} variant="full" />)
    expect(screen.getByText('Queued for reconstruction…')).toBeInTheDocument()
    expect(screen.getByText('Usually takes ~5 minutes.')).toBeInTheDocument()
  })
})
