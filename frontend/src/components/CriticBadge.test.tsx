import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { NotFoundError } from '../api/client'
import { getSceneCritic } from '../api/scenes'
import type { CriticReportResponse, CriticRuleFinding, CriticVlmFinding } from '../api/types'
import CriticBadge from './CriticBadge'

vi.mock('../api/scenes', () => ({
  getSceneCritic: vi.fn(),
}))

const mockGetSceneCritic = vi.mocked(getSceneCritic)

function ruleFinding(overrides: Partial<CriticRuleFinding> = {}): CriticRuleFinding {
  return {
    check: 'yaw_vs_wall',
    source: 'rule',
    unverified: false,
    object_id: 'bed_0',
    severity: 'medium',
    measured: 25.0,
    threshold: 15.0,
    detail: 'bed yaw 154.97 deg vs wall 0.00 deg',
    ...overrides,
  }
}

function vlmFinding(overrides: Partial<CriticVlmFinding> = {}): CriticVlmFinding {
  return {
    object_id_or_region: 'lamp_0',
    issue: 'lamp missing',
    severity: 'high',
    view: 'view_000',
    source: 'vlm',
    unverified: true,
    ...overrides,
  }
}

function report(overrides: Partial<CriticReportResponse> = {}): CriticReportResponse {
  return {
    scene: 'test',
    rules: { findings: [], n_findings: 0, notes: [] },
    vlm: { findings: [], n_findings: 0, model: 'google/gemma-4-31b-it' },
    vlm_advisory: { gates: false, note: 'advisory only', corroboration_sentence: '0 VLM findings, 0 corroborated' },
    summary: { n_rule_findings: 0, n_vlm_findings: 0, n_overlap: 0 },
    ...overrides,
  }
}

describe('CriticBadge', () => {
  afterEach(() => {
    vi.clearAllMocks()
  })

  it('renders nothing while loading', () => {
    mockGetSceneCritic.mockReturnValue(new Promise(() => {})) // never resolves
    const { container } = render(<CriticBadge sceneId="scene-1" />)
    expect(container).toBeEmptyDOMElement()
  })

  it('renders nothing when the scene has no critic report (404)', async () => {
    mockGetSceneCritic.mockRejectedValue(new NotFoundError(404, null))
    const { container } = render(<CriticBadge sceneId="scene-1" />)
    await waitFor(() => expect(mockGetSceneCritic).toHaveBeenCalledWith('scene-1'))
    expect(container).toBeEmptyDOMElement()
  })

  it('renders nothing when the report exists but has zero findings', async () => {
    mockGetSceneCritic.mockResolvedValue(report())
    const { container } = render(<CriticBadge sceneId="scene-1" />)
    await waitFor(() => expect(mockGetSceneCritic).toHaveBeenCalled())
    expect(container).toBeEmptyDOMElement()
  })

  it('renders nothing when there are ONLY VLM findings and zero rule findings - VLM never gates on its own', async () => {
    mockGetSceneCritic.mockResolvedValue(
      report({
        vlm: { findings: [vlmFinding(), vlmFinding({ object_id_or_region: 'chair_0' })], n_findings: 2, model: 'google/gemma-4-31b-it' },
      }),
    )
    const { container } = render(<CriticBadge sceneId="scene-1" />)
    await waitFor(() => expect(mockGetSceneCritic).toHaveBeenCalled())
    expect(container).toBeEmptyDOMElement()
  })

  it('the badge count comes from rule findings only, even when VLM findings are also present', async () => {
    mockGetSceneCritic.mockResolvedValue(
      report({
        rules: { findings: [ruleFinding()], n_findings: 1, notes: [] },
        vlm: { findings: [vlmFinding(), vlmFinding({ object_id_or_region: 'chair_0' })], n_findings: 2, model: 'google/gemma-4-31b-it' },
      }),
    )

    render(<CriticBadge sceneId="scene-1" />)

    // 1 rule finding + 2 VLM findings - the badge must say "1", never "3".
    expect(await screen.findByTestId('critic-badge')).toHaveTextContent('1 critic finding')
  })

  it('uses singular wording for exactly one rule finding', async () => {
    mockGetSceneCritic.mockResolvedValue(report({ rules: { findings: [ruleFinding()], n_findings: 1, notes: [] } }))

    render(<CriticBadge sceneId="scene-1" />)

    expect(await screen.findByTestId('critic-badge')).toHaveTextContent('1 critic finding')
    expect(screen.queryByText(/1 critic findings/)).not.toBeInTheDocument()
  })

  it('uses plural wording for more than one rule finding', async () => {
    mockGetSceneCritic.mockResolvedValue(
      report({ rules: { findings: [ruleFinding(), ruleFinding({ object_id: 'chair_4' })], n_findings: 2, notes: [] } }),
    )

    render(<CriticBadge sceneId="scene-1" />)

    expect(await screen.findByTestId('critic-badge')).toHaveTextContent('2 critic findings')
  })

  it('clicking the badge toggles a list of rule findings (object id, issue, severity), and clicking again hides it', async () => {
    mockGetSceneCritic.mockResolvedValue(report({ rules: { findings: [ruleFinding()], n_findings: 1, notes: [] } }))

    render(<CriticBadge sceneId="scene-1" />)
    const badge = await screen.findByTestId('critic-badge')

    expect(screen.queryByTestId('critic-findings-list')).not.toBeInTheDocument()

    fireEvent.click(badge)
    expect(screen.getByTestId('critic-findings-list')).toBeInTheDocument()
    expect(screen.getByText('bed_0')).toBeInTheDocument()
    expect(screen.getByText('bed yaw 154.97 deg vs wall 0.00 deg')).toBeInTheDocument()
    expect(screen.getByText('medium')).toBeInTheDocument()

    fireEvent.click(badge)
    expect(screen.queryByTestId('critic-findings-list')).not.toBeInTheDocument()
  })

  it('shows VLM findings in the expanded list, each explicitly labelled unverified', async () => {
    mockGetSceneCritic.mockResolvedValue(
      report({
        rules: { findings: [ruleFinding()], n_findings: 1, notes: [] },
        vlm: { findings: [vlmFinding({ issue: 'bed looks distorted' })], n_findings: 1, model: 'google/gemma-4-31b-it' },
      }),
    )

    render(<CriticBadge sceneId="scene-1" />)
    fireEvent.click(await screen.findByTestId('critic-badge'))

    const vlmRow = screen.getByTestId('critic-vlm-finding')
    expect(vlmRow).toHaveTextContent('unverified (vlm)')
    expect(vlmRow).toHaveTextContent('lamp_0')
    expect(vlmRow).toHaveTextContent('bed looks distorted')
  })

  it('does not label rule findings as unverified', async () => {
    mockGetSceneCritic.mockResolvedValue(
      report({
        rules: { findings: [ruleFinding()], n_findings: 1, notes: [] },
        vlm: { findings: [vlmFinding()], n_findings: 1, model: 'google/gemma-4-31b-it' },
      }),
    )

    render(<CriticBadge sceneId="scene-1" />)
    fireEvent.click(await screen.findByTestId('critic-badge'))

    expect(screen.queryAllByText('unverified (vlm)')).toHaveLength(1) // only the VLM row, not the rule row
  })

  it('re-fetches and resets when sceneId changes', async () => {
    mockGetSceneCritic.mockResolvedValue(report({ rules: { findings: [ruleFinding()], n_findings: 1, notes: [] } }))
    const { rerender } = render(<CriticBadge sceneId="scene-1" />)
    await waitFor(() => expect(mockGetSceneCritic).toHaveBeenCalledWith('scene-1'))

    rerender(<CriticBadge sceneId="scene-2" />)
    await waitFor(() => expect(mockGetSceneCritic).toHaveBeenCalledWith('scene-2'))
    expect(mockGetSceneCritic).toHaveBeenCalledTimes(2)
  })
})
