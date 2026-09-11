import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import SceneVerdict from './SceneVerdict'

const base = { platformName: 'Husky A200', reachableCount: 10, totalCount: 17, noStart: false }

describe('SceneVerdict', () => {
  it('is "N of M reachable" for the selected robot', () => {
    render(<SceneVerdict {...base} />)
    expect(screen.getByTestId('verdict-count')).toHaveTextContent('10 of 17')
    expect(screen.getByText(/Husky A200/)).toBeInTheDocument()
  })

  it('says the robot does not fit rather than showing 0 of 17', () => {
    render(<SceneVerdict {...base} reachableCount={0} noStart />)
    expect(screen.queryByTestId('verdict-count')).not.toBeInTheDocument()
    expect(screen.getByText(/does not fit in this room/)).toBeInTheDocument()
  })

  describe('tightest gap', () => {
    it('shows the measured p5 corridor width alongside reachable routes', () => {
      render(<SceneVerdict {...base} tightestGapM={0.6085} />)
      expect(screen.getByTestId('tightest-gap')).toHaveTextContent('tightest gap 0.61 m')
    })

    it('renders an em dash when the audit has no successful trial to measure', () => {
      render(<SceneVerdict {...base} tightestGapM={null} />)
      expect(screen.getByTestId('tightest-gap')).toHaveTextContent('tightest gap —')
    })

    it('is absent entirely when this scene has no audit', () => {
      render(<SceneVerdict {...base} />)
      expect(screen.queryByTestId('tightest-gap')).not.toBeInTheDocument()
    })

    it('is absent when nothing is reachable - it is a number about routes that exist', () => {
      render(<SceneVerdict {...base} reachableCount={0} tightestGapM={0.6085} />)
      expect(screen.queryByTestId('tightest-gap')).not.toBeInTheDocument()
    })

    it('is absent before the first reachability response', () => {
      render(<SceneVerdict {...base} reachableCount={null} tightestGapM={0.6085} />)
      expect(screen.getByTestId('verdict-count')).toHaveTextContent('…')
      expect(screen.queryByTestId('tightest-gap')).not.toBeInTheDocument()
    })
  })
})

// The verdict says the word rather than leaving it to be inferred from a count: 0 of 17
// and 13 of 17 are both numbers, and only one of them means "this robot cannot be here at
// all". Both branches carry `verdict-fit`, so a probe reads one testid either way.
describe('SceneVerdict fit verdict', () => {
  it('says "fits" alongside a real count, and how many it misses', () => {
    // 13 of 17 is "fits", with the four it cannot reach named rather than left to
    // subtraction - see "the fit badge and the count agree" below for why the bare word
    // stopped being enough.
    render(<SceneVerdict platformName="TurtleBot3 Burger" reachableCount={13} totalCount={17} noStart={false} />)
    expect(screen.getByTestId('verdict-fit').textContent).toBe('fits · 4 unreachable')
  })

  it('says "does not fit in this room" when the robot has nowhere to stand', () => {
    render(<SceneVerdict platformName="Husky A200" reachableCount={0} totalCount={17} noStart />)
    expect(screen.getByTestId('verdict-fit').textContent).toBe('does not fit in this room')
    // And it must NOT report an arithmetically-correct, thoroughly misleading 0 of 17.
    expect(screen.queryByTestId('verdict-count')).toBeNull()
  })
})

// ONE RULE. The badge used to say a flat "fits" beside any count at all - true about the
// start position, and read as true about the room. On own_0901_155452 that put "8 of 24
// reachable" next to a green "fits". The badge now carries the same fact the count does.
describe('the fit badge and the count agree', () => {
  it('says plain "fits" only when everything listed is reachable', () => {
    render(<SceneVerdict platformName="TurtleBot3 Burger" reachableCount={17} totalCount={17} noStart={false} />)
    expect(screen.getByTestId('verdict-fit')).toHaveTextContent('fits')
    expect(screen.getByTestId('verdict-fit')).not.toHaveTextContent('unreachable')
  })

  it('names how many it cannot reach when some are out', () => {
    render(<SceneVerdict platformName="TurtleBot3 Burger" reachableCount={8} totalCount={24} noStart={false} />)
    expect(screen.getByTestId('verdict-fit')).toHaveTextContent('fits · 16 unreachable')
  })

  it('says "does not fit" when the robot is wider than the tightest gap the audit found', () => {
    // Distinct from having nowhere to stand: this robot may have a start cell and still
    // be unable to make any of the journeys that were measured.
    render(
      <SceneVerdict
        platformName="Husky A200"
        reachableCount={3}
        totalCount={17}
        noStart={false}
        robotRadiusM={0.5528}
        tightestGapM={0.28}
      />,
    )
    expect(screen.getByTestId('verdict-fit')).toHaveTextContent('does not fit')
  })

  it('shows the unobserved distance in metres, from cells x the grid resolution', () => {
    render(
      <SceneVerdict
        platformName="TurtleBot3 Burger"
        reachableCount={13}
        totalCount={17}
        noStart={false}
        unknownCellsOnRoute={139}
        gridResolutionM={0.05}
      />,
    )
    expect(screen.getByTestId('unobserved-cells')).toHaveTextContent('via 7.0 m unobserved')
  })

  it('says so when the scene has no navigation layer, instead of a count of nothing', () => {
    render(
      <SceneVerdict
        platformName="TurtleBot3 Burger"
        reachableCount={0}
        totalCount={0}
        noStart
        layerAvailable={false}
      />,
    )
    expect(screen.getByTestId('verdict-fit')).toHaveTextContent('layer not available')
  })
})
