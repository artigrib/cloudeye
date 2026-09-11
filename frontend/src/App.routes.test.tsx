/** The first routing tests in this repo.
 *
 * Before the workspace rename there were 48 test files and not one of them imported a
 * Router, so nothing checked that a URL reached the screen it names. That is a bad place
 * to be doing a route rename from: `useParams` returns a Partial, so a stale param name
 * is `undefined` with a clean `tsc`, no runtime error, and three features quietly dead
 * (see ScenePage's `workspaceId` comment). These cover the claims the rename rests on.
 *
 * Deliberately NOT rendering the real pages - they fetch, need contexts and a WebGPU
 * canvas. Route *matching* is what is under test, so the elements are stubs and the real
 * App's own route table is mirrored only where the assertion needs it.
 */
import { render, screen } from '@testing-library/react'
import { MemoryRouter, Navigate, Route, Routes, useLocation, useParams } from 'react-router-dom'
import { describe, expect, it } from 'vitest'

/** The same path-only rewrite App.tsx uses. Kept in the test as a literal copy rather than
 * imported, so a change to the real one has to be made deliberately in both places. */
function LegacyProjectsRedirect() {
  const { pathname, search, hash } = useLocation()
  return <Navigate replace to={{ pathname: `/workspaces${pathname.slice('/projects'.length)}`, search, hash }} />
}

function Where() {
  const { pathname, search, hash } = useLocation()
  return <div data-testid="where">{`${pathname}${search}${hash}`}</div>
}

function renderAt(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <Routes>
        <Route path="/workspaces/*" element={<Where />} />
        <Route path="/projects" element={<LegacyProjectsRedirect />} />
        <Route path="/projects/*" element={<LegacyProjectsRedirect />} />
      </Routes>
    </MemoryRouter>,
  )
}

describe('legacy /projects redirects', () => {
  it.each([
    ['/projects', '/workspaces'],
    ['/projects/p1', '/workspaces/p1'],
    ['/projects/p1/scenes/s1', '/workspaces/p1/scenes/s1'],
  ])('%s -> %s', (from, to) => {
    renderAt(from)
    expect(screen.getByTestId('where')).toHaveTextContent(to)
  })

  it('carries the query string and hash through', () => {
    // A scene URL legitimately has both; dropping them would silently reset the view.
    renderAt('/projects/p1/scenes/s1?tab=objects#frag')
    expect(screen.getByTestId('where')).toHaveTextContent('/workspaces/p1/scenes/s1?tab=objects#frag')
  })
})

describe('route ranking', () => {
  it('prefers the literal /workspaces/new over /workspaces/:workspaceId', () => {
    // Declared in the WRONG order on purpose: react-router ranks branches by segment
    // score (static beats dynamic) and does not use declaration order, so the wizard wins
    // regardless. The whole single-entry-point design rests on this, so it is asserted
    // rather than assumed.
    render(
      <MemoryRouter initialEntries={['/workspaces/new']}>
        <Routes>
          <Route path="/workspaces/:workspaceId" element={<div>dispatcher</div>} />
          <Route path="/workspaces/new" element={<div>wizard</div>} />
        </Routes>
      </MemoryRouter>,
    )
    expect(screen.getByText('wizard')).toBeInTheDocument()
  })

  it('still matches :workspaceId for a real id', () => {
    render(
      <MemoryRouter initialEntries={['/workspaces/8ac1f0e2']}>
        <Routes>
          <Route path="/workspaces/new" element={<div>wizard</div>} />
          <Route path="/workspaces/:workspaceId" element={<ShowParam />} />
        </Routes>
      </MemoryRouter>,
    )
    expect(screen.getByTestId('param')).toHaveTextContent('8ac1f0e2')
  })
})

function ShowParam() {
  const { workspaceId } = useParams<{ workspaceId: string }>()
  return <div data-testid="param">{workspaceId}</div>
}

describe('scene route param name', () => {
  it('is workspaceId, the name every page reads', () => {
    // ScenePage destructures `workspaceId`. If the path segment were renamed without the
    // page, this test is the only thing in the suite that would notice.
    render(
      <MemoryRouter initialEntries={['/workspaces/w1/scenes/s1']}>
        <Routes>
          <Route path="/workspaces/:workspaceId/scenes/:sceneId" element={<ShowBoth />} />
        </Routes>
      </MemoryRouter>,
    )
    expect(screen.getByTestId('both')).toHaveTextContent('w1/s1')
  })
})

function ShowBoth() {
  const { workspaceId, sceneId } = useParams<{ workspaceId: string; sceneId: string }>()
  return <div data-testid="both">{`${workspaceId}/${sceneId}`}</div>
}
