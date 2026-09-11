import { Navigate, Route, Routes, useLocation } from 'react-router-dom'
import Layout from './components/Layout'
import HomePage from './pages/HomePage'
import LandingPage from './pages/LandingPage'
import MsaViewerPage from './pages/MsaViewerPage'
import NewWorkspacePage from './pages/NewWorkspacePage'
import ProcessingPage from './pages/ProcessingPage'
import WorkspacePage from './pages/WorkspacePage'
import ScenePage from './pages/ScenePage'
import UiShowcase from './pages/UiShowcase'

/** Everything under `/projects` moved to `/workspaces` - one entity, one name. Kept as a
 * redirect rather than deleted: the old shape is in browser history, in the demo/ probes
 * and in the recording drivers, and a rename that 404s them is a rename that broke the app.
 *
 * A path-only rewrite via one splat route, not three wrappers reading useParams: the
 * transform is the same string operation for every depth, so this also covers any
 * /projects/... path nobody remembered. `search`/`hash` are carried through (a scene URL
 * legitimately has both); `state` is deliberately not - it needs a separate prop and
 * nothing navigates to these paths with state.
 *
 * `replace` matters: without it the dead URL stays in the back stack and Back bounces
 * straight into the redirect again. */
function LegacyProjectsRedirect() {
  const { pathname, search, hash } = useLocation()
  return <Navigate replace to={{ pathname: `/workspaces${pathname.slice('/projects'.length)}`, search, hash }} />
}

export default function App() {
  return (
    <Routes>
      {/* Not linked from anywhere in the app chrome - a bare component/token catalog
          for design review (redesign plan section 1). */}
      <Route path="/_ui" element={<UiShowcase />} />
      {/* Not linked from anywhere in the app chrome - MSA (Measured Scene Assembly,
          var/scratch/msa/SPEC.md) scenes are file-based script output, not yet
          wired into the DB-backed pipeline this app otherwise routes through. */}
      <Route path="/_msa-viewer" element={<MsaViewerPage />} />
      {/* Marketing page (spec doc 1, section 7) - deliberately outside <Layout>: no
          sidebar, its own visual register. The app itself starts at /workspaces. */}
      <Route path="/" element={<LandingPage />} />
      <Route element={<Layout />}>
        <Route path="/workspaces" element={<HomePage />} />
        {/* The single entry point for getting a video in - see NewWorkspacePage. Static
            segments outrank dynamic ones in react-router's ranked matching, so this is
            never shadowed by "/workspaces/:workspaceId" below regardless of order here -
            demo/probe_workspace_routes.py proves it rather than trusting it. */}
        <Route path="/workspaces/new" element={<NewWorkspacePage />} />
        {/* All state comes from the server on every poll - see ProcessingPage. */}
        <Route path="/workspaces/:workspaceId/processing" element={<ProcessingPage />} />
        <Route path="/workspaces/:workspaceId" element={<WorkspacePage />} />
        <Route path="/workspaces/:workspaceId/scenes/:sceneId" element={<ScenePage />} />

      </Route>

      {/* Outside <Layout> on purpose: a redirect must not mount the Sidebar for a frame on
          a URL we are leaving. Both lines: a splat matches zero segments, but that is the
          kind of thing to state rather than believe, and the second costs nothing. */}
      <Route path="/projects" element={<LegacyProjectsRedirect />} />
      <Route path="/projects/*" element={<LegacyProjectsRedirect />} />

      {/* There was no catch-all before; an unknown path rendered a blank page. */}
      <Route path="*" element={<Navigate replace to="/workspaces" />} />
    </Routes>
  )
}
