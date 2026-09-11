import { Outlet, useMatch } from 'react-router-dom'
import Sidebar from './Sidebar'
import { CaptureGuideProvider } from '../state/capture-guide-context'

export default function Layout() {
  // The scene screen owns its whole width: its left column is scene-and-robot (workspace
  // switcher, room summary, ROBOT, EXPORT), which is a different thing from the global
  // workspace list this sidebar shows everywhere else, and two left columns side by side
  // would eat the canvas. Every other route keeps the sidebar exactly as it was.
  //
  // The pattern must track the route: it is matched structurally, so the PARAM name is
  // free, but a stale PATH here silently puts two left columns on the scene screen with
  // no error anywhere.
  const onScene = useMatch('/workspaces/:workspaceId/scenes/:sceneId') !== null

  return (
    <CaptureGuideProvider>
      <div className="flex h-screen min-w-[1280px] overflow-hidden bg-bg text-subtle">
        {!onScene && <Sidebar />}
        <main className="min-w-0 flex-1 overflow-y-auto">
          <Outlet />
        </main>
      </div>
    </CaptureGuideProvider>
  )
}
