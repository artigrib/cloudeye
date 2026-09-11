import { NavLink } from 'react-router-dom'
import { useProjects } from '../state/projects-context'
import WorkspaceSwitcher from './WorkspaceSwitcher'

export default function Sidebar() {
  const { projects, loading, error, showArchived, setShowArchived } = useProjects()
  return (
    <aside className="flex w-[var(--sidebar-w)] shrink-0 flex-col border-r-hair border-border bg-surface">
      <div className="flex items-center justify-between border-b-hair border-border px-4 py-4">
        <NavLink to="/workspaces" className="flex items-center gap-2 text-[15px] font-semibold tracking-tight text-fg">
          {/* Backend status indicator (spec 6.1): --accent when reachable, --data-unreachable
             on error. No health probe exists yet, so this is fixed at "ok" for now. */}
          <span className="inline-block h-1.5 w-1.5 shrink-0 rounded-full bg-accent" />
          CloudEye
        </NavLink>
      </div>

      {/* The workspace dropdown, whose first entry ("+ New workspace") is now the only
          way into the upload flow. Mounted here as well as in the scene screen's own left
          column so that single entry point exists on every route. */}
      <WorkspaceSwitcher />

      <div className="flex-1 overflow-y-auto px-2 pb-3">
        <div className="flex items-center justify-between px-2 pb-1 pt-2">
          <p className="font-mono text-[11px] uppercase tracking-[0.06em] text-muted">Workspaces</p>
          <label className="flex items-center gap-1 text-[11px] text-faint">
            <input
              type="checkbox"
              checked={showArchived}
              onChange={(e) => setShowArchived(e.target.checked)}
              className="accent-accent"
            />
            show archived
          </label>
        </div>
        {loading && <p className="px-2 py-1 text-sm text-muted">Loading…</p>}
        {error && <p className="px-2 py-1 text-sm text-data-unreachable">{error}</p>}
        {!loading && !error && projects.length === 0 && (
          <p className="px-2 py-1 text-sm text-muted">No workspaces yet.</p>
        )}
        <nav className="flex flex-col gap-0.5">
          {projects.map((p) => (
            <NavLink
              key={p.id}
              to={`/workspaces/${p.id}`}
              className={({ isActive }) =>
                `flex items-center gap-1.5 truncate rounded-md px-2 py-1.5 text-sm transition-colors duration-100 ${
                  isActive ? 'bg-surface-raised text-fg' : 'text-muted hover:bg-fg/5 hover:text-subtle'
                } ${p.archived ? 'opacity-50' : ''}`
              }
              title={p.name}
            >
              <span className="truncate">{p.name}</span>
              {p.archived && (
                <span className="shrink-0 rounded border-hair border-border bg-surface-raised px-1 py-0.5 text-[10px] leading-none text-faint">
                  archived
                </span>
              )}
            </NavLink>
          ))}
        </nav>
      </div>
    </aside>
  )
}
