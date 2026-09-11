import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from 'react'
import { listProjects } from '../api/projects'
import type { ProjectResponse } from '../api/types'

interface ProjectsContextValue {
  projects: ProjectResponse[]
  loading: boolean
  error: string | null
  refresh: () => Promise<void>
  /** Whether archived projects are included in `projects` - off by default, so an
   * archived project (see Project.archived) stays out of the sidebar/home gallery
   * unless a viewer explicitly asks to see it. Shared across the app rather than
   * per-page state, so the sidebar's toggle and anything else reading `projects`
   * always agree on what's currently visible. */
  showArchived: boolean
  setShowArchived: (value: boolean) => void
}

const ProjectsContext = createContext<ProjectsContextValue | null>(null)

export function ProjectsProvider({ children }: { children: ReactNode }) {
  const [projects, setProjects] = useState<ProjectResponse[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [showArchived, setShowArchived] = useState(false)

  const refresh = useCallback(async () => {
    try {
      const res = await listProjects(0, 200, showArchived)
      setProjects(res.items)
      setError(null)
    } catch {
      setError('Could not reach the CloudEye backend.')
    } finally {
      setLoading(false)
    }
  }, [showArchived])

  useEffect(() => {
    void refresh()
  }, [refresh])

  return (
    <ProjectsContext.Provider value={{ projects, loading, error, refresh, showArchived, setShowArchived }}>
      {children}
    </ProjectsContext.Provider>
  )
}

export function useProjects(): ProjectsContextValue {
  const ctx = useContext(ProjectsContext)
  if (!ctx) throw new Error('useProjects must be used within a ProjectsProvider')
  return ctx
}
