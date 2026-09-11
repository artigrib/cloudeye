import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import App from './App.tsx'
import { ProjectsProvider } from './state/projects-context.tsx'
import './index.css'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter>
      <ProjectsProvider>
        <App />
      </ProjectsProvider>
    </BrowserRouter>
  </StrictMode>,
)
