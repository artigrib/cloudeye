import { Link } from 'react-router-dom'
import LandingComparison from '../components/LandingComparison'
import { buttonClassName } from '../components/ui/Button'

/** CloudEye's marketing page (spec doc 1, section 7) - a separate visual register from
 * the app: effects are allowed here, the app stays quiet (spec 2's "смелость тратится в
 * одном месте"). Rendered outside <Layout> - no sidebar, responsive to 375px. */
export default function LandingPage() {
  return (
    <div className="min-h-screen bg-bg text-subtle">
      <header className="mx-auto flex max-w-6xl items-center justify-between px-6 py-5">
        <span className="flex items-center gap-2 text-[15px] font-semibold tracking-tight text-fg">
          <span className="inline-block h-1.5 w-1.5 rounded-full bg-accent" />
          cloudeye
        </span>
        <nav className="hidden items-center gap-6 text-sm text-muted sm:flex">
          <a href="#how-it-works" className="hover:text-subtle">
            Product
          </a>
          <a href="#compare" className="hover:text-subtle">
            Examples
          </a>
          <a href="#export" className="hover:text-subtle">
            Docs
          </a>
        </nav>
        <Link to="/workspaces" className={buttonClassName('secondary')}>
          Try it
        </Link>
      </header>

      <section className="relative overflow-hidden bg-canvas">
        <div className="mx-auto flex max-w-6xl flex-col items-center gap-8 px-6 py-20 text-center">
          <h1 className="max-w-2xl text-display font-medium leading-[1.02] tracking-[-0.02em] text-fg">
            Know if the robot fits, before you buy it
          </h1>
          <p className="max-w-xl text-body text-subtle">
            Film a room walkthrough on your phone. Get a point cloud, a traversability
            map, and a fit check against 8 robot platforms.
          </p>
          <div className="flex flex-wrap items-center justify-center gap-3">
            <Link to="/workspaces" className={buttonClassName('primary', 'large')}>
              Try a room
            </Link>
            <Link to="/workspaces" className={buttonClassName('secondary', 'large')}>
              Upload video
            </Link>
          </div>
        </div>
        {/* Hero clip: a real captured scene (02_modular_home), recorded from the live 3D
            viewer - replaces the earlier synthetic WebGL point cloud (spec doc 1,
            section 7.1) with actual product output. autoplay+muted+loop+playsInline so
            it behaves like a background loop with no controls.
            NOTE: this capture is a static loaded shot, not a full orbit - the intended
            camera-drag orbit (see scripts/record_hero.mjs equivalent in
            var/scratch/autopilot-20260905/pw/) couldn't be captured reliably this
            session (heavy concurrent CPU load on the box made the drag's synthetic
            pointer events register far too slowly/unpredictably vs. wall-clock
            recording time - see PR-feat-landing-assets.md). Re-record on a quieter box
            with the same script to get the orbiting version; swap the file in place,
            same path. */}
        <video
          className="mx-auto block h-[420px] w-full max-w-5xl object-contain px-6 pb-16"
          src="/landing/hero-orbit.webm"
          autoPlay
          muted
          loop
          playsInline
        />
      </section>

      <section id="compare" className="mx-auto max-w-6xl px-6 py-20">
        <h2 className="text-center text-h1 font-medium text-fg">One house. Different robots.</h2>
        <p className="mx-auto mt-2 max-w-xl text-center text-sm text-muted">
          This is a real captured scene - a modular home walkthrough, reconstructed by
          CloudEye - and the reachability numbers below are the app's own output for it,
          not hand-picked.
        </p>
        <div className="mt-10">
          <LandingComparison />
        </div>
      </section>

      <section id="how-it-works" className="mx-auto max-w-6xl px-6 py-20">
        <h2 className="text-center text-h1 font-medium text-fg">How it works</h2>
        <div className="mt-10 grid grid-cols-1 gap-8 sm:grid-cols-3">
          {[
            { n: '1', title: 'Walk the room', text: 'Film a walkthrough on your phone or an FPV drone.' },
            { n: '2', title: 'Reconstruct', text: 'Get a point cloud, a traversability map, and detected objects.' },
            { n: '3', title: 'Check the fit', text: "Pick a robot and see what it can and can't reach." },
          ].map((step) => (
            <div key={step.n} className="flex flex-col gap-2">
              <span className="font-mono text-xs text-faint">{step.n}</span>
              <h3 className="text-h3 font-medium text-fg">{step.title}</h3>
              <p className="text-sm text-muted">{step.text}</p>
            </div>
          ))}
        </div>
      </section>

      <section id="export" className="bg-canvas px-6 py-20">
        <div className="mx-auto grid max-w-6xl grid-cols-1 gap-10 md:grid-cols-2 md:items-center">
          <div>
            <h2 className="text-h1 font-medium text-fg">Drop it straight into Isaac Sim</h2>
            <p className="mt-3 max-w-md text-sm text-subtle">
              Export the reconstructed room as OpenUSD, metric scale - walls and floor as
              collision boxes, objects as convex hulls. Static collision only, ready to
              place a robot into.
            </p>
          </div>
          <div className="overflow-hidden rounded-md border-hair border-border bg-surface">
            {/* Real capture, not a mockup: Burger + Husky path-following the exported
                USD collision mesh of 02_modular_home inside Isaac Sim (isaac_demo.mp4,
                14s / 840 frames, converted to gif for inline README/marketing use). */}
            <img
              src="/landing/isaac-demo.gif"
              alt="TurtleBot3 Burger and Husky A200 path-following the exported scene inside NVIDIA Isaac Sim"
              className="block w-full"
            />
            <div className="flex flex-col gap-0.5 px-4 py-3 font-mono text-[11px] text-muted">
              <span>OpenUSD · metric scale</span>
              <span>walls as boxes</span>
              <span>objects as convex hulls</span>
            </div>
            <div className="border-t-hair border-border px-4 py-3">
              <Link to="/workspaces" className={buttonClassName('secondary')}>
                Download for Isaac Sim
              </Link>
            </div>
          </div>
        </div>
      </section>

      <footer className="mx-auto max-w-6xl px-6 py-12">
        <div className="grid grid-cols-2 gap-8 sm:grid-cols-4">
          <div className="col-span-2 sm:col-span-1">
            <span className="flex items-center gap-2 text-sm font-semibold text-fg">
              <span className="inline-block h-1.5 w-1.5 rounded-full bg-accent" />
              cloudeye
            </span>
            <p className="mt-2 text-xs text-muted">Reachability checks for embodied AI.</p>
          </div>
          <FooterColumn title="Product" links={['Overview', 'Robots', 'Export']} />
          <FooterColumn title="Resources" links={['How it works', 'Comparison']} />
          <FooterColumn title="Company" links={['About']} />
        </div>
        <p className="mt-10 text-xs text-faint">© {new Date().getFullYear()} CloudEye.</p>
      </footer>
    </div>
  )
}

function FooterColumn({ title, links }: { title: string; links: string[] }) {
  return (
    <div>
      <h4 className="font-mono text-[11px] uppercase tracking-[0.06em] text-muted">{title}</h4>
      <ul className="mt-2 flex flex-col gap-1.5">
        {links.map((l) => (
          <li key={l} className="text-sm text-subtle">
            {l}
          </li>
        ))}
      </ul>
    </div>
  )
}
