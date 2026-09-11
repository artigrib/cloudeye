import { useEffect, useState } from 'react'
import { loadLandingReachability, type LandingReachability } from '../lib/landingReachability'

/** Spec 7.2 - "not feature #2, this is proof": the visitor clicks between robot
 * platforms and watches the reachable count change against the same room, entirely off
 * the static reachability.json (no backend calls). */
export default function LandingComparison() {
  const [data, setData] = useState<LandingReachability | null>(null)
  const [platformId, setPlatformId] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    loadLandingReachability().then((d) => {
      if (cancelled) return
      setData(d)
      setPlatformId(d.platforms[0]?.id ?? null)
    })
    return () => {
      cancelled = true
    }
  }, [])

  if (!data || !platformId) return null

  const platform = data.platforms.find((p) => p.id === platformId)!
  const reachableCount = data.objects.filter((o) => o.reachableBy.includes(platformId)).length
  const { width_m, depth_m } = data.room
  const margin = Math.max(width_m, depth_m) * 0.08

  return (
    <div className="grid grid-cols-1 gap-6 md:grid-cols-[220px_1fr]">
      <div className="flex flex-col gap-1">
        {data.platforms.map((p) => (
          <button
            key={p.id}
            type="button"
            onClick={() => setPlatformId(p.id)}
            className={`rounded-md border-hair px-3 py-2 text-left transition-colors duration-100 ${
              p.id === platformId ? 'border-accent bg-accent-dim text-fg' : 'border-border text-subtle hover:bg-fg/5'
            }`}
          >
            <span className="block text-sm font-medium">{p.display_name}</span>
            <span className="block font-mono text-xs text-muted">{p.radius_m.toFixed(2)} m</span>
          </button>
        ))}
      </div>

      <div className="flex flex-col items-center gap-4 rounded-md border-hair border-border bg-surface p-6">
        <svg
          viewBox={`${-margin} ${-margin} ${width_m + margin * 2} ${depth_m + margin * 2}`}
          className="h-auto w-full max-w-md"
          style={{ transform: 'scaleY(-1)' }}
        >
          <rect x={0} y={0} width={width_m} height={depth_m} fill="var(--canvas)" stroke="var(--occ-obstacle)" strokeWidth={0.02} />
          {data.objects.map((o) => {
            const reachable = o.reachableBy.includes(platformId)
            return (
              <circle
                key={o.name}
                cx={o.position[0]}
                cy={o.position[2]}
                r={0.09}
                fill={reachable ? 'var(--data-object)' : 'var(--data-unreachable)'}
                className="transition-colors duration-200"
              />
            )
          })}
        </svg>

        <div className="text-center">
          <p className="font-mono text-[34px] leading-none tabular-nums text-fg">
            {reachableCount} / {data.objects.length}
          </p>
          <p className="text-[13px] text-muted">reachable by {platform.display_name}</p>
        </div>
      </div>
    </div>
  )
}
