import { useEffect, useState } from 'react'
import { getReachability } from '../api/scenes'
import type { RobotPlatformResponse } from '../api/types'
import type { ObjectGroup } from '../lib/objectGroups'
import PanelSection from './ui/PanelSection'

/** undefined = still in flight, null = the request for this platform's radius failed. */
type Row = { reachable: Set<string> } | null | undefined

interface Props {
  sceneId: string
  /** The WHOLE registry, not the picker's two - the point of this table is the
   * platforms the picker does not offer. */
  platforms: RobotPlatformResponse[]
  /** The same groups, in the same order, the object list is showing right now - see
   * lib/objectGroups.ts. These are the columns. */
  groups: ObjectGroup[]
  selectedPlatformId: string | null
}

/** One row per platform, one column per object - the same objects, in the same order,
 * as the object list beside it.
 *
 * The table it replaces gave each platform a single "reachable 13/17" number, which
 * says how many but never which: two platforms reaching 13 of 17 look identical there
 * even when they reach a different 13. Reading down a column here shows which robots
 * lose a particular object, which is the comparison anyone actually came for. */
export default function PlatformComparison({ sceneId, platforms, groups, selectedPlatformId }: Props) {
  const [rows, setRows] = useState<Record<string, Row>>({})

  useEffect(() => {
    if (platforms.length === 0) return
    let cancelled = false
    setRows({})
    for (const p of platforms) {
      if (p.radius_m == null) continue
      getReachability(sceneId, p.radius_m, [], p.id)
        .then((r) => {
          if (cancelled) return
          setRows((prev) => ({ ...prev, [p.id]: { reachable: new Set(r.reachable_object_ids) } }))
        })
        .catch((err) => {
          console.error('Failed to fetch reachability for platform %s', p.id, err)
          if (cancelled) return
          setRows((prev) => ({ ...prev, [p.id]: null }))
        })
    }
    return () => {
      cancelled = true
    }
  }, [sceneId, platforms])

  const total = groups.reduce((sum, g) => sum + g.items.length, 0)

  return (
    <PanelSection title="Compare all" count={platforms.length} className="min-h-0 flex-1 overflow-hidden">
      <div className="h-full overflow-auto px-2 pb-2 pt-1" data-testid="platform-comparison">
        <table className="border-separate border-spacing-0 text-[11px]">
          <thead>
            <tr>
              <th className="sticky left-0 z-10 bg-surface px-1.5 pb-1 text-left align-bottom font-normal text-muted">
                platform
              </th>
              {groups.map((g) => (
                <th key={g.key} className="px-0 pb-1 align-bottom font-normal" data-comparison-column={g.key}>
                  <span
                    className="block h-[74px] w-[18px] truncate text-left text-muted"
                    style={{ writingMode: 'vertical-rl', transform: 'rotate(180deg)' }}
                    title={`${g.name} (${g.items.length})`}
                  >
                    {g.name}
                  </span>
                </th>
              ))}
              <th className="sticky right-0 z-10 bg-surface px-1.5 pb-1 text-right align-bottom font-normal text-muted">
                total
              </th>
            </tr>
          </thead>
          <tbody>
            {platforms.map((p) => {
              const row = rows[p.id]
              const selected = p.id === selectedPlatformId
              const count = row ? groups.reduce((n, g) => n + g.items.filter((o) => row.reachable.has(o.id)).length, 0) : null
              return (
                <tr key={p.id} data-testid="platform-comparison-row" data-platform-id={p.id}>
                  <td
                    className={`sticky left-0 z-10 max-w-[120px] truncate px-1.5 py-0.5 ${
                      selected ? 'bg-accent-dim text-fg' : 'bg-surface text-subtle'
                    }`}
                    title={p.display_name}
                  >
                    {p.display_name}
                  </td>
                  {groups.map((g) => {
                    const reachable = row ? g.items.filter((o) => row.reachable.has(o.id)).length : null
                    const label =
                      p.radius_m == null
                        ? 'n/a'
                        : row === undefined
                          ? '·'
                          : row === null || reachable === null
                            ? '—'
                            : reachable === g.items.length
                              ? '✓'
                              : reachable === 0
                                ? '·'
                                : `${reachable}/${g.items.length}`
                    const missed = row != null && reachable !== null && reachable < g.items.length
                    return (
                      <td
                        key={g.key}
                        data-comparison-cell={`${p.id}:${g.key}`}
                        title={`${p.display_name} · ${g.name}: ${
                          reachable === null ? 'unknown' : `${reachable} of ${g.items.length} reachable`
                        }`}
                        className={`px-0 py-0.5 text-center font-mono tabular-nums ${
                          missed ? 'text-data-unreachable' : 'text-subtle'
                        } ${selected ? 'bg-accent-dim' : ''}`}
                      >
                        {label}
                      </td>
                    )
                  })}
                  <td
                    className={`sticky right-0 z-10 px-1.5 py-0.5 text-right font-mono tabular-nums ${
                      selected ? 'bg-accent-dim text-fg' : 'bg-surface text-muted'
                    }`}
                  >
                    {count === null ? '…' : `${count}/${total}`}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </PanelSection>
  )
}
