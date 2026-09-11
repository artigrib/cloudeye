import type { ReactNode } from 'react'
import type { RobotPlatformResponse } from '../api/types'
import { formatMetres } from '../lib/formatMeasure'
import PanelSection from './ui/PanelSection'
import Select from './ui/Select'

interface Props {
  /** Every registered platform (app/robots.py - eight today), the same list "Compare
   * all" enumerates. */
  platforms: RobotPlatformResponse[]
  selectedPlatformId: string | null
  onSelectPlatform: (id: string) => void
  /** One caption line under the picker - where this robot actually starts
   * (StartPositionCaption). It lives inside this section rather than in one of its own
   * because it is a fact ABOUT the selected platform: a section header, a coordinate and
   * a line of prose were three rows spent restating what the picker had just decided. */
  caption?: ReactNode
}

/** The robot picker. One control: which robot the screen's verdict is about.
 *
 * The clearance slider that used to sit under it is gone. It let the radius be any
 * value between 0.10 m and 0.60 m, which meant the screen's headline number was about
 * a robot that does not exist - and the two numbers this screen is actually for
 * (13 of 17 for TurtleBot3 Burger at 0.10 m, 10 of 17 for Husky A200 at 0.5528 m) were
 * two positions of a slider among fifty. The radius now comes from the selected
 * platform's registry entry and from nowhere else, so the verdict is always about a
 * real robot. */
export default function RobotPanel({ platforms, selectedPlatformId, onSelectPlatform, caption }: Props) {
  return (
    // px-s4 like every other section body in this column, and no bottom rule: a rule PLUS
    // the 24px gap between sections is the doubled separator ui/PanelSection's own comment
    // says was removed everywhere else. The body used to be px-3, four pixels tighter than
    // its own header.
    <PanelSection title="Robot" className="shrink-0">
      <div className="flex flex-col gap-s2 px-s4 pb-s2 pt-s1">
        <Select
          value={selectedPlatformId}
          onChange={onSelectPlatform}
          disabled={platforms.length === 0}
          placeholder="Loading…"
          aria-label="Robot platform"
          options={platforms.map((p) => ({
            value: p.id,
            label: p.display_name,
            sublabel: p.radius_m != null ? `${formatMetres(p.radius_m)} · ${p.kinematics}` : p.kinematics,
          }))}
        />
        {caption}
      </div>
    </PanelSection>
  )
}
