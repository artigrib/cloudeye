import { useState } from 'react'
import { IconRotate360, IconTarget } from '@tabler/icons-react'
import Badge from '../components/ui/Badge'
import Button from '../components/ui/Button'
import Card from '../components/ui/Card'
import Checkbox from '../components/ui/Checkbox'
import Input from '../components/ui/Input'
import PanelSection from '../components/ui/PanelSection'
import SegmentedControl from '../components/ui/SegmentedControl'
import Select from '../components/ui/Select'
import Slider from '../components/ui/Slider'
import Tooltip from '../components/ui/Tooltip'
import {
  ACCENT,
  DATA_OBJECT,
  DATA_ROBOT,
  DATA_TRAJECTORY,
  DATA_UNREACHABLE,
  HEIGHT_RAMP,
  OCC_FREE,
  OCC_OBSTACLE,
  OCC_UNKNOWN,
  oklchToSrgb,
  type OklchColor,
} from '../lib/tokens'

function Row({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="flex flex-col gap-3 border-b-hair border-border pb-8">
      <h2 className="font-mono text-[11px] uppercase tracking-[0.06em] text-muted">{title}</h2>
      <div className="flex flex-wrap items-start gap-4">{children}</div>
    </section>
  )
}

function Swatch({ name, token }: { name: string; token: OklchColor }) {
  const [r, g, b] = oklchToSrgb(token)
  const rgb = `rgb(${Math.round(r * 255)}, ${Math.round(g * 255)}, ${Math.round(b * 255)})`
  return (
    <div className="flex flex-col gap-1">
      <div className="h-12 w-24 rounded-sm border-hair border-border" style={{ background: rgb }} />
      <span className="font-mono text-[11px] text-muted">{name}</span>
    </div>
  )
}

/** Un-linked showcase of every primitive in every state, plus the full palette and type
 * scale - route: /_ui. Cheaper than Storybook, no extra dependency. */
export default function UiShowcase() {
  const [segment, setSegment] = useState<'2d' | '3d'>('2d')
  const [selectValue, setSelectValue] = useState<string | null>('burger')
  const [checked, setChecked] = useState(true)
  const [sliderValue, setSliderValue] = useState(0.25)
  const [inputValue, setInputValue] = useState('')

  return (
    <div className="mx-auto flex max-w-4xl flex-col gap-10 px-8 py-10">
      <h1 className="text-h1 font-medium text-fg">UI Showcase</h1>

      <Row title="Palette — surfaces & text">
        <Swatch name="bg" token={{ l: 0.13, c: 0, h: 0 }} />
        <Swatch name="surface" token={{ l: 0.17, c: 0, h: 0 }} />
        <Swatch name="surface-raised" token={{ l: 0.21, c: 0, h: 0 }} />
        <Swatch name="canvas" token={{ l: 0.09, c: 0, h: 0 }} />
        <Swatch name="fg" token={{ l: 0.97, c: 0, h: 0 }} />
        <Swatch name="subtle" token={{ l: 0.76, c: 0, h: 0 }} />
        <Swatch name="muted" token={{ l: 0.56, c: 0, h: 0 }} />
        <Swatch name="faint" token={{ l: 0.42, c: 0, h: 0 }} />
      </Row>

      <Row title="Palette — accent & data semantics">
        <Swatch name="accent" token={ACCENT} />
        <Swatch name="data-object" token={DATA_OBJECT} />
        <Swatch name="data-trajectory" token={DATA_TRAJECTORY} />
        <Swatch name="data-robot" token={DATA_ROBOT} />
        <Swatch name="data-unreachable" token={DATA_UNREACHABLE} />
        <Swatch name="occ-free" token={OCC_FREE} />
        <Swatch name="occ-obstacle" token={OCC_OBSTACLE} />
        <Swatch name="occ-unknown" token={OCC_UNKNOWN} />
      </Row>

      <Row title="Palette — height ramp (h-0 → h-4)">
        {HEIGHT_RAMP.map((token, i) => (
          <Swatch key={i} name={`h-${i}`} token={token} />
        ))}
      </Row>

      <Row title="Type scale">
        <div className="flex flex-col gap-2">
          <p className="text-display font-medium text-fg">Display</p>
          <p className="text-h1 font-medium text-fg">Heading 1</p>
          <p className="text-h2 font-medium text-fg">Heading 2</p>
          <p className="text-h3 font-medium text-fg">Heading 3</p>
          <p className="text-body text-subtle">Body text, the default size.</p>
          <p className="text-small text-subtle">Small text - captions, secondary copy.</p>
          <p className="font-mono text-micro uppercase tracking-[0.06em] text-muted">
            Micro, mono only - OBJECTS (40)
          </p>
        </div>
      </Row>

      <Row title="Button">
        <div className="flex flex-col gap-2">
          <div className="flex gap-2">
            <Button variant="primary">Send</Button>
            <Button variant="secondary">Download for Isaac Sim</Button>
            <Button variant="ghost">Reset</Button>
          </div>
          <div className="flex items-center gap-2">
            <Button size="compact">Compact</Button>
            <Button size="default">Default</Button>
            <Button size="large">Large</Button>
          </div>
          <div className="flex gap-2">
            <Button icon={<IconTarget size={16} />}>With icon</Button>
            <Button disabled>Disabled</Button>
          </div>
        </div>
      </Row>

      <Row title="SegmentedControl">
        <SegmentedControl
          value={segment}
          onChange={setSegment}
          segments={[
            { value: '2d', label: '2D' },
            { value: '3d', label: '3D' },
          ]}
          aria-label="View mode"
        />
      </Row>

      <Row title="Select">
        <Select
          className="w-64"
          value={selectValue}
          onChange={setSelectValue}
          aria-label="Robot platform"
          options={[
            { value: 'burger', label: 'TurtleBot3 Burger', sublabel: '0.10 m · differential_drive' },
            { value: 'go2', label: 'Unitree Go2', sublabel: '0.25 m · legged' },
            { value: 'husky', label: 'Husky A200', sublabel: '0.55 m · skid_steer' },
          ]}
        />
      </Row>

      <Row title="Checkbox">
        <div className="flex flex-col gap-2">
          <Checkbox checked={checked} onChange={setChecked} label="Include point cloud" />
          <Checkbox checked={false} onChange={() => {}} label="Disabled" disabled />
        </div>
      </Row>

      <Row title="Slider">
        <div className="w-72">
          <Slider
            value={sliderValue}
            onChange={setSliderValue}
            min={0.1}
            max={0.6}
            label="Clearance"
            formatValue={(v) => `${v.toFixed(2)} m`}
          />
        </div>
      </Row>

      <Row title="Input">
        <Input value={inputValue} onChange={(e) => setInputValue(e.target.value)} placeholder="Room name" className="w-64" />
      </Row>

      <Row title="Card">
        <Card className="w-64" hoverable>
          <p className="text-sm font-medium text-fg">Living Room</p>
          <p className="mt-1 text-xs text-muted">Sep 1, 2026</p>
        </Card>
      </Row>

      <Row title="Badge">
        <div className="flex gap-2">
          <Badge variant="unreachable">unreachable</Badge>
          <Badge variant="live">live</Badge>
          <Badge variant="neutral">fragment</Badge>
        </div>
      </Row>

      <Row title="PanelSection">
        <div className="w-72 rounded-md border-hair border-border bg-surface">
          <PanelSection title="Objects" count={40}>
            <p className="px-4 py-2 text-sm text-subtle">bed, rug, chair…</p>
          </PanelSection>
        </div>
      </Row>

      <Row title="Tooltip">
        <Tooltip content="Reset view">
          <Button icon={<IconRotate360 size={16} />} aria-label="Reset view" />
        </Tooltip>
      </Row>
    </div>
  )
}
