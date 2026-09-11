import SegmentedControl from './ui/SegmentedControl'

export type MapViewMode = '2d' | '3d'

interface Props {
  value: MapViewMode
  onChange: (mode: MapViewMode) => void
  className?: string
}

/** The 2D|3D switch for the scene map - one shared area, not two panels. Active segment
 * is a neutral fill (spec 3.2/5.2): the accent color is spent on actions, not state. */
export default function MapViewToggle({ value, onChange, className = '' }: Props) {
  return (
    <SegmentedControl
      value={value}
      onChange={onChange}
      aria-label="View mode"
      className={className}
      segments={[
        { value: '2d', label: '2D' },
        { value: '3d', label: '3D' },
      ]}
    />
  )
}
