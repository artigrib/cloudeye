// Content for the capture-onboarding screen (see CaptureGuide component). Kept as data
// separate from rendering because this list is expected to grow as more failure modes
// show up in practice - see the six rules' rationale in the project's capture-failure
// notes (long blank walls, glossy floors, blur, missing floor tilt).

import {
  IconArrowBigDownLine,
  IconClock,
  IconRoute,
  IconSofa,
  IconSparkles,
  IconWalk,
  type TablerIcon,
} from '@tabler/icons-react'

export interface CaptureGuideRule {
  icon: TablerIcon
  title: string
  text: string
}

export const CAPTURE_GUIDE_RULES: CaptureGuideRule[] = [
  {
    icon: IconWalk,
    title: 'Go slow',
    text: "Walk at a steady pace, no sudden moves or sharp turns. Blurry frames get discarded and can't be used to build geometry.",
  },
  {
    icon: IconArrowBigDownLine,
    title: 'Tilt the camera down',
    text: "The floor needs to be in frame. Without it the robot won't know where it can drive.",
  },
  {
    icon: IconRoute,
    title: "Walk the room's perimeter",
    text: "Don't spin in place. The reconstruction needs different angles of the same objects.",
  },
  {
    icon: IconSofa,
    title: 'Stay close to furniture',
    text: 'Reconstruction relies on detail. A bare wall gives the model nothing to latch onto.',
  },
  {
    icon: IconSparkles,
    title: 'Avoid mirrors and glossy floors in frame',
    text: 'The model mistakes reflections for real objects.',
  },
  {
    icon: IconClock,
    title: '30-90 seconds is enough',
    text: "Filming longer doesn't improve the result.",
  },
]

export const CAPTURE_GUIDE_UNSUITABLE =
  'Not suitable for: long empty hallways, rooms with mirrored walls, or filming while walking briskly.'

const SEEN_KEY = 'cloudeye:captureGuideSeen'

export function hasSeenCaptureGuide(): boolean {
  try {
    return localStorage.getItem(SEEN_KEY) === '1'
  } catch {
    return true
  }
}

export function markCaptureGuideSeen(): void {
  try {
    localStorage.setItem(SEEN_KEY, '1')
  } catch {
    // localStorage unavailable (private mode, disabled) - non-fatal, just means the
    // guide auto-shows again next time.
  }
}
