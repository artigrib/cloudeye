import { describe, expect, it } from 'vitest'
import {
  bubbleClassName,
  describeOutcome,
  describeOutcomeDetail,
  isChatLlmNotConnected,
  isProviderConfigError,
  pathLengthM,
  totalPathLengthM,
} from './ChatPanel'
import type { CommandResponse } from '../api/types'

// M7b addition (e): the command panel shows path length in metres and duration
// ("2.34 m · 4.7s") instead of a bare duration ("Move · 0.1s") - these are the pure
// geometry helpers behind that display, tested independent of rendering.
describe('pathLengthM', () => {
  it('sums consecutive-point Euclidean distances', () => {
    expect(pathLengthM([[0, 0], [3, 4]])).toBeCloseTo(5) // 3-4-5 triangle
    expect(
      pathLengthM([
        [0, 0],
        [3, 4],
        [3, 0],
      ]),
    ).toBeCloseTo(9) // 5 + 4
  })

  it('is 0 for null, empty or single-point paths', () => {
    expect(pathLengthM(null)).toBe(0)
    expect(pathLengthM([])).toBe(0)
    expect(pathLengthM([[1, 1]])).toBe(0)
  })
})

describe('totalPathLengthM', () => {
  function moveStep(path: [number, number][]): CommandResponse['steps'][number] {
    return { type: 'move', path, duration_sec: 0, object: null, position: null, at: null, length_m: null }
  }
  function otherStep(type: 'pick' | 'place', extra: Partial<CommandResponse['steps'][number]> = {}): CommandResponse['steps'][number] {
    return { type, path: null, duration_sec: 2, object: null, position: null, at: null, length_m: null, ...extra }
  }

  it('sums path length across every move step, ignoring pick/place', () => {
    const steps: CommandResponse['steps'] = [
      moveStep([
        [0, 0],
        [3, 4],
      ]), // 5
      otherStep('pick', { object: 'mug' }),
      moveStep([
        [3, 4],
        [3, 0],
      ]), // 4
      otherStep('place', { at: 'table' }),
    ]
    expect(totalPathLengthM(steps)).toBeCloseTo(9)
  })

  it('is 0 for a command with no move steps', () => {
    const steps: CommandResponse['steps'] = [otherStep('pick', { object: 'mug' })]
    expect(totalPathLengthM(steps)).toBe(0)
  })
})

// 2026-09-09: the panel replays listCommands on mount, so one unconfigured deployment
// left a permanent red "OPENROUTER_API_KEY is not configured" box in the COMMANDS panel
// of every later session - it was in demo take 2's shots 4, 5 and 6. A configuration
// failure is about the server, not about the command that hit it, so it is dropped from
// the replay; a live attempt still shows it.
describe('isProviderConfigError', () => {
  it('matches the missing-key failure the backend actually writes', () => {
    expect(isProviderConfigError('OPENROUTER_API_KEY is not configured')).toBe(true)
    expect(isProviderConfigError('VERTEX_API_KEY is not configured')).toBe(true)
  })

  it('does not match a real per-command failure', () => {
    expect(isProviderConfigError('no path found from (21, 74) to (8, 18) - goal is unreachable from start')).toBe(false)
    expect(isProviderConfigError("no object matching 'sink' in this scene")).toBe(false)
    expect(isProviderConfigError('OpenRouter API error: rate limited')).toBe(false)
    expect(isProviderConfigError(null)).toBe(false)
    expect(isProviderConfigError(undefined)).toBe(false)
  })
})

// Step 4: the chat LLM named in the workspace's job spec may have no credentials here -
// it has none in this worktree, by design. That is a fact about the deployment, not
// about the command, so the panel answers it with "Chat LLM not connected - templated
// commands still work" rather than "not understood", and never replays it.
describe('isChatLlmNotConnected', () => {
  function command(overrides: Partial<CommandResponse>): CommandResponse {
    return {
      command_id: 'c1',
      scene_id: 's1',
      user_text: 'what is in this room?',
      action: null,
      parsed_action: null,
      steps: [],
      total_duration_sec: null,
      total_length_m: null,
      provider_used: 'openrouter-nemotron',
      error_code: null,
      status: 'failed',
      error_message: null,
      created_at: '2026-09-09T12:00:00Z',
      ...overrides,
    }
  }

  it("trusts the backend's own code, whichever credential is missing", () => {
    for (const message of [
      'OPENROUTER_API_KEY is not configured',
      'GCP_PROJECT_ID is not configured',
      'Vertex credentials are not configured: ADC not configured',
    ]) {
      expect(isChatLlmNotConnected(command({ error_code: 'chat_llm_not_connected', error_message: message }))).toBe(true)
    }
  })

  it('still recognises a row written before the code field existed', () => {
    expect(isChatLlmNotConnected(command({ error_message: 'OPENROUTER_API_KEY is not configured' }))).toBe(true)
  })

  it('leaves a real per-command failure alone - that one belongs to its command', () => {
    expect(isChatLlmNotConnected(command({ error_message: "no object matching 'sink' in this scene" }))).toBe(false)
    expect(isChatLlmNotConnected(command({ error_message: 'openrouter-nemotron API error: rate limited' }))).toBe(false)
  })

  it('is false for a command that did not fail, and for none at all', () => {
    expect(isChatLlmNotConnected(command({ status: 'done', error_code: 'chat_llm_not_connected' }))).toBe(false)
    expect(isChatLlmNotConnected(null)).toBe(false)
    expect(isChatLlmNotConnected(undefined)).toBe(false)
  })
})

// Chat bubbles used to lose newlines and overflow on one unbreakable token: five copies of
// the same class list, none of them with `whitespace-pre-wrap` or `break-words`. The
// classes are the whole behaviour here, so they are what is pinned - and `min-w-0` with
// them, because without it a flex item refuses to shrink below its longest word and
// `break-words` never gets to apply.
describe('bubbleClassName', () => {
  const WRAPPING = ['whitespace-pre-wrap', 'break-words', 'min-w-0']

  it('wraps every kind of bubble', () => {
    for (const side of ['me', 'them'] as const) {
      for (const tone of ['plain', 'error'] as const) {
        const cls = bubbleClassName(side, tone).split(' ')
        for (const need of WRAPPING) expect(cls).toContain(need)
      }
    }
  })

  it('puts the user on the right and everything else on the left', () => {
    expect(bubbleClassName('me').split(' ')).toContain('ml-auto')
    expect(bubbleClassName('them').split(' ')).not.toContain('ml-auto')
  })

  it('gives an error bubble the unreachable colour, and a plain one the surface', () => {
    expect(bubbleClassName('them', 'error')).toContain('text-data-unreachable')
    expect(bubbleClassName('them', 'plain')).toContain('text-subtle')
  })

  it('appends a caller class and drops an absent one', () => {
    expect(bubbleClassName('them', 'plain', 'mt-2').endsWith('mt-2')).toBe(true)
    expect(bubbleClassName('them', 'plain', undefined)).not.toContain('undefined')
  })
})

// ONE CARD PER REPLY. A finished command used to render as three lines that all said the
// same thing: a "GOTO → CABINET" chip echoing the request, a per-step
// "Move · 2.4s · 1.20 m", and a "Total: 2.4s · 1.20 m" under it. For the single-step
// command a goto plans, the last two are the same number twice.
describe('describeOutcome', () => {
  const objects = [
    { id: '1', name: 'Cabinet', is_fragment: false },
    { id: '2', name: 'bed', is_fragment: false },
  ] as never as Parameters<typeof describeOutcome>[1]

  const command = (over: Partial<CommandResponse>): CommandResponse =>
    ({
      status: 'done',
      parsed_action: null,
      steps: [],
      total_duration_sec: null,
      total_length_m: null,
      ...over,
    }) as CommandResponse

  it('names what it reached, in the scene\'s own spelling of the name', () => {
    // The parsed action carries the user's lowercase "cabinet"; the card says "Cabinet".
    const c = command({
      parsed_action: { action: 'goto', target: 'cabinet' },
      steps: [{ type: 'move', duration_sec: 2.4, length_m: 1.2, path: [[0, 0], [1, 0]] }] as never,
    })
    expect(describeOutcome(c, objects)).toBe('Reached Cabinet')
  })

  it('does not call a zero-length route a journey', () => {
    // "Move · 0.0s · 0.00 m" was what the old panel printed when the robot was already
    // standing there, which reads as a bug rather than as an answer.
    const c = command({
      parsed_action: { action: 'goto', target: 'bed' },
      steps: [{ type: 'move', duration_sec: 0, length_m: 0, path: [[1, 1]] }] as never,
    })
    expect(describeOutcome(c, objects)).toBe('Already at bed')
  })

  it('says pick and place for the steps that are not a move', () => {
    const picked = command({
      parsed_action: { action: 'take', target: 'bed' },
      steps: [{ type: 'pick', duration_sec: 1, object: 'bed' }] as never,
    })
    expect(describeOutcome(picked, objects)).toBe('Picked up bed')
  })

  it('still answers when nothing named the target', () => {
    const c = command({ steps: [{ type: 'move', duration_sec: 3, length_m: 2, path: [[0, 0], [2, 0]] }] as never })
    expect(describeOutcome(c, objects)).toBe('Arrived')
  })
})

describe('describeOutcomeDetail', () => {
  const base = { status: 'done', parsed_action: null, steps: [] } as never as CommandResponse

  it('is the caption under the title: how long, how far, once', () => {
    expect(describeOutcomeDetail({ ...base, total_duration_sec: 6.8, total_length_m: 3.38 })).toBe('6.8s · 3.38 m')
  })

  it('drops the distance when there is none rather than printing 0.00 m', () => {
    expect(describeOutcomeDetail({ ...base, total_duration_sec: 1.2, total_length_m: 0 })).toBe('1.2s')
  })

  it('is null when there is no journey to describe at all', () => {
    expect(describeOutcomeDetail({ ...base, total_duration_sec: null, total_length_m: 0 })).toBeNull()
  })
})
