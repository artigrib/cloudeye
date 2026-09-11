import { describe, expect, it } from 'vitest'
import {
  applyCompletion,
  completionNames,
  currentToken,
  suggestObjectNames,
  wrapIndex,
} from './objectNameComplete'

const HERO = ['bed', 'chair', 'desk', 'television', 'lamp', 'curtain', 'nightstand', 'pillow']

describe('completionNames', () => {
  it('drops fragments and duplicates, keeping list order', () => {
    const names = completionNames([
      { name: 'lamp', is_fragment: false },
      { name: 'Lamp', is_fragment: false },
      { name: 'shard', is_fragment: true },
      { name: 'desk', is_fragment: false },
    ])
    expect(names).toEqual(['lamp', 'desk'])
  })
})

describe('currentToken', () => {
  it('is the last whitespace-delimited word', () => {
    expect(currentToken('go to the si')).toBe('si')
    expect(currentToken('lamp')).toBe('lamp')
  })

  it('is empty for an empty input or a trailing space', () => {
    expect(currentToken('')).toBe('')
    // A trailing space means "I finished that word" - offering the whole object list at
    // that moment is noise, and it would also swallow the Enter meant to send.
    expect(currentToken('go to the sink ')).toBe('')
  })
})

describe('suggestObjectNames', () => {
  it('completes the last word, not the whole field', () => {
    expect(suggestObjectNames('go to the la', HERO)).toEqual(['lamp'])
  })

  it('puts prefix matches before substring matches', () => {
    const names = ['nightstand', 'stand mixer']
    // "stand" is a prefix of one and inside the other; the prefix match leads.
    expect(suggestObjectNames('stand', names)).toEqual(['stand mixer', 'nightstand'])
  })

  it('is case-insensitive in both directions', () => {
    expect(suggestObjectNames('DE', HERO)).toEqual(['desk'])
    expect(suggestObjectNames('de', ['DESK'])).toEqual(['DESK'])
  })

  it('offers nothing for an empty token', () => {
    expect(suggestObjectNames('', HERO)).toEqual([])
    expect(suggestObjectNames('go to the desk ', HERO)).toEqual([])
  })

  it('offers nothing once the token already IS the name', () => {
    // Otherwise the popup stays open on a finished word and eats the Enter that was
    // meant to send the command.
    expect(suggestObjectNames('desk', HERO)).toEqual([])
  })

  it('caps the list', () => {
    const many = Array.from({ length: 20 }, (_, i) => `object${i}`)
    expect(suggestObjectNames('object', many, 6)).toHaveLength(6)
  })
})

describe('applyCompletion', () => {
  it('replaces the last token and leaves a trailing space', () => {
    expect(applyCompletion('go to the la', 'lamp')).toBe('go to the lamp ')
  })

  it('appends when there is no token yet', () => {
    expect(applyCompletion('go to the ', 'lamp')).toBe('go to the lamp ')
  })

  it('handles a single-word input', () => {
    expect(applyCompletion('la', 'lamp')).toBe('lamp ')
  })
})

describe('wrapIndex', () => {
  it('cycles in both directions', () => {
    expect(wrapIndex(3, 3)).toBe(0)
    expect(wrapIndex(-1, 3)).toBe(2)
    expect(wrapIndex(1, 3)).toBe(1)
  })

  it('is -1 for an empty list', () => {
    expect(wrapIndex(0, 0)).toBe(-1)
  })
})
