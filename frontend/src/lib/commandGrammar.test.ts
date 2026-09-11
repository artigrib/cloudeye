import { describe, expect, it } from 'vitest'
import { TEMPLATED_FORMS, parseCommand, type Vocabulary } from './commandGrammar'

const vocab: Vocabulary = {
  objects: ['desk', 'bed', 'lamp', 'nightstand', 'coffee table', 'television'],
  robots: ['TurtleBot3 Burger', 'Husky A200', 'Unitree Go2'],
}

const parse = (text: string) => parseCommand(text, vocab)

describe('go to <object>', () => {
  it('parses the canonical form', () => {
    expect(parse('go to desk')).toEqual({ kind: 'goto', object: 'desk' })
  })

  it('accepts goto, go, odd spacing and any case', () => {
    for (const text of ['goto desk', 'go desk', '  GO   TO   Desk  ', 'GoTo desk']) {
      expect(parse(text)).toEqual({ kind: 'goto', object: 'desk' })
    }
  })

  it('handles a two-word object name', () => {
    expect(parse('go to coffee table')).toEqual({ kind: 'goto', object: 'coffee table' })
  })

  it('returns the name as the scene spells it, not as it was typed', () => {
    expect(parse('go to TELEVISION')).toEqual({ kind: 'goto', object: 'television' })
  })

  it('falls through to freeform for an object this room does not have', () => {
    expect(parse('go to the sink')).toEqual({ kind: 'freeform', text: 'go to the sink' })
  })

  it('never matches half a word', () => {
    expect(parse('go to bedside')).toEqual({ kind: 'freeform', text: 'go to bedside' })
  })
})

describe('can <robot> reach <object>', () => {
  it('parses the canonical form', () => {
    expect(parse('can Husky A200 reach desk')).toEqual({
      kind: 'canReach',
      robot: 'Husky A200',
      object: 'desk',
    })
  })

  it('tolerates a question mark and any case', () => {
    expect(parse('can husky a200 reach the desk?')).toEqual({ kind: 'freeform', text: 'can husky a200 reach the desk?' })
    expect(parse('can husky a200 reach desk?')).toEqual({
      kind: 'canReach',
      robot: 'Husky A200',
      object: 'desk',
    })
  })

  it('accepts "get to" as well as "reach"', () => {
    expect(parse('can Unitree Go2 get to lamp')).toEqual({
      kind: 'canReach',
      robot: 'Unitree Go2',
      object: 'lamp',
    })
  })

  it('is freeform for a robot that is not a registered platform', () => {
    expect(parse('can a forklift reach desk').kind).toBe('freeform')
  })
})

describe('compare all', () => {
  it('parses exactly, in any case', () => {
    expect(parse('compare all')).toEqual({ kind: 'compareAll' })
    expect(parse('  Compare   All  ')).toEqual({ kind: 'compareAll' })
  })

  it('does not swallow a longer sentence that merely starts with it', () => {
    expect(parse('compare all the robots for me').kind).toBe('freeform')
  })
})

describe('move <object> <dx> <dy>', () => {
  it('parses two signed distances', () => {
    expect(parse('move desk 0.5 -1.25')).toEqual({ kind: 'move', object: 'desk', dx: 0.5, dy: -1.25 })
  })

  it('accepts a metres unit on either number', () => {
    expect(parse('move desk 0.5m -1m')).toEqual({ kind: 'move', object: 'desk', dx: 0.5, dy: -1 })
    expect(parse('move desk 0.5 m -1 m').kind).toBe('freeform') // four tokens, not two
  })

  it('reads the numbers off the end, so a two-word object still parses', () => {
    expect(parse('move coffee table 1 0')).toEqual({
      kind: 'move',
      object: 'coffee table',
      dx: 1,
      dy: 0,
    })
  })

  it('is freeform when a distance is missing or not a number', () => {
    expect(parse('move desk 0.5').kind).toBe('freeform')
    expect(parse('move desk left a bit').kind).toBe('freeform')
    expect(parse('move desk').kind).toBe('freeform')
  })
})

describe('everything else', () => {
  it('is freeform, carrying the text through for the chat LLM', () => {
    expect(parse('what is in this room?')).toEqual({ kind: 'freeform', text: 'what is in this room?' })
  })

  it('treats empty input as freeform rather than throwing', () => {
    expect(parse('   ')).toEqual({ kind: 'freeform', text: '' })
  })
})

describe('TEMPLATED_FORMS', () => {
  it('lists exactly the four forms the parser accepts', () => {
    expect(TEMPLATED_FORMS).toEqual([
      'go to <object>',
      'can <robot> reach <object>',
      'compare all',
      'move <object> <dx> <dy>',
    ])
  })
})
