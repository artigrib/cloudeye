import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from 'react'
import { ApiError, NotFoundError } from '../api/client'
import { listCommands, postCommand } from '../api/scenes'
import type { CommandResponse, RobotPlatformResponse, SceneObjectResponse } from '../api/types'
import { TEMPLATED_FORMS, parseCommand } from '../lib/commandGrammar'
import { formatMetres, formatSeconds } from '../lib/formatMeasure'
import {
  applyCompletion,
  completionNames,
  suggestObjectNames,
  wrapIndex,
} from '../lib/objectNameComplete'
import Badge from './ui/Badge'
import Button from './ui/Button'
import Input from './ui/Input'
import PanelSection from './ui/PanelSection'

interface Entry {
  id: string
  userText: string
  command: CommandResponse | null // null = object genuinely not found (404), not a plan failure
  notFoundMessage?: string
  /** True only while this entry's request is still in flight - a placeholder "parsing…"
   * chip shows until the response (success, plan failure, or not-found) replaces it. */
  pending?: boolean
  /** An answer this panel produced itself, with no model and no round trip - the reply
   * to a templated command that isn't a route ("can X reach Y", "compare all",
   * "move ..."). `approximate` badges the one that is: a move is a bbox cut/paste on
   * the occupancy grid, not a re-scan of the room. */
  localReply?: { text: string; approximate?: boolean }
}

/** The object a name refers to, using the same tie-break the backend's `resolve_object`
 * uses for duplicates: best-observed first. A name the scene doesn't have returns
 * undefined and the command falls through to the chat LLM, which can say so in words. */
function resolveObjectByName(
  objects: readonly SceneObjectResponse[],
  name: string,
): SceneObjectResponse | undefined {
  const lower = name.toLowerCase()
  return objects
    .filter((o) => !o.is_fragment && o.name.toLowerCase() === lower)
    .sort((a, b) => b.num_views - a.num_views || b.num_points - a.num_points)[0]
}

export interface ChatPanelHandle {
  /** Issues a direct "goto" for an already-known object - skips the VLM parse. Used by
   * an object row's Go button, by double-clicking a row, and by double-clicking the
   * object on the map. */
  goto: (objectId: string, name: string) => Promise<void>
}

/** M7b addition (e): sum of consecutive-point Euclidean distances along a planned
 * path (metres, XZ-plane points as returned by `app/services/command_service.py`'s
 * `build_steps` - `[[x, z], ...]`). Geometric ground truth, independent of the
 * server's `duration_sec` (which is itself `length / speed_mps`, see
 * `app/config.py`'s `robot_speed_mps` / `demo/record_isaac.py`'s
 * `PATH_PLANNING_SPEED_MPS` - both 0.5 m/s today, not the same symbol) - computing
 * length from the points already on the response avoids needing a new backend
 * field or assuming the two speed constants stay in sync. Null/short paths -> 0. */
export function pathLengthM(path: readonly (readonly [number, number])[] | null): number {
  if (!path || path.length < 2) return 0
  let total = 0
  for (let i = 1; i < path.length; i++) {
    const [x0, z0] = path[i - 1]
    const [x1, z1] = path[i]
    total += Math.hypot(x1 - x0, z1 - z0)
  }
  return total
}

/** Total path length (metres) across every 'move' step in a command - what the
 * "Total:" line shows alongside `total_duration_sec`. */
export function totalPathLengthM(steps: CommandResponse['steps']): number {
  return steps.filter((s) => s.type === 'move').reduce((sum, s) => sum + pathLengthM(s.path), 0)
}

function describeStep(step: CommandResponse['steps'][number]): string {
  if (step.type === 'move') {
    // length_m is the runtime planner's own route length (app.services.pathfinding.
    // PathResult.length_m) - shown alongside duration, not in place of it. Older
    // persisted commands (from before this field existed) have it null; fall back to
    // the client-computed geometric length from step.path in that case.
    const lengthM = step.length_m ?? pathLengthM(step.path)
    return `Move · ${formatSeconds(step.duration_sec)} · ${formatMetres(lengthM)}`
  }
  if (step.type === 'pick') return `Pick up ${step.object ?? 'object'}`
  return `Place at ${step.at ?? 'target'}`
}

/** What one finished command DID, as a sentence - the card's title.
 *
 * A reply used to be three lines that all said the same thing: a `goto → cabinet` chip, a
 * per-step `Move · 2.4s · 1.20 m`, and a `Total: 2.4s · 1.20 m` under it. For the single-
 * step command a goto plans, the step line and the total line are the same number twice,
 * and the chip is the request echoed back. One card, one title, one caption. */
export function describeOutcome(
  command: CommandResponse,
  objects: readonly SceneObjectResponse[],
): string {
  const target =
    (typeof command.parsed_action?.target === 'string' ? command.parsed_action.target : null) ??
    command.steps.find((s) => s.object)?.object ??
    null
  const named = target ? (objects.find((o) => o.name.toLowerCase() === target.toLowerCase())?.name ?? target) : null
  const moved = command.steps.some((s) => s.type === 'move' && (s.length_m ?? pathLengthM(s.path)) > 0.005)
  const picked = command.steps.some((s) => s.type === 'pick')
  const placed = command.steps.some((s) => s.type === 'place')

  if (picked && named) return `Picked up ${named}`
  if (placed && named) return `Placed at ${named}`
  // A goto whose route is zero-length is not a journey - the robot was already standing
  // there, and "Move · 0.0s · 0.00 m" reads as a bug rather than as an answer.
  if (!moved) return named ? `Already at ${named}` : 'Already there'
  return named ? `Reached ${named}` : 'Arrived'
}

/** The caption under the title: how long it took and how far it went, once. Null when the
 * command carries no journey to describe. */
export function describeOutcomeDetail(command: CommandResponse): string | null {
  const lengthM = command.total_length_m ?? totalPathLengthM(command.steps)
  if (command.total_duration_sec === null && !lengthM) return null
  if (!lengthM) return formatSeconds(command.total_duration_sec)
  return `${formatSeconds(command.total_duration_sec)} · ${formatMetres(lengthM)}`
}

/** Deduplicated, non-fragment object names for this scene, in list order - what a
 * "target not found" error offers as the real alternatives (see STRINGS.notFound). */
function existingObjectNames(objects: readonly SceneObjectResponse[]): string[] {
  const seen = new Set<string>()
  const names: string[] = []
  for (const o of objects) {
    if (o.is_fragment || seen.has(o.name)) continue
    seen.add(o.name)
    names.push(o.name)
  }
  return names
}

const STRINGS = {
  notFound: (name: string | undefined, objects: readonly SceneObjectResponse[]) => {
    const subject = name ? `"${name}"` : 'That object'
    const names = existingObjectNames(objects)
    if (names.length === 0) return `${subject} isn't in this scene.`
    return `${subject} isn't in this scene - try: ${names.join(', ')}.`
  },
  commandFailed: "Couldn't plan that command.",
  networkError: 'Request failed.',
  cantReach: (name?: string) => (name ? `Can't reach "${name}".` : "Can't find a path there."),
  /** The chat LLM named in this workspace's job spec has no credentials on this
   * deployment. Deliberately NOT "not understood": the command was understood
   * perfectly well as something only a model can answer, and the four templated
   * commands are unaffected - saying otherwise would send the user away from a screen
   * that still works. */
  chatLlmNotConnected: `Chat LLM not connected - templated commands still work: ${TEMPLATED_FORMS.join(', ')}.`,
}

/** Three staggered dots while the agent is composing an answer.
 *
 * Was `ParsedActionChip`, which also rendered the parsed `{action, target}` back as a
 * badge once the answer arrived - "GOTO → CABINET" directly above the user's own
 * "goto cabinet", and directly above a card that says what actually happened. The echo is
 * gone; the indicator, which says something the user cannot otherwise see, is not. */
function AgentTyping() {
  return (
    <span
      className="flex h-6 w-fit items-center gap-1 rounded-lg rounded-bl-sm border-hair border-border bg-surface px-s2"
      data-testid="agent-typing"
      aria-label="The agent is answering"
    >
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="h-1 w-1 animate-pulse rounded-full bg-muted"
          style={{ animationDelay: `${i * 160}ms` }}
        />
      ))}
    </span>
  )
}

/** Pulls the object name out of the backend's (always-English) 404 detail, e.g.
 * "no object matching 'sink' in this scene", so a localized message can still name it. */
function extractMissingName(detail?: string): string | undefined {
  return detail?.match(/matching '(.+?)' in this scene/)?.[1]
}

/** The pathfinder's own failure text is technical ("no path found from (21, 74) to
 * (8, 18) - goal is unreachable from start") - grid cell coordinates a user never asked
 * about. Detect that shape and show which object instead; the raw text still goes to
 * the console for debugging, never to the chat. */
/** A failure that is about how the SERVER is configured, not about the command that hit
 * it: today `vlm_client` raises `VLMError("OPENROUTER_API_KEY is not configured")` for
 * every typed command when no key is set. Re-typing the same words cannot fix it and
 * neither can naming a different object, so it is not a result belonging to that command
 * the way "no path found" or "no such object" are.
 *
 * It matters because the panel replays `listCommands` on mount: without this, one
 * unconfigured deployment leaves a permanent red box in the COMMANDS panel of every
 * later session, for commands nobody in that session issued. Demo take 2's shots 4, 5
 * and 6 all carried exactly that box (2026-09-09). A live attempt still shows it - that
 * is the moment the message is actionable - but a replayed one is dropped. */
export function isProviderConfigError(text: string | null | undefined): boolean {
  return !!text && /\b[A-Z0-9_]*API_KEY\b[^.]*\bis not configured\b/.test(text)
}

/** The same question, asked of a whole command row: the backend's own `error_code` when
 * it has one, falling back to the text pattern above for rows written before that field
 * existed. The code is authoritative - the message names whichever credential is
 * missing (OPENROUTER_API_KEY, GCP_PROJECT_ID, ADC) and a pattern cannot keep up. */
export function isChatLlmNotConnected(command: CommandResponse | null | undefined): boolean {
  if (!command || command.status !== 'failed') return false
  return command.error_code === 'chat_llm_not_connected' || isProviderConfigError(command.error_message)
}

function isCellCoordinateError(text: string): boolean {
  return /\(-?\d+,\s*-?\d+\)/.test(text)
}

/** Loose "does this name refer to a real object" check - exact, case-insensitive, or
 * substring in either direction, same tie-break family as the backend's own
 * resolve_object (app/services/command_service.py), just without its num_views/
 * num_points ranking (this only needs yes/no, never which one). */
function mentionsKnownObject(text: string, objects: readonly SceneObjectResponse[]): boolean {
  const lower = text.toLowerCase()
  return objects.some((o) => {
    if (o.is_fragment) return false
    const name = o.name.toLowerCase()
    return lower === name || lower.includes(name) || name.includes(lower)
  })
}

/** `status: "failed"` covers two very different backend shapes, and the VLM prompt
 * (app/services/vlm_client.py's build_command_messages) is explicitly instructed to
 * take the FIRST one for anything not in this scene's object list - so it's the common
 * case, not the exception:
 *   1. The VLM itself declares `{"error": "..."}` before ever resolving an object - no
 *      `parsed_action.target` at all, just its own free-text explanation as
 *      `error_message`. This is what "go to the sink" actually produces today, not the
 *      404/ObjectNotFoundError path below.
 *   2. `resolve_object` (or the pathfinder) failed on a target the VLM DID resolve.
 * Either way, if nothing in the user's own text names a real object in this scene, this
 * reads as the same "target not found" case the 404 handler already covers - so it gets
 * the same actionable message (spec: list the real alternatives) instead of surfacing
 * whatever prose that particular model happened to write. */
function describeCommandFailure(command: CommandResponse, objects: readonly SceneObjectResponse[]): string {
  const raw = command.error_message
  if (raw && isCellCoordinateError(raw)) {
    console.debug('command %s failed (technical): %s', command.command_id, raw)
    const target = typeof command.parsed_action?.target === 'string' ? command.parsed_action.target : undefined
    return STRINGS.cantReach(target)
  }
  const target = typeof command.parsed_action?.target === 'string' ? command.parsed_action.target : undefined
  if (target ? !mentionsKnownObject(target, objects) : !mentionsKnownObject(command.user_text, objects)) {
    return STRINGS.notFound(target, objects)
  }
  return raw ?? STRINGS.commandFailed
}

/** One chat bubble. Five kinds of message were repeating the same six classes, and not one
 * of them wrapped: no `whitespace-pre-wrap`, so a multi-line reply or a multi-line
 * `error_message` collapsed into one run-on line, and no `break-words`, so a single
 * unbreakable token - a UUID, a URL, `no_free_cell_within_2m` - ran straight out of the
 * bubble and past the panel. The second one is also a hard failure in
 * demo/probe_visual.py, whose `truncated` check is `scrollWidth - clientWidth > 1`.
 *
 * `min-w-0` is what lets `break-words` actually apply inside the flex column: a flex
 * item's default `min-width: auto` refuses to shrink below its own longest word. */
export function bubbleClassName(
  side: 'me' | 'them' = 'them',
  tone: 'plain' | 'error' = 'plain',
  className?: string,
): string {
  const corner = side === 'me' ? 'ml-auto rounded-br-sm' : 'rounded-bl-sm'
  const skin =
    side === 'me'
      ? 'bg-surface-raised text-fg'
      : tone === 'error'
        ? 'border-hair border-data-unreachable/30 bg-data-unreachable/10 text-data-unreachable'
        : 'border-hair border-border bg-surface text-subtle'
  return [
    'min-w-0 max-w-[85%] whitespace-pre-wrap break-words rounded-lg px-3 py-1.5 text-sm',
    corner,
    skin,
    className,
  ]
    .filter(Boolean)
    .join(' ')
}

function Bubble({
  side = 'them',
  tone = 'plain',
  className,
  children,
  ...rest
}: {
  side?: 'me' | 'them'
  tone?: 'plain' | 'error'
  className?: string
  children: React.ReactNode
} & React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={bubbleClassName(side, tone, className)} {...rest}>
      {children}
    </div>
  )
}

const ChatPanel = forwardRef<ChatPanelHandle, {
  sceneId: string
  onCommandResult: (command: CommandResponse) => void
  /** This scene's objects - the autocomplete vocabulary, the templated grammar's object
   * vocabulary, and the real alternatives a "target not found" error offers (see
   * STRINGS.notFound). */
  objects: readonly SceneObjectResponse[]
  /** Every registered platform, for the "can <robot> reach <object>" vocabulary - the
   * question is worth asking about platforms the picker does not offer. */
  platforms: RobotPlatformResponse[]
  /** Answers "can <robot> reach <object>" without a model: ScenePage asks
   * /reachability at that platform's own radius and turns the result into a sentence.
   * Returns the sentence to show. */
  onCanReach: (robotName: string, objectName: string) => Promise<string>
  /** "compare all" - opens the platform x object matrix. */
  onCompareAll: () => void
  /** "move <object> <dx> <dy>" - applies the approximate bbox cut/paste and re-asks
   * reachability with it. Returns the sentence to show. */
  onMoveObject: (objectName: string, dx: number, dy: number) => Promise<string>
  /** The selected platform's id, sent with every command alongside `radius`. The grid a
   * route is planned on is derived at this platform's own height, so without it the
   * command panel would plan on the default robot's map while the object list beside it
   * reads this one's - two lengths for one route. */
  robotId?: string | null
  /** Current radius-slider value, sent with every command so what actually gets
   * planned matches what the map/object list showed as reachable. */
  radius?: number
  /** The robot's current on-screen position, sent as `from` with every command so
   * planning continues from there instead of teleporting back to the scene's fixed
   * start on every command. */
  robotPosition?: { x: number; z: number } | null
  /** Whether an auto tour is currently in progress - disables manual input so a typed
   * or double-clicked command can't race the tour's own sequential goto calls. */
  tourRunning: boolean
  /** True once reachability/the robot's position are known and at least one reachable
   * object exists to visit - see ScenePage. */
  tourDisabled: boolean
  onToggleTour: () => void
  /** Returns the robot to the selected platform's start. Lives in this panel's bottom
   * bar rather than in the ROBOT panel: per the layout spec the bottom bar carries every
   * action that acts on the robot right now - send, tour, reset - in one place. */
  onResetRobot: () => void
  /** The most recent command's move total ("Move · 6.0s · 2.98 m"), or null when no
   * route has been planned since the last reset. Rendered above the input. */
  pathStatus: string | null
  /** The object the robot is driving to RIGHT NOW, or null when it is standing still.
   * Takes the status line over `pathStatus` while it is set: what the robot is doing is
   * more urgent than what the last route measured, and the two share one line so the
   * bottom bar's height never changes. */
  drivingTo: string | null
  /** The opening line, built from this scene's own numbers by ScenePage (see
   * `openingMessage`). A string, not a fetch: there is no model behind it and no delay in
   * front of it, so it is on screen the moment the scene's data is. */
  opening: string
}>(function ChatPanel({ sceneId, onCommandResult, objects, platforms, onCanReach, onCompareAll, onMoveObject, radius, robotId, robotPosition, tourRunning, tourDisabled, onToggleTour, onResetRobot, pathStatus, drivingTo, opening }, ref) {
  const [entries, setEntries] = useState<Entry[]>([])
  const [input, setInput] = useState('')
  // Object-name completion for the input (see lib/objectNameComplete.ts for the rules).
  // `dismissed` is per-token: Escape closes the popup for the word being typed, and the
  // next keystroke that changes the token opens it again - otherwise Escape would mute
  // completion for the rest of the session.
  const [completionIndex, setCompletionIndex] = useState(-1)
  const [completionDismissed, setCompletionDismissed] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const [sending, setSending] = useState(false)
  const [networkError, setNetworkError] = useState<string | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)

  /** Shared by typed submissions and double-click "goto"s: posts the command, appends
   * the history entry, and handles the two failure shapes (unresolvable object vs. a
   * network/server error) the same way either time. `historyLabel` is what shows in the
   * chat bubble if the object turns out not to exist (a real user_text only comes back
   * with a successful response).
   *
   * A pending placeholder entry (showing the typing dots - see AgentTyping) is
   * appended immediately, before the request even goes out, and replaced in place once
   * the response (or error) comes back - so the parsed-command chip always renders
   * before whatever result follows it, never alongside or after. */
  async function send(opts: { text?: string; targetObjectId?: string }, historyLabel: string) {
    if (sending) return
    setSending(true)
    setNetworkError(null)
    const tempId = `pending-${Date.now()}`
    setEntries((prev) => [...prev, { id: tempId, userText: historyLabel, command: null, pending: true }])

    try {
      const from: [number, number] | undefined = robotPosition ? [robotPosition.x, robotPosition.z] : undefined
      const command = await postCommand(sceneId, { ...opts, radius, robotId, from })
      setEntries((prev) =>
        prev.map((e) => (e.id === tempId ? { id: command.command_id, userText: command.user_text, command } : e)),
      )
      onCommandResult(command)
    } catch (err) {
      if (err instanceof NotFoundError) {
        const detail = (err.body as { detail?: string } | null)?.detail
        setEntries((prev) =>
          prev.map((e) =>
            e.id === tempId
              ? {
                  id: tempId,
                  userText: historyLabel,
                  command: null,
                  notFoundMessage: STRINGS.notFound(extractMissingName(detail), objects),
                }
              : e,
          ),
        )
      } else if (err instanceof ApiError) {
        setNetworkError(STRINGS.networkError)
        setEntries((prev) => prev.filter((e) => e.id !== tempId))
      } else {
        throw err
      }
    } finally {
      setSending(false)
    }
  }

  useImperativeHandle(ref, () => ({
    goto: (objectId: string, name: string) => send({ targetObjectId: objectId }, `goto ${name}`),
  }))

  useEffect(() => {
    let cancelled = false
    listCommands(sceneId, 0, 200).then((res) => {
      if (cancelled) return
      setEntries(
        res.items
          // See isChatLlmNotConnected: a past deployment's missing credentials are not
          // this session's conversation, and replaying that pins a red box in the panel.
          .filter((c) => !isChatLlmNotConnected(c))
          .map((c) => ({ id: c.command_id, userText: c.user_text, command: c })),
      )
    })
    return () => {
      cancelled = true
    }
  }, [sceneId])

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight })
  }, [entries])

  /** Append an answer this panel produced itself - no request, no model. */
  function appendLocal(userText: string, reply: string, approximate = false) {
    setEntries((prev) => [
      ...prev,
      { id: `local-${Date.now()}-${prev.length}`, userText, command: null, localReply: { text: reply, approximate } },
    ])
  }

  async function submit(e: React.FormEvent) {
    e.preventDefault()
    const text = input.trim()
    if (!text || sending || tourRunning) return
    setInput('')

    // Templated first, and locally: these four are the commands this screen is FOR, and
    // they must keep working on a deployment with no model credentials - which is every
    // deployment in this worktree, by design. Anything else is a question only a model
    // can answer, and goes to the one the workspace's job spec names.
    const parsed = parseCommand(text, {
      objects: objects.filter((o) => !o.is_fragment).map((o) => o.name),
      robots: platforms.map((p) => p.display_name),
    })

    if (parsed.kind === 'goto') {
      const target = resolveObjectByName(objects, parsed.object)
      // A name the grammar matched always resolves; the fallback exists so a vocabulary
      // that has drifted from the object list degrades to the model rather than to
      // nothing at all.
      if (target) {
        await send({ targetObjectId: target.id }, text)
        return
      }
    } else if (parsed.kind === 'canReach') {
      setSending(true)
      try {
        appendLocal(text, await onCanReach(parsed.robot, parsed.object))
      } finally {
        setSending(false)
      }
      return
    } else if (parsed.kind === 'compareAll') {
      onCompareAll()
      appendLocal(text, 'Showing every platform against every object.')
      return
    } else if (parsed.kind === 'move') {
      setSending(true)
      try {
        appendLocal(text, await onMoveObject(parsed.object, parsed.dx, parsed.dy), true)
      } finally {
        setSending(false)
      }
      return
    }

    await send({ text }, text)
  }

  const suggestions = completionDismissed ? [] : suggestObjectNames(input, completionNames(objects))
  const activeSuggestion = completionIndex >= 0 ? suggestions[completionIndex] : undefined

  function acceptCompletion(name: string) {
    setInput(applyCompletion(input, name))
    setCompletionIndex(-1)
    setCompletionDismissed(false)
    inputRef.current?.focus()
  }

  function onInputKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (suggestions.length === 0) return
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setCompletionIndex((i) => wrapIndex(i + 1, suggestions.length))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setCompletionIndex((i) => wrapIndex(i - 1, suggestions.length))
    } else if (e.key === 'Escape') {
      e.preventDefault()
      setCompletionDismissed(true)
      setCompletionIndex(-1)
    } else if (e.key === 'Tab') {
      // Tab always completes - the top suggestion when nothing is highlighted, which is
      // what every shell does and what a typist reaches for first.
      e.preventDefault()
      acceptCompletion(activeSuggestion ?? suggestions[0])
    } else if (e.key === 'Enter' && activeSuggestion) {
      // Only steals Enter when a suggestion is actually HIGHLIGHTED. With the popup open
      // but nothing chosen, Enter still sends - typing a full command and pressing Enter
      // must never be intercepted by a list the user was ignoring.
      e.preventDefault()
      acceptCompletion(activeSuggestion)
    }
  }

  return (
    <div className="flex h-full flex-col" data-testid="commands-panel">
      {/* AGENT, not COMMANDS: this is the card's own header, and what is under it is a
         conversation rather than a log of issued commands. */}
      <PanelSection title="Agent" className="shrink-0" />

      {/* min-h-0 is load-bearing: without it, a flex-1 child's default min-height:auto
         stops it shrinking below its content's natural height, so under a tight sidebar
         (e.g. platform/robot panels above eating most of the height) this would overflow
         its flex slot instead of scrolling - pushing the form below it half off-panel
         rather than leaving it its full padded size (see the form's own shrink-0). */}
      {/* An empty history is CENTRED, not stacked at the top. On a scene with no commands
         yet the hint used to sit against the header with the rest of the panel empty
         below it - measured at 55% of the panel's height in one unbroken run
         (demo/probe_visual.py). Centring an empty state is what makes it read as "nothing
         here yet" rather than as content that stopped. */}
      <div
        ref={scrollRef}
        className={`flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto px-s3 py-s3 ${
          entries.length === 0 ? 'justify-center' : ''
        }`}
        data-testid="commands-history"
      >
        {/* THE OPENING MESSAGE, from this scene's own numbers. Templated - no model call
            behind it and no artificial delay in front of it, so it is on screen the moment
            the scene's data is. It says what the screen already knows and the reader does
            not yet: how big the room is, how much is in it, and whether the selected robot
            can get around it.
            
            It is the FIRST LINE OF THE CONVERSATION, not an empty-state placeholder: a
            scene with replayed history had one command panel and a scene without had a
            different one, and the sentence that orients you is worth more on the scene you
            have already worked in, not less. The templated-forms hint below it is the part
            that is only useful before you have typed anything, so that part still goes. */}
        <div className="flex flex-col gap-3 text-bodyux text-muted">
          <p data-testid="chat-opening" className="text-fg">
            {opening}
          </p>
          {entries.length === 0 && (
            <div className="flex flex-col gap-s1">
              <p>These work without any model configured:</p>
              <ul className="flex flex-col gap-0.5 font-mono text-xs">
                {TEMPLATED_FORMS.map((form) => (
                  <li key={form}>{form}</li>
                ))}
              </ul>
              <p>Anything else goes to this workspace's chat LLM.</p>
            </div>
          )}
        </div>
        {/* 12px between replies and 12px inside one: the reply is a unit, and a tighter
            inner gap made the user's line and the answer read as one paragraph. */}
        <div className="flex flex-col gap-3">
          {entries.map((entry) => (
            <div key={entry.id} className="flex flex-col gap-3">
              <Bubble side="me">{entry.userText}</Bubble>

              {/* The typing indicator while the request is in flight, and nothing once it
                 lands. The parsed {action, target} chip that used to sit here echoed the
                 request back - "GOTO → CABINET" above the user's own "goto cabinet" - and
                 then the card below said what actually happened. The card is the answer. */}
              {entry.pending && <AgentTyping />}

              {!entry.pending && entry.command === null && entry.notFoundMessage && (
                <Bubble tone="error">{entry.notFoundMessage}</Bubble>
              )}

              {entry.localReply && (
                <Bubble data-testid="local-reply">
                  {entry.localReply.text}
                  {entry.localReply.approximate && (
                    <Badge variant="neutral" className="ml-2 align-middle" data-testid="approximate-badge">
                      approximate
                    </Badge>
                  )}
                </Bubble>
              )}

              {entry.command?.status === 'failed' && (
                <Bubble
                  tone="error"
                  data-testid={isChatLlmNotConnected(entry.command) ? 'chat-llm-not-connected' : 'command-failed'}
                >
                  {/* Not "not understood": this one is about the deployment, and the
                      templated commands are unaffected. */}
                  {isChatLlmNotConnected(entry.command)
                    ? STRINGS.chatLlmNotConnected
                    : describeCommandFailure(entry.command, objects)}
                </Bubble>
              )}

              {/* ONE CARD PER REPLY. What happened, and underneath it how long and how
                  far - once. This was a chip, a per-step line and a "Total:" line, which
                  on the single-step command a goto plans is the same number printed twice
                  under a restatement of the request. The multi-step case (pick, place)
                  keeps its steps behind the title, where they add something. */}
              {entry.command?.status === 'done' && (
                <Bubble>
                  <div className="flex min-w-0 flex-col gap-1">
                    <p className="text-bodyux text-fg" data-testid="reply-title">
                      {describeOutcome(entry.command, objects)}
                    </p>
                    {describeOutcomeDetail(entry.command) && (
                      <p className="font-mono text-[11px] text-faint" data-testid="reply-detail">
                        {describeOutcomeDetail(entry.command)}
                      </p>
                    )}
                    {entry.command.steps.length > 1 && (
                      <div className="mt-0.5 flex flex-col gap-0.5">
                        {entry.command.steps.map((step, i) => (
                          <p key={i} className="font-mono text-xs text-muted">
                            {describeStep(step)}
                          </p>
                        ))}
                      </div>
                    )}
                    {/* Which model produced this, when one did. A templated command and
                        a direct goto say nothing here, because nothing was asked. */}
                    {entry.command.provider_used && (
                      <p className="font-mono text-[11px] text-faint" data-provider-used={entry.command.provider_used}>
                        via {entry.command.provider_used}
                      </p>
                    )}
                  </div>
                </Bubble>
              )}
            </div>
          ))}
        </div>
      </div>

      {networkError && <p className="shrink-0 px-3 pt-1 text-xs text-data-unreachable">{networkError}</p>}

      {/* shrink-0: the input row must never be squeezed to make room for history above
         it - it always keeps its full padded size, and the history area (min-h-0
         flex-1 above) is what shrinks/scrolls instead when the panel is short. */}
      <form onSubmit={submit} className="flex shrink-0 flex-col gap-s2 border-t-hair border-border p-s3" data-testid="agent-bottom-bar">
        {(drivingTo || pathStatus) && (
          <p className="font-mono text-caption text-muted" data-testid="path-status">
            {drivingTo ? (
              <span data-testid="driving-status" className="text-fg">
                Driving to {drivingTo}…
              </span>
            ) : (
              pathStatus
            )}
          </p>
        )}
        <div className="relative">
          {suggestions.length > 0 && (
            <ul
              data-testid="command-suggestions"
              className="absolute bottom-full left-0 z-20 mb-1 w-full overflow-hidden rounded-md border-hair border-border bg-surface-raised shadow-xl"
            >
              {suggestions.map((name, i) => (
                <li key={name}>
                  <button
                    type="button"
                    data-suggestion={name}
                    // onMouseDown, not onClick: the input's blur would otherwise fire
                    // first and the click would land on nothing.
                    onMouseDown={(ev) => {
                      ev.preventDefault()
                      acceptCompletion(name)
                    }}
                    onMouseEnter={() => setCompletionIndex(i)}
                    className={`block w-full px-2.5 py-1 text-left text-sm ${
                      i === completionIndex ? 'bg-accent-dim text-fg' : 'text-subtle hover:bg-fg/5'
                    }`}
                  >
                    {name}
                  </button>
                </li>
              ))}
            </ul>
          )}
          <Input
            ref={inputRef}
            value={input}
            onChange={(e) => {
              setInput(e.target.value)
              setCompletionIndex(-1)
              setCompletionDismissed(false)
            }}
            onKeyDown={onInputKeyDown}
            placeholder="Command the robot"
            disabled={tourRunning}
            className="w-full"
            autoComplete="off"
            data-testid="command-input"
          />
        </div>
        <div className="flex gap-2">
          <Button
            variant="secondary"
            type="submit"
            disabled={tourRunning || !input.trim()}
            loading={sending}
            className="flex-1"
            data-testid="send-command"
          >
            Send
          </Button>
          <Button
            variant="secondary"
            onClick={onToggleTour}
            disabled={!tourRunning && tourDisabled}
            title={tourRunning ? 'Stop the auto tour' : 'Visit every reachable object automatically'}
            className={tourRunning ? 'text-data-unreachable' : ''}
            data-testid="auto-tour"
          >
            {tourRunning ? 'Stop' : 'Auto tour'}
          </Button>
        </div>
        <Button variant="ghost" onClick={onResetRobot} className="w-full" data-testid="reset-robot">
          Reset robot
        </Button>
      </form>
    </div>
  )
})

export default ChatPanel
