import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link, useParams, useSearchParams } from 'react-router-dom'
import { NotFoundError } from '../api/client'
import { updateProject } from '../api/projects'
import {
  getCameraTrack,
  getFitProb,
  getReachability,
  getSceneLayers,
  getScene,
  getSceneMap,
  sceneCloudUrl,
  sceneMeshUrl,
  sceneMsaGlbUrl,
  sceneMsaMetaUrl,
} from '../api/scenes'
import { getRobots } from '../api/robots'
import { getVideo } from '../api/videos'
import type {
  CameraPoseResponse,
  CommandResponse,
  FitProbResponse,
  SceneLayersResponse,
  OccupancyGridResponse,
  ReachabilityResponse,
  RobotPlatformResponse,
  SceneObjectResponse,
  SceneResponse,
  VideoResponse,
} from '../api/types'
import ChatPanel, { totalPathLengthM, type ChatPanelHandle } from '../components/ChatPanel'
import CriticBadge from '../components/CriticBadge'
import JobTimeline from '../components/JobTimeline'
import MapViewToggle, { type MapViewMode } from '../components/MapViewToggle'
import ObjectList from '../components/ObjectList'
import PlatformComparison from '../components/PlatformComparison'
import RobotPanel from '../components/RobotPanel'
import SceneToolbar from '../components/SceneToolbar'
import SceneVerdict from '../components/SceneVerdict'
import SceneView from '../components/SceneView'
import StartPositionCaption from '../components/StartPositionCaption'
import VideoPlayer from '../components/VideoPlayer'
import WorkspaceSwitcher from '../components/WorkspaceSwitcher'
import Badge from '../components/ui/Badge'
import Button, { buttonClassName } from '../components/ui/Button'
import PanelSection from '../components/ui/PanelSection'
import { findNearestPoseIndex } from '../lib/cameraTrack'
import { formatMetres, formatSeconds, formatSigned } from '../lib/formatMeasure'
import { openingMessage } from '../lib/openingMessage'
import { groupObjects, visibleObjects } from '../lib/objectGroups'
import { applyMovesToGrid, applyMovesToObjects, moveParams, type ObjectMove } from '../lib/objectMoves'
import { useAutoTour } from '../lib/useAutoTour'
import { useCommandAnimation } from '../lib/useCommandAnimation'
import { useMediaQuery } from '../lib/useMediaQuery'
import { loadRenderMode, saveRenderMode, type RenderMode } from '../lib/renderMode'
import { resolveSceneLayers, type PointsSource } from '../lib/layerAvailability'
import {
  DEFAULT_SCENE_LAYERS,
  type SceneLayerKey,
  type SceneLayerState,
} from '../lib/sceneLayers'
import { vocabSourceLabel } from '../lib/vocabSourceLabel'
import { useCaptureGuide } from '../state/capture-guide-context'
import { useProjects } from '../state/projects-context'

/** One frozen empty array, so the "scene has not loaded yet" branch of `shownObjects` is
 * not itself a fresh identity on every render - the whole point of memoising it. */
const EMPTY_OBJECTS: SceneObjectResponse[] = []
/** Same reason, for the two per-object maps the object list and the map both read. `?? {}`
 * inline hands every consumer a brand-new object on every render, and this component
 * re-renders on every animation frame of a move. */
const EMPTY_REASONS: Record<string, string> = {}
const EMPTY_LENGTHS: Record<string, number> = {}

const POLL_MS = 5000

export default function ScenePage() {
  // `workspaceId`, not `projectId`: the route is /workspaces/:workspaceId/scenes/:sceneId.
  // useParams returns a Partial, so a stale name here would be `undefined` with a CLEAN
  // tsc and no runtime error - it would just silently stop the primary_scene_id auto-set
  // below (killing every gallery preview), stop `project` resolving, and pin the workspace
  // dropdown on its placeholder. Every route in this app uses :workspaceId.
  const { sceneId, workspaceId } = useParams<{ sceneId: string; workspaceId: string }>()
  const { projects, refresh: refreshProjects } = useProjects()
  const { openGuide } = useCaptureGuide()
  const [scene, setScene] = useState<SceneResponse | null>(null)
  const [grid, setGrid] = useState<OccupancyGridResponse | null>(null)
  const [video, setVideo] = useState<VideoResponse | null>(null)
  const [cameraTrack, setCameraTrack] = useState<CameraPoseResponse[] | null>(null)
  // "camera path" toggle - shared by the 2D and 2D views (see the header checkbox
  // below), off by default and never persisted: it's an occasional inspection tool, not
  // something that should be on by default every visit (per spec).
  const [showCameraPath, setShowCameraPath] = useState(false)
  // The video player's current playback position, seconds - null until the video's own
  // `timeupdate` has fired at least once. Drives the camera-path highlight below.
  const [videoTimeSec, setVideoTimeSec] = useState<number | null>(null)
  const [notFound, setNotFound] = useState(false)
  const [selectedObjectId, setSelectedObjectId] = useState<string | null>(null)
  const [hoveredObjectId, setHoveredObjectId] = useState<string | null>(null)
  // A click on an Objects-list row asks the 3D view to frame its camera on that object
  // (see SceneMap3D's focusObjectId/focusNonce props) - `nonce` is bumped on every
  // request (even a re-click of the same object) so the effect that drives the tween
  // can tell "asked again" from "nothing changed", which the object id alone can't.
  const [focusRequest, setFocusRequest] = useState<{ id: string; nonce: number } | null>(null)
  const focusNonceRef = useRef(0)
  const handleFocusObject = useCallback((id: string) => {
    focusNonceRef.current += 1
    setFocusRequest({ id, nonce: focusNonceRef.current })
  }, [])
  const [currentPath, setCurrentPath] = useState<[number, number][] | null>(null)
  // Every point the robot has actually travelled, accumulated across commands - unlike
  // currentPath above (the latest command's own planned route, replaced each time),
  // this only ever grows. Cleared solely by "Reset robot to start" (handleResetRobot)
  // or a genuine scene switch (see the sceneId effect below).
  const [trail, setTrail] = useState<[number, number][]>([])
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const chatRef = useRef<ChatPanelHandle>(null)

  // View mode lives in the URL (?view=3d) so it's shareable via link, per spec.
  const [searchParams, setSearchParams] = useSearchParams()
  const viewMode: MapViewMode = searchParams.get('view') === '3d' ? '3d' : '2d'
  const setViewMode = useCallback(
    (mode: MapViewMode) => {
      setSearchParams(
        (prev) => {
          const next = new URLSearchParams(prev)
          if (mode === '3d') next.set('view', '3d')
          else next.delete('view')
          return next
        },
        { replace: true },
      )
    },
    [setSearchParams],
  )

  // Spec 9 hotkeys: "2"/"3" switch view. Ignored while typing into an input/textarea
  // (the command box) so "2"/"3" can still be typed as ordinary text there.
  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      const target = e.target as HTMLElement | null
      if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA')) return
      if (e.key === '2') setViewMode('2d')
      else if (e.key === '3') setViewMode('3d')
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [setViewMode])

  const isMid = useMediaQuery('(max-width: 1399px)')

  // Robot platform registry (GET /api/robots) - static per deployment, fetched once and
  // held here rather than per-scene. EVERY registered platform is offered, in the
  // picker and in "Compare all" alike: which robots a customer owns is not something
  // this screen gets to decide, and a robot that is missing from the picker cannot be
  // asked the question the screen exists to answer.
  const [robots, setRobots] = useState<RobotPlatformResponse[]>([])
  const [selectedPlatformId, setSelectedPlatformId] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    getRobots().then((res) => {
      if (cancelled) return
      setRobots(res.items)
      setSelectedPlatformId((cur) => cur ?? res.default_robot_id)
    })
    return () => {
      cancelled = true
    }
  }, [])

  const selectedPlatform = robots.find((p) => p.id === selectedPlatformId) ?? null

  // The radius comes from the selected platform's registry entry and from nowhere else.
  // There used to be a clearance slider that overrode it, which meant the screen's
  // headline could be about a 0.37 m robot that does not exist; the picker is the only
  // way to change it now. null until the registry response arrives, in which case the
  // reachability request below omits `radius` and takes the server's own default.
  const robotRadius = selectedPlatform?.radius_m ?? null

  const [reachability, setReachability] = useState<ReachabilityResponse | null>(null)
  // The Monte-Carlo fit-probability audit (scripts.audit.fit_prob), if one has been run
  // for this scene. Exactly ONE number is read out of it now - `corridor_width_p5_m`,
  // the verdict's "tightest gap". The "Fits: N/100" headline it used to drive is gone:
  // on this screen "fits" means the robot has a start position, which is a fact about
  // the grid, not a percentage (see StartPositionPanel).
  const [fitProb, setFitProb] = useState<FitProbResponse | null>(null)
  // This scene's nvblox layer pack, when one is sampled on its grid - the ESDF slice and
  // the unobserved mask the "Occupancy" render pass draws over the band mask. null for a
  // scene with no pack (a 404, and the common case), which the map degrades to drawing
  // the occupancy grid alone.
  const [layers, setLayers] = useState<SceneLayersResponse | null>(null)
  const [unreachableNotice, setUnreachableNotice] = useState<string | null>(null)
  // The most recent command, for the bottom bar's path status line. Cleared by Reset
  // robot, since the route it described no longer exists after a reset.
  const [lastCommand, setLastCommand] = useState<CommandResponse | null>(null)
  const noticeTimer = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Fragments are hidden by default and the toggle lives on the toolbar, because it
  // decides what BOTH the object list and the platform comparison enumerate.
  const [showFragments, setShowFragments] = useState(false)
  // "Compare all" swaps the object list for the platform x object matrix in the same
  // column, rather than opening a third place to look.
  const [compareAll, setCompareAll] = useState(false)
  // THE FOUR LAYERS the 3D view composes - Map, Points, Mesh, Layout (lib/sceneLayers.ts).
  // Map + Points on by default: they are the two every scene has, and the two the screen's
  // question is about. Mesh and Layout are large files and are not fetched until asked for.
  const [sceneLayers, setSceneLayers] = useState<SceneLayerState>(DEFAULT_SCENE_LAYERS)
  const handleToggleLayer = useCallback((key: SceneLayerKey, value: boolean) => {
    setSceneLayers((prev) => ({ ...prev, [key]: value }))
  }, [])
  // Which of them this scene HAS. Map and Points are derived from data every done scene
  // carries (the grid, and /cloud with its documented fallback to /mesh), so only the two
  // optional exports are probed - one Range request each, see probeLayerAvailable.
  const [layersAvailable, setLayersAvailable] = useState<SceneLayerState>({
    map: true,
    points: false,
    mesh: false,
    layout: false,
  })
  // WHICH FILE the Points layer reads. /cloud where the scene has one; /mesh only where
  // /mesh is a points primitive. 'none' disables the Points toggle rather than leaving it
  // live over nothing.
  const [pointsSource, setPointsSource] = useState<PointsSource>('none')

  // "What if that were somewhere else": the moves the agent's `move <object> <dx> <dy>`
  // command has applied. Sent with every reachability request, so the verdict, the rows
  // and the map's reachable area all answer the MOVED room; see lib/objectMoves.ts and
  // the backend's `move` parameter. Cleared by "reset" on the toolbar.
  const [moves, setMoves] = useState<ObjectMove[]>([])
  // Stable per moves-change, so the reachability effect below re-runs when a move is
  // applied and not on every render.
  const moveQuery = useMemo(() => moveParams(moves), [moves])

  // Only trust `reachability` once its radius matches the selected platform's -
  // otherwise the previous platform's result would flash while the new fetch is in
  // flight, and the verdict would briefly claim the wrong robot's number.
  const currentReachability =
    reachability && reachability.radius === (robotRadius ?? reachability.radius) ? reachability : null
  // Memoised for the same reason as `shownObjects` below: this is a dep of the object-box
  // effect, and a new Set per render is a new identity per animation frame.
  const reachableObjectIds = useMemo(
    () => (currentReachability ? new Set(currentReachability.reachable_object_ids) : null),
    [currentReachability],
  )
  // Whether this scene has a navigation layer at all. Both endpoints answer 200 either
  // way and say so in the body, so there is no error branch here - just two shapes. The
  // reachability answer is the one that decides, because it is what the verdict, the
  // route and the goto all come from; `grid` is checked too, so the canvas cannot draw a
  // stale picture while the bar says there is nothing to draw from.
  const layerAvailable = (currentReachability?.layer_available ?? true) && (grid?.layer_available ?? true)
  // Where THIS platform actually starts, from the reachability response (see its
  // start_status doc). null when the platform fits nowhere in the scene, or before the
  // first response for the current radius has arrived.
  const resolvedStart =
    currentReachability && currentReachability.start_x != null && currentReachability.start_z != null
      ? { x: currentReachability.start_x, z: currentReachability.start_z }
      : null

  const handleSelectPlatform = useCallback((id: string) => {
    setSelectedPlatformId(id)
  }, [])

  const robotMeshUrl = selectedPlatform ? `/${selectedPlatform.mesh_path}` : undefined
  const robotHeightM = selectedPlatform?.dimensions_m?.height_m ?? undefined
  // Rectangle footprint indicator (SceneMap3D's "Footprint indicator" effect) - the
  // selected platform's real L x W, same source as robotHeightM above. undefined (not
  // the radius) when no platform is selected or its dimensions aren't measured yet -
  // SceneMap3D falls back to a radius-derived square itself in that case, same
  // "always show a usable answer" convention app.robots.resolve_footprint_m uses on
  // the backend.
  const robotLengthM = selectedPlatform?.dimensions_m?.length_m ?? undefined
  const robotWidthM = selectedPlatform?.dimensions_m?.width_m ?? undefined

  useEffect(() => {
    if (!sceneId) return
    let cancelled = false

    async function poll() {
      try {
        const res = await getScene(sceneId!)
        if (cancelled) return
        setScene(res)
        if (res.status === 'queued' || res.status === 'processing') {
          timer.current = setTimeout(poll, POLL_MS)
        }
        // The map is NOT fetched here any more - it depends on the selected platform, and
        // this effect keys on the scene alone. See the effect below.
      } catch (err) {
        if (cancelled) return
        if (err instanceof NotFoundError) setNotFound(true)
        else throw err
      }
    }

    void poll()
    return () => {
      cancelled = true
      if (timer.current) clearTimeout(timer.current)
    }
  }, [sceneId])

  // The 2D map, RE-FETCHED WHEN THE PLATFORM CHANGES. The grid is derived from the
  // layer's obstacle-height map at the selected robot's own roof, so it is not one array
  // per scene any more: a 0.192 m TurtleBot drives under a 0.694 m bed top that walls off
  // a 0.40 m Go2, and those are different pictures of the same room. Fetching it once on
  // load would leave a user planning one robot's routes while looking at another's map -
  // the same "two answers to one question" the single-layer change removed everywhere
  // else. Waits for the platform: with `robot_id` unset the server answers with the
  // DEFAULT platform's map, which is a third answer nobody asked for.
  useEffect(() => {
    if (scene?.status !== 'done' || !selectedPlatformId) return
    let cancelled = false
    getSceneMap(scene.id, selectedPlatformId)
      .then((g) => {
        if (!cancelled) setGrid(g)
      })
      .catch((err) => {
        // 200 either way now - a scene with no layer answers `layer_available: false`, so
        // reaching here means a real transport failure, not a missing layer.
        console.error('Failed to fetch the scene map', err)
      })
    return () => {
      cancelled = true
    }
  }, [scene?.id, scene?.status, selectedPlatformId])

  useEffect(() => {
    if (!scene) return
    let cancelled = false
    getVideo(scene.video_id).then((v) => {
      if (!cancelled) setVideo(v)
    })
    return () => {
      cancelled = true
    }
  }, [scene?.video_id])

  // The canonical MSA object list used to be fetched here, purely to override the object
  // list's header total and its group badges. It is gone: `/msa-objects` uses its own id
  // namespace ("bed_0") with no mapping to the SceneObject rows this screen selects,
  // hovers and plans against, so counting one and addressing the other put two numbers
  // for one room on the same screen - "OBJECTS (15)" beside "13 of 17 reachable". One
  // count now, from `visibleObjects`. The endpoint is untouched; nothing here reads it.

  useEffect(() => {
    if (!scene || scene.status !== 'done') return
    let cancelled = false
    getCameraTrack(scene.id)
      .then((r) => {
        if (!cancelled) setCameraTrack(r.poses)
      })
      .catch((err) => {
        // Display-only overlay, off by default - not worth surfacing further than the
        // console if it fails to load.
        console.error('Failed to fetch camera track', err)
      })
    return () => {
      cancelled = true
    }
  }, [scene?.id, scene?.status])

  // Nothing sets `primary_scene_id` today (it exists on the API but every project comes
  // back with it null) - the project-preview thumbnail on HomePage (spec 11) reads a
  // project's primary scene to know which cached preview to show, so without this a
  // project can never grow a thumbnail. Opportunistic and one-shot: the first scene of
  // a project that finishes reconstruction becomes its primary scene, silently, and an
  // existing choice is never overwritten - there's no UI yet for picking one deliberately.
  useEffect(() => {
    if (!scene || scene.status !== 'done' || !workspaceId) return
    const proj = projects.find((p) => p.id === workspaceId)
    if (!proj || proj.primary_scene_id) return
    updateProject(workspaceId, { primary_scene_id: scene.id })
      .then(() => refreshProjects())
      .catch(() => {})
  }, [scene, workspaceId, projects, refreshProjects])

  // The shared render-mode selector (spec 6) - persisted per scene id like SceneMap3D's
  // ceilingMode, but owned here since both 2D and 3D read it.
  const [renderMode, setRenderMode] = useState<RenderMode>('clearance')
  useEffect(() => {
    if (scene?.id) saveRenderMode(scene.id, renderMode)
  }, [scene?.id, renderMode])

  // A trail from a different scene wouldn't mean anything against this one's point
  // cloud - reset it on a genuine scene switch (ScenePage isn't remounted on a sceneId
  // route-param change, see useCommandAnimation's own seededForRef comment). Re-seeds
  // renderMode from that scene's own saved value too.
  const trailSceneIdRef = useRef(scene?.id)
  useEffect(() => {
    if (trailSceneIdRef.current === scene?.id) return
    trailSceneIdRef.current = scene?.id
    setTrail([])
    if (scene?.id) setRenderMode(loadRenderMode(scene.id))
  }, [scene?.id])

  useEffect(() => {
    if (!scene) return
    let cancelled = false
    getReachability(scene.id, robotRadius ?? undefined, moveQuery, selectedPlatformId)
      .then((r) => {
        if (cancelled) return
        setReachability(r)
      })
      .catch((err) => {
        if (cancelled) return
        // Leaves `reachability` at its last value (or null), so the panel keeps showing
        // "Checking reachability…" for this radius rather than a stale reachable count -
        // no dedicated error state exists here. A radius this large legitimately can
        // fail server-side (e.g. no free grid
        // cell within the connectivity search's fixed radius near the robot's start
        // position) for a wide platform - logged, not surfaced further, since retrying
        // won't help until the platform changes again.
        console.error('Failed to fetch reachability', err)
      })
    return () => {
      cancelled = true
    }
  }, [scene?.id, robotRadius, moveQuery])

  // Scene-level, not radius/platform dependent (the audit covers every registered
  // platform in one file), so this fetches once per scene rather than on every platform
  // switch like reachability above.
  useEffect(() => {
    if (!scene) return
    let cancelled = false
    setFitProb(null)
    getFitProb(scene.id)
      .then((r) => {
        if (!cancelled) setFitProb(r)
      })
      .catch((err) => {
        if (cancelled) return
        // 404 (NotFoundError) is the expected/common case - most scenes have no
        // fit-probability audit on disk - so it is silently left at null, which the
        // verdict reads as "show no gap line at all".
        if (!(err instanceof NotFoundError)) console.error('Failed to fetch fit-prob', err)
      })
    return () => {
      cancelled = true
    }
  }, [scene?.id])

  useEffect(() => {
    if (!scene || scene.status !== 'done') return
    let cancelled = false
    setLayers(null)
    getSceneLayers(scene.id)
      .then((r) => {
        if (!cancelled) setLayers(r)
      })
      .catch((err) => {
        if (cancelled) return
        // A scene with no pack is the ordinary case, not a failure - see the state's own
        // comment. Anything else is worth a console line but never blocks the map.
        if (!(err instanceof NotFoundError)) console.error('Failed to fetch scene layers', err)
      })
    return () => {
      cancelled = true
    }
  }, [scene?.id, scene?.status])

  // What this scene's three optional files actually ARE, asked of the files rather than
  // of their URLs. Nothing here can stop the 3D view: every branch resolves, a rejection
  // is caught, and Map + Points are decided by the grid and /cloud, neither of which this
  // touches. A 404 on /mesh or /msa-glb leaves those two toggles disabled and the rest of
  // the screen exactly as it was.
  useEffect(() => {
    if (!scene || scene.status !== 'done') return
    const ctrl = new AbortController()
    const sceneId = scene.id
    setLayersAvailable({ map: true, points: false, mesh: false, layout: false })
    setPointsSource('none')
    resolveSceneLayers(
      {
        cloud: sceneCloudUrl(sceneId),
        mesh: sceneMeshUrl(sceneId),
        msaGlb: sceneMsaGlbUrl(sceneId),
      },
      ctrl.signal,
    )
      .then((facts) => {
        if (ctrl.signal.aborted) return
        console.info(`[layers] scene ${sceneId}: ${facts.log}`)
        setPointsSource(facts.pointsSource)
        setLayersAvailable({
          map: true,
          points: facts.pointsSource !== 'none',
          // The Mesh layer draws a surface. A points-primitive export is the Points
          // layer's file, and enabling Mesh over it would draw the same file twice.
          mesh: facts.mesh.triangles,
          layout: facts.layout,
        })
      })
      .catch((err) => {
        // resolveSceneLayers swallows its own failures, so reaching here is a bug in it
        // rather than a scene without a mesh. Logged, and the layers stay disabled - the
        // 3D view keeps rendering whatever Map and Points already have.
        if (!ctrl.signal.aborted) console.error('Failed to resolve the scene layers', err)
      })
    return () => ctrl.abort()
  }, [scene?.id, scene?.status])

  // undefined = no audit for this scene / no platform selected, so the verdict shows no
  // gap at all; null = an audit exists but had no successful trial to take a percentile
  // over, which the shared formatter renders as an em dash. See SceneVerdict.
  const tightestGapM =
    fitProb && selectedPlatformId && fitProb.platforms[selectedPlatformId]
      ? fitProb.platforms[selectedPlatformId].corridor_width_p5_m
      : undefined

  const handleUnreachableClick = useCallback(
    (name: string) => {
      if (noticeTimer.current) clearTimeout(noticeTimer.current)
      setUnreachableNotice(`The robot can't reach "${name}" at a ${formatMetres(robotRadius)} radius.`)
      noticeTimer.current = setTimeout(() => setUnreachableNotice(null), 4000)
    },
    [robotRadius],
  )

  useEffect(() => {
    return () => {
      if (noticeTimer.current) clearTimeout(noticeTimer.current)
    }
  }, [])

  const initialRobotPos =
    scene?.robot_start_x != null && scene?.robot_start_z != null
      ? { x: scene.robot_start_x, z: scene.robot_start_z }
      : null
  const { state: anim, play, reset } = useCommandAnimation(scene?.id, initialRobotPos)

  // The in-flight play() promise for the most recent command - lets the auto tour (see
  // visitTourStop below) await the robot actually *arriving*, not just the command
  // having been posted. handleCommandResult always runs synchronously inside
  // ChatPanel.goto()'s `send()` before that call's own promise resolves, so by the time
  // a `chatRef.current.goto()` await returns, this ref already points at that stop's
  // play() promise.
  const playPromiseRef = useRef<Promise<void> | null>(null)

  // Put the robot on the start that belongs to the SELECTED platform. A wider platform
  // does not fit where a narrower one stood, so this is recomputed per radius on the
  // server and re-seeded here; keyed on the coordinates themselves, so a radius change
  // that resolves to the same cell does not disturb a robot the user has already driven.
  // This also removes the trap HANDOFF 5b records: planning continues from wherever the
  // robot is, so switching to Husky used to re-plan from wherever TurtleBot had stopped
  // and produced a 0.37 m stub instead of a route.
  const seededStartRef = useRef<string | null>(null)
  useEffect(() => {
    if (!resolvedStart) return
    const key = `${resolvedStart.x.toFixed(4)},${resolvedStart.z.toFixed(4)}`
    if (seededStartRef.current === key) return
    seededStartRef.current = key
    setCurrentPath(null)
    setTrail([])
    setLastCommand(null)
    reset(resolvedStart)
  }, [resolvedStart, reset])

  const handleCommandResult = useCallback(
    (command: CommandResponse) => {
      if (!scene) return
      setLastCommand(command)
      const path = command.steps.flatMap((s) => s.path ?? [])
      setCurrentPath(path.length ? path : null)
      if (path.length) setTrail((prev) => [...prev, ...path])
      playPromiseRef.current = play(command, scene.objects)
    },
    [scene, play],
  )

  // Backs the auto tour: reuses the exact same goto + animation path a double-click
  // already takes (see handleGoto below), just awaited so the tour's loop only ever
  // moves to the next stop once the robot has actually arrived at this one.
  const visitTourStop = useCallback(async (stop: SceneObjectResponse) => {
    await chatRef.current?.goto(stop.id, stop.name)
    await playPromiseRef.current
  }, [])

  // Cuts an interrupted tour's animation off exactly where it stands, rather than
  // snapping back to the start - reuses useCommandAnimation's own reset(), the same
  // cancellation path "Reset robot to start" already uses, just targeting the robot's
  // current position instead of the scene's fixed start.
  const handleTourInterrupt = useCallback(() => {
    reset(anim.robotPosition)
  }, [reset, anim.robotPosition])

  const tour = useAutoTour({
    objects: scene?.objects ?? [],
    visitStop: visitTourStop,
    selectedPlatformId,
    onInterrupt: handleTourInterrupt,
  })
  // Mirrors tour.running for handleGoto's guard below, which is defined with a stable
  // (deps-free) identity and so can't close over the hook's latest `running` value directly.
  const tourRunningRef = useRef(tour.running)
  tourRunningRef.current = tour.running
  // Same reason as the line above: handleGoto is a useCallback with empty deps, so it
  // reads current state through a ref rather than being rebuilt on every answer.
  const layerAvailableRef = useRef(layerAvailable)
  layerAvailableRef.current = layerAvailable

  // Double-click on an object (list row, 2D marker, or 3D marker): send the robot
  // straight there, no chat round-trip. ChatPanel.goto does the actual postCommand
  // call (it already owns the history/error-handling plumbing shared with typed
  // commands) and calls handleCommandResult itself once the command comes back.
  // Ignored while an auto tour is running - manual input is disabled in the UI for the
  // same reason (see ChatPanel's tourRunning prop): a manual goto here would race the
  // tour's own sequential goto calls and desync playPromiseRef above.
  const handleGoto = useCallback((id: string, name: string) => {
    if (tourRunningRef.current) return
    // Nothing to plan on. Refused here rather than let the request go and come back with
    // an empty route the panel would have to explain.
    if (!layerAvailableRef.current) return
    void chatRef.current?.goto(id, name)
  }, [])

  const handleResetRobot = () => {
    // Reset shares its cancellation mechanism with the animation a tour stop is
    // awaiting (see useCommandAnimation's `generation` counter) - without stopping the
    // tour first, this would abort the in-flight stop's animation without cancelling
    // the tour loop, which would then immediately continue on to the next stop from
    // wherever this reset just put the robot.
    if (tour.running) tour.stop()
    setCurrentPath(null)
    setTrail([])
    setLastCommand(null)
    // Back to the SELECTED PLATFORM's start (resolvedStart), not the scene's persisted
    // one - the reset button now sits next to the platform picker's own consequences.
    reset(resolvedStart ?? initialRobotPos)
  }

  // --- the object sets this screen talks about -------------------------------------
  //
  // ABOVE the early returns below, and they have to stay above them: these are hooks, and
  // a hook that only runs on some renders is a Rules-of-Hooks violation that takes the
  // whole page out. They were plain expressions further down until 2026-09-10.
  //
  // MEMOISED, and that is not a micro-optimisation. `useCommandAnimation` calls setState
  // on every animation frame of a move, so this whole body runs ~60 times a second while
  // the robot drives. `shownObjects` is a dep of SceneMap3D's object-box effect, whose
  // cleanup is `host.replaceChildren()` - a fresh array identity per frame tore down and
  // rebuilt every box and label in the 2D map, 60 times a second, each rebuild forcing a
  // synchronous layout to measure the new labels. Measured on hero-74 during a goto to the
  // desk: 15 fps with 14 frames over 20 ms and a worst frame of 1099.9 ms, against 60.00
  // fps and a 16.8 ms worst frame sitting idle on the same screen.
  const shownObjects = useMemo(
    () => (scene ? applyMovesToObjects(scene.objects, moves) : EMPTY_OBJECTS),
    [scene, moves],
  )
  const shownGrid = useMemo(
    () => (grid && scene ? applyMovesToGrid(grid, scene.objects, moves) : null),
    [grid, scene, moves],
  )

  // Which objects this screen is talking about right now - one rule, shared by the
  // object list, the platform comparison's columns and the verdict's denominator.
  const listedObjects = useMemo(
    () => visibleObjects(shownObjects, showFragments),
    [shownObjects, showFragments],
  )
  const listedGroups = useMemo(() => groupObjects(listedObjects), [listedObjects])
  const listedIds = useMemo(() => new Set(listedObjects.map((o) => o.id)), [listedObjects])
  // Counted over the listed objects rather than taking the response's array length, so
  // the numerator and the denominator are always about the same set.
  const reachableListedCount = useMemo(
    () => (reachableObjectIds ? listedObjects.filter((o) => reachableObjectIds.has(o.id)).length : null),
    [reachableObjectIds, listedObjects],
  )

  // The chat's opening line. Derived, not fetched: it is the same numbers the verdict bar
  // and the object list are already showing, said once in a sentence.
  const opening = useMemo(
    () =>
      openingMessage({
        grid,
        objectCount: listedIds.size,
        reachableCount: reachableListedCount,
        platformName: selectedPlatform?.display_name ?? null,
        reachability: currentReachability,
      }),
    [grid, listedIds, reachableListedCount, selectedPlatform, currentReachability],
  )


  if (notFound) {
    return (
      <div className="mx-auto max-w-3xl px-8 py-10">
        <p className="text-sm text-muted">Scene not found.</p>
        <Link to="/workspaces" className={`mt-3 inline-flex ${buttonClassName('ghost')}`}>
          Back to workspaces
        </Link>
      </div>
    )
  }

  if (!scene) {
    return <div className="mx-auto max-w-3xl px-8 py-10 text-sm text-muted">Loading…</div>
  }

  if (scene.status === 'failed') {
    return (
      <div className="mx-auto max-w-md px-8 py-20 text-center">
        <p className="text-sm text-data-unreachable">Reconstruction failed.</p>
        <JobTimeline scene={scene} variant="full" />
        <Button variant="ghost" size="compact" className="mt-3" onClick={openGuide}>
          Capture guide
        </Button>
      </div>
    )
  }

  if (scene.status !== 'done') {
    return (
      <div className="mx-auto max-w-md px-8 py-20 text-center">
        <JobTimeline scene={scene} variant="full" />
      </div>
    )
  }

  const tourDisabled = !reachableObjectIds || !anim.robotPosition || reachableObjectIds.size === 0

  // Which pose to highlight for the camera-path view's video-player sync - only while
  // the path is actually shown (nothing to highlight against otherwise) and the video
  // has reported a playback position at least once.
  const highlightedPoseIndex =
    showCameraPath && cameraTrack && videoTimeSec != null
      ? findNearestPoseIndex(cameraTrack, videoTimeSec)
      : null

  // The bottom bar's path status: the most recent command's move total, in the same
  // words the chat history uses per step ("Move · 6.0s · 2.98 m"). Derived, not stored -
  // `currentPath` is already the path the viewer is drawing, and `lastCommand` carries
  // the server's own duration/length so this never re-derives geometry the backend
  // already computed.
  const pathStatus =
    lastCommand && lastCommand.steps.some((s) => s.type === 'move')
      ? `Move · ${formatSeconds(lastCommand.total_duration_sec)} · ${formatMetres(totalPathLengthM(lastCommand.steps))}`
      : null

  const handleToggleTour = () => {
    if (tour.running) tour.stop()
    else tour.start(reachableObjectIds, anim.robotPosition)
  }

  // The scene as the screen is CURRENTLY showing it: with any "what if that were
  // somewhere else" moves applied to the objects and to the occupancy grid, so the
  // picture agrees with the reachability answer (which the server computed the same
  // way - see lib/objectMoves.ts). Identical to scene.objects/grid when nothing has
  // been moved, which is every ordinary session.


  // The object the robot is moving toward right now, by name - null the moment it stops.
  // Same gate the object list's live highlight uses: `activeObjectId` outlives the
  // journey, `anim.playing` does not.
  const drivingTo =
    anim.playing && anim.activeObjectId
      ? (shownObjects.find((o) => o.id === anim.activeObjectId)?.name ?? null)
      : null


  // --- the agent's three non-route templated commands -------------------------------
  // Each is answered here, with no model: "compare all" is a UI state, and the other two
  // are questions about reachability, which the API already answers. See
  // lib/commandGrammar.ts for what reaches them.

  const handleCompareAll = () => setCompareAll(true)

  const platformByName = (name: string) =>
    robots.find((p) => p.display_name.toLowerCase() === name.toLowerCase()) ?? null

  /** Same tie-break as the backend's resolve_object: best-observed instance of a name. */
  const objectByName = (name: string) =>
    shownObjects
      .filter((o) => !o.is_fragment && o.name.toLowerCase() === name.toLowerCase())
      .sort((a, b) => b.num_views - a.num_views || b.num_points - a.num_points)[0] ?? null

  const handleCanReach = async (robotName: string, objectName: string): Promise<string> => {
    const platform = platformByName(robotName)
    const object = objectByName(objectName)
    if (!platform) return `I don't know a robot called "${robotName}".`
    if (!object) return `There's no "${objectName}" in this room.`
    if (platform.radius_m == null) {
      return `${platform.display_name} has no measured radius yet, so there's nothing to check it against.`
    }
    try {
      // That platform's OWN radius, not the one the picker is showing - the question
      // names the robot, so the answer has to be about that robot.
      const answer = await getReachability(scene.id, platform.radius_m, moveQuery, platform.id)
      if (answer.start_status === 'none') {
        return `No - ${platform.display_name} has nowhere to stand in this room at ${formatMetres(platform.radius_m)}, so nothing is reachable for it.`
      }
      if (answer.reachable_object_ids.includes(object.id)) {
        return `Yes - ${platform.display_name} reaches ${object.name}, ${formatMetres(answer.path_length_m[object.id])} from its start.`
      }
      // The backend's own word, verbatim, exactly as the object row prints it.
      const reason = answer.unreachable_reasons[object.id]
      return `No - ${platform.display_name} cannot reach ${object.name}${reason ? `: ${reason}.` : '.'}`
    } catch (err) {
      console.error('can-reach check failed', err)
      return `Couldn't check that - the reachability request failed.`
    }
  }

  const handleMoveObject = async (objectName: string, dx: number, dy: number): Promise<string> => {
    const object = objectByName(objectName)
    if (!object) return `There's no "${objectName}" in this room.`
    const next = [...moves.filter((m) => m.objectId !== object.id), { objectId: object.id, dx, dz: dy }]
    setMoves(next)

    const before = reachableListedCount
    // formatSigned, not a hand-rolled sign: `dx >= 0 ? '+' : ''` prints "-0.00" for a
    // delta of -0.001, which claims a movement in a direction at a precision that cannot
    // show one.
    const offset = `(${formatSigned(dx)}, ${formatSigned(dy)}) m`
    const caveat =
      "Approximate: the object's bounding box is cut out of the occupancy grid and pasted at the offset, " +
      'which also clears whatever else that rectangle contained, so the room can only come out looking more ' +
      'open than it is. Routes are still planned against the unmoved room.'
    try {
      const answer = await getReachability(scene.id, robotRadius ?? undefined, moveParams(next), selectedPlatformId)
      const reachable = new Set(answer.reachable_object_ids)
      const after = listedObjects.filter((o) => reachable.has(o.id)).length
      const platform = selectedPlatform?.display_name ?? 'The robot'
      const delta = before === null || after === before ? '' : ` (was ${before})`
      return `Moved ${object.name} by ${offset}. ${platform} now reaches ${after} of ${listedIds.size}${delta}. ${caveat}`
    } catch (err) {
      console.error('reachability after move failed', err)
      return `Moved ${object.name} by ${offset}, but the reachability request failed, so the count above may be stale. ${caveat}`
    }
  }

  // LEFT column - "scene and robot": which view, which robot, where it starts.
  //
  // FOUR CONTROLS, ONE LEFT EDGE AND ONE WIDTH: the robot select, the workspace select,
  // the view toggle and the video card. Every section body is `px-s4` and every control
  // in it is full width, so the column reads as one stack rather than as four boxes that
  // each chose their own inset - the workspace select used to pad itself as well (12px on
  // top of this 16px) and was the one control that lined up with nothing.
  //
  // The video is LAST and takes what is left (`flex-1`), which is what removes the
  // scrollbar at 1440x900: the column's height is spent on the one thing that can use any
  // amount of it, instead of on a gap under a fixed stack.
  const leftColumn = (
    <>
      {/* ROBOT is first, directly under the verdict bar, because the verdict is ABOUT the
         robot: "13 of 17 reachable · TurtleBot3 Burger" is a sentence whose subject is
         this dropdown, and the two were four sections apart. Everything the screen shows
         changes when this changes, so it reads top-down now - pick the robot, read the
         answer, look at the room.

         START POSITION is that section's caption now, not a section of its own: it is a
         fact about the platform this picker selects, and it was three rows (header,
         coordinate, prose) for a number nobody types in. The coordinate and the reason
         are in its (i). */}
      <RobotPanel
        platforms={robots}
        selectedPlatformId={selectedPlatformId}
        onSelectPlatform={handleSelectPlatform}
        caption={
          <StartPositionCaption
            start={
              currentReachability
                ? {
                    status: currentReachability.start_status,
                    x: currentReachability.start_x,
                    z: currentReachability.start_z,
                    movedM: currentReachability.start_moved_m,
                  }
                : null
            }
            radiusM={robotRadius}
          />
        }
      />

      {/* The left panel's first section header lands on the same baseline as the right
         one's - probe_visual asserts they are within 1px, and RobotPanel above is a
         PanelSection too, so that still holds.
         The workspace dropdown and the room's dimensions live INSIDE a section; they used
         to sit above any header, which is what put the panels 168px out of step.

         The project name is gone from the body: the dropdown above it already says it,
         and it said it twice in two different type sizes. */}
      <PanelSection title="Workspace" className="mt-s6">
        <div className="flex flex-col gap-s2 px-s4 pb-s2 pt-s1" data-testid="room-summary">
          {/* No chrome of its own here: the section body places it, like every other
             control in this column. The default (a bottom rule and 12px of padding) is
             the global sidebar's, where it is the only thing in its band. */}
          <WorkspaceSwitcher className="" />
          <div className="ui-caption flex flex-wrap items-center gap-x-s3 gap-y-s1 font-mono">
            {shownGrid && (
              <span data-testid="room-size">
                {(shownGrid.width * shownGrid.resolution).toFixed(1)} ×{' '}
                {(shownGrid.height * shownGrid.resolution).toFixed(1)} m
              </span>
            )}
            {scene.ceiling_y != null && (
              <span data-testid="room-ceiling">ceiling {formatMetres(scene.ceiling_y)}</span>
            )}
            {scene.vocab_source && (
              <span data-testid="room-vocab">vocab: {vocabSourceLabel(scene.vocab_source, scene.vocab_model)}</span>
            )}
          </div>
          {scene.oversized_object_warning && (
            <span
              data-testid="room-oversized"
              className="rounded-sm border-hair border-status-partial/40 bg-status-partial/10 px-s2 py-s1 text-caption text-status-partial"
              title={`Unusually large for its type - likely two objects merged during reconstruction: ${scene.oversized_object_warning}`}
            >
              ⚠ oversized object: {scene.oversized_object_warning}
            </span>
          )}
          <div className="flex flex-wrap items-center gap-x-s3 gap-y-s1">
            <CriticBadge sceneId={scene.id} />
          </div>
        </div>
      </PanelSection>

      <PanelSection title="View" className="mt-s6">
        <div className="px-s4 pt-s1">
          {/* Full width, like every other control in this column. `w-fit` made the one
             two-option segmented control the only thing here that did not reach both
             margins, which read as an unfinished row rather than as a deliberate size. */}
          <MapViewToggle value={viewMode} onChange={setViewMode} className="w-full" />
        </div>
      </PanelSection>

      {anim.label && (
        <div className="mt-s6 px-s4">
          <Badge variant="neutral" className="w-full gap-1.5" data-testid="anim-status">
            <span className={`h-1.5 w-1.5 rounded-full ${anim.playing ? 'bg-accent' : 'bg-muted'}`} />
            {anim.label}
          </Badge>
        </div>
      )}

      {/* LAST, and it takes the rest of the column. `min-h-0` is what lets it actually
         shrink inside the flex parent; without it the video's natural height wins and the
         column grows a scrollbar instead. The player letterboxes inside whatever box it
         is given (object-fit: contain), so a portrait source and a landscape source both
         fit without either being cropped or stretching the column. */}
      <PanelSection title="Source video" className="mt-s6 flex min-h-0 flex-1 flex-col">
        <div className="flex min-h-0 flex-1 flex-col pt-s1 pb-s2">
          <VideoPlayer videoId={scene.video_id} filename={video?.filename} onTimeUpdate={setVideoTimeSec} />
        </div>
      </PanelSection>
    </>
  )

  // RIGHT column - "objects and agent". The object list (or, with "Compare all" on, the
  // platform x object matrix in its place), chat filling what is left, and one bottom
  // bar carrying the command input, SEND, AUTO TOUR, RESET ROBOT and the path status.
  const rightColumn = (
    <>
      {unreachableNotice && (
        <p className="shrink-0 border-b-hair border-data-unreachable/20 bg-data-unreachable/10 px-3 py-2 text-xs text-data-unreachable">
          {unreachableNotice}
        </p>
      )}
      {compareAll ? (
        <PlatformComparison
          sceneId={scene.id}
          platforms={robots}
          groups={listedGroups}
          selectedPlatformId={selectedPlatformId}
        />
      ) : (
        // A CAP, not a share: the chips block takes the height its content needs up to
        // 35% of the panel and scrolls inside that, and everything below it belongs to
        // the chat. `flex-1` gave the list every pixel the chat did not claim, so on a
        // 17-object scene two thirds of the column was a list nobody was reading while
        // the conversation had four lines. A scene with no objects - the 15 fps hero is
        // one - takes only the room its one line needs, same as before.
        <div
          className={`overflow-y-auto ${
            listedObjects.length > 0 ? 'min-h-0 max-h-[35%] shrink-0' : 'shrink-0'
          }`}
          data-testid="objects-scroll"
        >
          <ObjectList
            objects={shownObjects}
            showFragments={showFragments}
            selectedObjectId={selectedObjectId}
            onSelectObject={setSelectedObjectId}
            activeObjectId={anim.activeObjectId}
            // "Right now", per the spec: the last command's target stays in
            // `activeObjectId` after arrival, so the live highlight is gated on the
            // animation still running.
            targetObjectId={anim.playing ? anim.activeObjectId : null}
            pickedIds={anim.pickedIds}
            hoveredObjectId={hoveredObjectId}
            onHoverObject={setHoveredObjectId}
            onFocusObject={handleFocusObject}
            onGoto={handleGoto}
            reachableObjectIds={reachableObjectIds}
            unreachableReasons={currentReachability?.unreachable_reasons ?? EMPTY_REASONS}
            pathLengthM={currentReachability?.path_length_m ?? EMPTY_LENGTHS}
            onUnreachableClick={handleUnreachableClick}
          />
        </div>
      )}
      {/* THE CHAT IS A CARD - one step lighter than the panel it sits on (--card-bg is
         surface-raised), a hairline in white 8% and an 8px radius, the same shape the
         video card in the left column has. It is the one place on this screen you type
         into, and it read as the leftover space under the object list. `overflow-hidden`
         is what keeps the history's own scrollbar inside the rounded corner. mt-s2 is the
         measured gap to the chips above it. */}
      <div className="mx-s3 mb-s3 mt-s2 flex min-h-0 flex-1 flex-col overflow-hidden rounded-card border-hair border-card-border bg-card">
        <ChatPanel
          ref={chatRef}
          sceneId={scene.id}
          onCommandResult={handleCommandResult}
          objects={shownObjects}
          platforms={robots}
          onCanReach={handleCanReach}
          onCompareAll={handleCompareAll}
          onMoveObject={handleMoveObject}
          radius={robotRadius ?? undefined}
          robotId={selectedPlatformId}
          robotPosition={anim.robotPosition}
          tourRunning={tour.running}
          tourDisabled={tourDisabled}
          onToggleTour={handleToggleTour}
          onResetRobot={handleResetRobot}
          pathStatus={pathStatus}
          drivingTo={drivingTo}
          opening={opening}
        />
      </div>
    </>
  )

  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden">
      <SceneToolbar
        sceneId={scene.id}
        verdict={
          <SceneVerdict
            platformName={selectedPlatform?.display_name ?? null}
            reachableCount={reachableListedCount}
            totalCount={listedIds.size}
            noStart={currentReachability?.start_status === 'none'}
            robotRadiusM={robotRadius}
            layerAvailable={layerAvailable}
            unknownCellsOnRoute={currentReachability?.unknown_cells_on_route ?? 0}
            gridResolutionM={grid?.resolution ?? null}
            tightestGapM={tightestGapM}
          />
        }
        compareAll={compareAll}
        onToggleCompareAll={() => setCompareAll((v) => !v)}
        movedCount={moves.length}
        onResetMoves={() => setMoves([])}
      />

      <div className="flex min-h-0 flex-1 overflow-hidden">
        {/* LEFT - scene and robot. Same width as the inspector column so the canvas sits
           optically centred. */}
        <div
          className="flex shrink-0 flex-col overflow-y-auto border-r-hair border-border bg-surface pt-s2"
          style={{ width: isMid ? '260px' : 'var(--inspector-w)' }}
          data-testid="left-column"
        >
          {leftColumn}
        </div>

        <div className="flex min-h-0 flex-1">
          <div className="relative min-w-0 flex-1 border-r-hair border-border">
            {!layerAvailable && (
              <div
                className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center bg-canvas/80"
                data-testid="layer-unavailable"
              >
                <div className="max-w-sm rounded border-hair border-border bg-surface px-4 py-3 text-center">
                  <p className="text-[15px] text-fg">Layer not available</p>
                  <p className="mt-1 text-xs text-muted">
                    This scene has no navigation layer, so there is nothing to plan a route
                    on. The map, the reachability verdict and “go to” all read that one
                    layer — see the API response for the command that produces it.
                  </p>
                </div>
              </div>
            )}
            <SceneView
              viewMode={viewMode}
              renderMode={renderMode}
              onChangeRenderMode={setRenderMode}
              showCameraPath={cameraTrack && cameraTrack.length > 1 ? showCameraPath : null}
              onChangeShowCameraPath={setShowCameraPath}
              showFragments={showFragments}
              onChangeShowFragments={setShowFragments}
              layers={sceneLayers}
              onToggleLayer={handleToggleLayer}
              layersAvailable={layersAvailable}
              sceneProps={{
                sceneId: scene.id,
                meshUrl: sceneMeshUrl(scene.id),
                cloudUrl: sceneCloudUrl(scene.id),
                msaGlbUrl: sceneMsaGlbUrl(scene.id),
                msaMetaUrl: sceneMsaMetaUrl(scene.id),
                ceilingY: scene.ceiling_y,
                grid: shownGrid,
                objects: shownObjects,
                selectedObjectId,
                onSelectObject: setSelectedObjectId,
                hoveredObjectId,
                onHoverObject: setHoveredObjectId,
                activeObjectId: anim.activeObjectId,
                reachableObjectIds,
                reachableCells: currentReachability?.reachable_cells ?? null,
                layers,
                msaMeshVisible: sceneLayers.layout && layersAvailable.layout,
                showMap: sceneLayers.map,
                showPoints: sceneLayers.points,
                showMesh: sceneLayers.mesh && layersAvailable.mesh,
                meshLayerUrl: sceneMeshUrl(scene.id),
                // Points reads ONE of the two, never both: /cloud where the scene has it,
                // /mesh only where /mesh is a points primitive. Passing both and letting
                // the loader fall back is what made "the same file twice" possible.
                pointsCloudUrl: pointsSource === 'cloud' ? sceneCloudUrl(scene.id) : null,
                pointsMeshUrl: pointsSource === 'mesh' ? sceneMeshUrl(scene.id) : null,
                unreachableReasons: currentReachability?.unreachable_reasons ?? EMPTY_REASONS,
                robotPosition: anim.robotPosition,
                startPosition: resolvedStart,
                robotRadius,
                robotLengthM,
                robotWidthM,
                robotMeshUrl,
                robotHeightM,
                currentPath,
                trailPoints: trail,
                cameraTrack: showCameraPath ? cameraTrack : null,
                highlightedPoseIndex,
                onGoto: handleGoto,
                onUnreachableClick: handleUnreachableClick,
                focusObjectId: focusRequest?.id ?? null,
                focusNonce: focusRequest?.nonce ?? 0,
              }}
            />

          </div>

          {/* Always inline, never an overlay drawer - the app has a hard 1280px floor
             (spec 9), so there's no viewport width where a collapsed/drawer panel is
             actually needed; below the floor the layout just scrolls horizontally. */}
          <div
            className="flex min-h-0 shrink-0 flex-col bg-surface pt-s2"
            style={{ width: isMid ? '280px' : 'var(--inspector-w)' }}
            data-testid="right-column"
          >
            {rightColumn}
          </div>
        </div>
      </div>
    </div>
  )
}
