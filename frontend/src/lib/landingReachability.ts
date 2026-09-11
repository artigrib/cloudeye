// Mirrors scripts/build-landing-scene.mjs's reachability.json output - the landing
// page's static, precomputed comparison data (spec 7.2: "без запросов к бэкенду").

export interface LandingPlatform {
  id: string
  display_name: string
  radius_m: number
}

export interface LandingObject {
  name: string
  position: [number, number, number]
  reachableBy: string[]
}

export interface LandingReachability {
  room: { width_m: number; depth_m: number; ceiling_m: number }
  platforms: LandingPlatform[]
  objects: LandingObject[]
  robotStart: [number, number, number]
}

export async function loadLandingReachability(): Promise<LandingReachability> {
  const res = await fetch('/landing/reachability.json')
  if (!res.ok) throw new Error(`Failed to load landing reachability data: ${res.status}`)
  return res.json() as Promise<LandingReachability>
}
