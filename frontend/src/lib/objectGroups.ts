import type { SceneObjectResponse } from '../api/types'

export interface ObjectGroup {
  /** Lowercased name - the grouping key and the id used by `data-object-group`. */
  key: string
  /** The name as the first-seen instance spells it. */
  name: string
  items: SceneObjectResponse[]
}

/** Group a scene's objects by name, the way the object list has always shown them:
 * "lamp (4)" is one row that expands into four instances, not four rows.
 *
 * Extracted out of ObjectList so the platform comparison can use exactly the same
 * columns as the list it sits next to - "same object columns" is the whole point of
 * that table, and two independent groupings would silently drift apart the first time
 * either changed its sort.
 *
 * Instances inside a group are ordered by `num_views` descending (the best-observed
 * instance first), and the groups themselves keep first-appearance order. */
export function groupObjects(objects: readonly SceneObjectResponse[]): ObjectGroup[] {
  const byKey = new Map<string, ObjectGroup>()
  for (const o of objects) {
    const key = o.name.toLowerCase()
    let g = byKey.get(key)
    if (!g) {
      g = { key, name: o.name, items: [] }
      byKey.set(key, g)
    }
    g.items.push(o)
  }
  for (const g of byKey.values()) g.items.sort((a, b) => b.num_views - a.num_views)
  return Array.from(byKey.values())
}

/** Non-fragment objects, or everything when fragments are being shown - the one rule
 * for "which objects is this screen talking about", shared by the object list, the
 * platform comparison and the verdict's denominator. */
export function visibleObjects(
  objects: readonly SceneObjectResponse[],
  showFragments: boolean,
): SceneObjectResponse[] {
  return showFragments ? [...objects] : objects.filter((o) => !o.is_fragment)
}
