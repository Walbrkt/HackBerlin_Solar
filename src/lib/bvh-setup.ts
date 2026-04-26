/**
 * Patches THREE so any mesh's `.raycast()` uses the BVH index when one has
 * been built on its geometry. Importing this module once is enough — the
 * patch is global. After this, calling `geometry.computeBoundsTree()` on a
 * BufferGeometry stores a `MeshBVH` on it; subsequent raycasts (including
 * R3F's pointer events) skip per-triangle scans.
 *
 * Reference: https://github.com/gkjohnson/three-mesh-bvh
 */
import * as THREE from "three";
import {
  acceleratedRaycast,
  computeBoundsTree,
  disposeBoundsTree,
} from "three-mesh-bvh";

THREE.Mesh.prototype.raycast = acceleratedRaycast;
(THREE.BufferGeometry.prototype as unknown as { computeBoundsTree: typeof computeBoundsTree })
  .computeBoundsTree = computeBoundsTree;
(THREE.BufferGeometry.prototype as unknown as { disposeBoundsTree: typeof disposeBoundsTree })
  .disposeBoundsTree = disposeBoundsTree;

/** Build a BVH for every mesh under `root`. Idempotent. */
export function indexModelForBVH(root: THREE.Object3D): void {
  root.traverse((obj) => {
    const mesh = obj as THREE.Mesh;
    if (!mesh.isMesh || !mesh.geometry) return;
    const geom = mesh.geometry as THREE.BufferGeometry & {
      boundsTree?: unknown;
      computeBoundsTree?: () => void;
    };
    if (geom.boundsTree || !geom.computeBoundsTree) return;
    geom.computeBoundsTree();
  });
}

/** Dispose any BVH indices on `root` (call before discarding the model). */
export function disposeModelBVH(root: THREE.Object3D): void {
  root.traverse((obj) => {
    const mesh = obj as THREE.Mesh;
    if (!mesh.isMesh || !mesh.geometry) return;
    const geom = mesh.geometry as THREE.BufferGeometry & {
      disposeBoundsTree?: () => void;
    };
    geom.disposeBoundsTree?.();
  });
}
