import { useCallback, useMemo, useRef, useState, Suspense, useEffect } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, Grid, Environment, useGLTF } from "@react-three/drei";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { DRACOLoader } from "three/examples/jsm/loaders/DRACOLoader.js";
import { MeshoptDecoder } from "three/examples/jsm/libs/meshopt_decoder.module.js";

// Shared loader with Draco + Meshopt support (handles compressed .glb files)
const dracoLoader = new DRACOLoader();
dracoLoader.setDecoderPath("https://www.gstatic.com/draco/versioned/decoders/1.5.7/");
dracoLoader.setDecoderConfig({ type: "js" });

const gltfLoader = new GLTFLoader();
gltfLoader.setDRACOLoader(dracoLoader);
gltfLoader.setMeshoptDecoder(MeshoptDecoder);
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import {
  Upload,
  Sun,
  Trash2,
  Euro,
  Ruler,
  Grid3x3,
  Sparkles,
  Battery,
  Target,
  MousePointerClick,
} from "lucide-react";
import { cn } from "@/lib/utils";
import DesignFromPrompt, {
  type DesignFromPromptResponse,
} from "@/components/DesignFromPrompt";
import PlacementHUD from "@/components/PlacementHUD";
import { indexModelForBVH } from "@/lib/bvh-setup";

// €/panel and €/kWh used for the live HUD; mirrors backend constants.
const PRICE_PER_PANEL = 1500;
const PRICE_PER_KWH_BATTERY = 800;
// Panel's local "out" axis (the BoxGeometry depth axis is Z, the GLB models we
// load are oriented so their face normal also points along local +Z).
const PANEL_LOCAL_NORMAL = new THREE.Vector3(0, 0, 1);
// World up — these models are Z-up (see model.box.min.z usage below).
const WORLD_UP = new THREE.Vector3(0, 0, 1);

/**
 * Registry of panel .glb assets. To add another model from Sketchfab:
 *   1. Drop the file under `public/solar_panels/<name>.glb`.
 *   2. Add an entry below with its physical dimensions and price.
 *   3. Make sure the model's face normal is along local +Z (same convention
 *      the existing panels use); otherwise re-export with the correct orientation.
 *   4. The Panel component auto-fits the GLB to `data.width × data.height`
 *      via a Box3 measurement, so any reasonable size works.
 *
 * `flat` goes on horizontal surfaces; `slanted` is for tilted surfaces
 * (assumes the slanted glb has a wedge / mounting frame baked in).
 */
const PANEL_MODELS = {
  flat: {
    url: "/solar_panels/solar_panel_flat.glb",
    label: "Flat panel",
    pricePerUnit: 1500, // EUR per panel including install — tune from supplier data.
    width: 1.0,
    height: 1.65,
    depth: 0.18,
    color: "hsl(140, 100%, 55%)", // preview tint
  },
  slanted: {
    url: "/solar_panels/solar_panel_slanted.glb",
    label: "Slanted (tilt-mounted) panel",
    pricePerUnit: 1700, // wedge frame premium.
    width: 1.0,
    height: 1.65,
    depth: 0.45,
    color: "hsl(28, 95%, 60%)",
  },
} as const;
type PanelModelKey = keyof typeof PANEL_MODELS;

// Preload both so swapping mid-session doesn't stall the canvas.
useGLTF.preload(PANEL_MODELS.flat.url);
useGLTF.preload(PANEL_MODELS.slanted.url);

// Threshold (degrees from world-up) below which a surface is considered "flat".
// Real flat roofs typically have a 1–5° drainage tilt; 10° is a safe cutoff.
const FLAT_TILT_THRESHOLD_DEG = 10;

/**
 * Pick the appropriate panel model for a hit normal.
 *   - Flat surface (≤10° from world up) → flat panel laid directly on the roof.
 *   - Tilted surface                     → slanted (tilt-frame) panel.
 */
function pickPanelModel(worldNormal: THREE.Vector3): PanelModelKey {
  const tiltDeg = THREE.MathUtils.radToDeg(
    Math.acos(THREE.MathUtils.clamp(worldNormal.dot(WORLD_UP), -1, 1)),
  );
  return tiltDeg <= FLAT_TILT_THRESHOLD_DEG ? "flat" : "slanted";
}

/**
 * Auto-place up to `count` panel poses near `origin`, staying on the same roof
 * surface. Implementation is a flood-fill (BFS) over a tangent-plane grid:
 *
 *   - Start at the seed cell `(0,0)`. Expand to its 4-neighbours.
 *   - For each candidate cell, ray-cast straight along the seed's normal axis
 *     (BVH-accelerated). Accept the cell only if the hit:
 *        a) shares a normal with the seed within `maxNormalDeg`         (no walls / dormers),
 *        b) lies within `maxPlaneOffset` of the seed's tangent plane    (no ground / trees), and
 *        c) is no further than `maxJump` from its parent cell's hit pos (no jumping voids).
 *   - Accepted cells push their neighbours; rejected cells become walls.
 *   - Result keeps panels parallel to the seed orientation but each one is
 *     placed at its actual hit point so it hugs the local surface.
 *
 * The (b) check is the one that distinguishes "house roof" from
 * "ground / trees / lawn next to the house" when both happen to be flat —
 * grid sampling alone can't tell them apart. (c) handles thin gaps where
 * the ray punches straight through to the ground beyond the eaves.
 *
 * Each kept pose carries the cell's own normal, so callers can pick the
 * appropriate panel model (flat/slanted) per cell.
 */
function proposeLayout(
  origin: SnapPose,
  count: number,
  model: THREE.Object3D,
  opts: {
    gap?: number;
    maxNormalDeg?: number;
    /** Max signed distance from seed's tangent plane (metres). */
    maxPlaneOffset?: number;
    /** Max distance between adjacent accepted hit points (metres). */
    maxJump?: number;
    /** Search radius cap, in cells, to keep BFS bounded. */
    maxRadius?: number;
    /** Ray origin offset above the surface (metres). */
    lift?: number;
  } = {},
): SnapPose[] {
  const gap = opts.gap ?? 0.2;
  const maxNormalDeg = opts.maxNormalDeg ?? 20;
  const maxPlaneOffset = opts.maxPlaneOffset ?? 0.8;
  const maxRadius = opts.maxRadius ?? Math.max(6, Math.ceil(Math.sqrt(count) * 2) + 3);
  const lift = opts.lift ?? 5;

  const stepU = PANEL_W + gap;
  const stepV = PANEL_H + gap;
  // Default maxJump must be larger than the longest cell step or the BFS will
  // reject every legitimate neighbour. Allow an extra ~25% for slope geometry,
  // and add the plane-tolerance so a cell whose hit sits at the edge of the
  // tangent-plane band still passes.
  const maxJump = opts.maxJump ?? Math.max(stepU, stepV) * 1.25 + maxPlaneOffset;

  const originPos = new THREE.Vector3(...origin.position);
  const originQ = new THREE.Quaternion(...origin.quaternion);
  const xAxis = new THREE.Vector3(1, 0, 0).applyQuaternion(originQ);
  const yAxis = new THREE.Vector3(0, 1, 0).applyQuaternion(originQ);
  const zAxis = new THREE.Vector3(0, 0, 1).applyQuaternion(originQ).normalize();

  const cosLimit = Math.cos(THREE.MathUtils.degToRad(maxNormalDeg));
  const raycaster = new THREE.Raycaster();
  const tmpNormal = new THREE.Vector3();

  type Cell = { i: number; j: number; parentHit: THREE.Vector3 };
  const accepted = new Map<string, { pose: SnapPose; dist2: number }>();
  const rejected = new Set<string>();
  const queue: Cell[] = [{ i: 0, j: 0, parentHit: originPos.clone() }];
  const key = (i: number, j: number) => `${i},${j}`;

  while (queue.length > 0 && accepted.size < count * 3) {
    const cell = queue.shift()!;
    const k = key(cell.i, cell.j);
    if (accepted.has(k) || rejected.has(k)) continue;
    if (Math.abs(cell.i) > maxRadius || Math.abs(cell.j) > maxRadius) {
      rejected.add(k);
      continue;
    }

    const target = originPos
      .clone()
      .addScaledVector(xAxis, cell.i * stepU)
      .addScaledVector(yAxis, cell.j * stepV);
    const rayOrigin = target.clone().addScaledVector(zAxis, lift);
    raycaster.set(rayOrigin, zAxis.clone().negate());
    const hits = raycaster.intersectObject(model, true);
    const hit = hits[0];
    if (!hit || !hit.face) {
      rejected.add(k);
      continue;
    }

    tmpNormal
      .copy(hit.face.normal)
      .transformDirection(hit.object.matrixWorld)
      .normalize();

    // (a) normal angle vs. seed normal
    if (tmpNormal.dot(zAxis) < cosLimit) {
      rejected.add(k);
      continue;
    }
    // (b) signed distance from seed's tangent plane: filters ground/trees/lawn
    const planeOffset = Math.abs(hit.point.clone().sub(originPos).dot(zAxis));
    if (planeOffset > maxPlaneOffset) {
      rejected.add(k);
      continue;
    }
    // (c) reachable from parent without a big vertical jump
    const jump = hit.point.distanceTo(cell.parentHit);
    if (jump > maxJump) {
      rejected.add(k);
      continue;
    }

    const hitWorldNormal: [number, number, number] = [tmpNormal.x, tmpNormal.y, tmpNormal.z];
    accepted.set(k, {
      pose: {
        position: [hit.point.x, hit.point.y, hit.point.z],
        quaternion: [...origin.quaternion],
        worldNormal: hitWorldNormal,
      },
      dist2: target.distanceToSquared(originPos),
    });

    // Expand to 4-neighbours.
    const hitVec = hit.point.clone();
    queue.push({ i: cell.i + 1, j: cell.j, parentHit: hitVec });
    queue.push({ i: cell.i - 1, j: cell.j, parentHit: hitVec });
    queue.push({ i: cell.i, j: cell.j + 1, parentHit: hitVec });
    queue.push({ i: cell.i, j: cell.j - 1, parentHit: hitVec });
  }

  return [...accepted.values()]
    .sort((a, b) => a.dist2 - b.dist2)
    .slice(0, count)
    .map((c) => c.pose);
}

/**
 * Build a panel pose from a surface hit. The panel's local Z is rotated to
 * align with the world-space surface normal; X is forced to be horizontal
 * (along the eave on a sloped roof), with `yawRad` rotating around the normal
 * for user-controlled in-plane rotation.
 *
 * Edge case: when the surface is perfectly horizontal (flat roof), the
 * "horizontal X" direction is undefined — fall back to world X.
 */
function surfacePoseFromHit(
  point: THREE.Vector3,
  worldNormal: THREE.Vector3,
  yawRad: number,
): SnapPose {
  const z = worldNormal.clone().normalize();
  let x = new THREE.Vector3().crossVectors(WORLD_UP, z);
  if (x.lengthSq() < 1e-6) x.set(1, 0, 0); // flat roof: pick world X.
  x.normalize();
  const y = new THREE.Vector3().crossVectors(z, x).normalize();
  // Build the base rotation: columns are the panel's local X/Y/Z in world coords.
  const baseQ = new THREE.Quaternion().setFromRotationMatrix(
    new THREE.Matrix4().makeBasis(x, y, z),
  );
  // Apply user yaw around the panel's local Z (= surface normal).
  const yawQ = new THREE.Quaternion().setFromAxisAngle(PANEL_LOCAL_NORMAL, yawRad);
  baseQ.multiply(yawQ);
  return {
    position: [point.x, point.y, point.z],
    quaternion: [baseQ.x, baseQ.y, baseQ.z, baseQ.w],
    worldNormal: [z.x, z.y, z.z],
  };
}

const COST_PER_PANEL = 250;
const PANEL_W = 1.0;
const PANEL_H = 1.65;
const PANEL_GAP = 0.05;
const GRID_SIZE = 150;
const GRID_DIVISIONS = 25;
const GRID_CELL_HEIGHT = GRID_SIZE / GRID_DIVISIONS;

type PanelData = {
  id: string;
  position: [number, number, number];
  quaternion: [number, number, number, number];
  width: number;
  height: number;
  depth?: number;
  /** Which GLB model to render this panel as. Defaults to "flat" if absent. */
  modelKey?: PanelModelKey;
};

type Candidate = {
  id: string;
  quaternion: [number, number, number, number];
  slots: PanelData[];
  outline: [number, number, number][];
  height: number;
};

// --- 2-D OBB collision (SAT) ---
// Extract Z-axis rotation angle from a unit quaternion.
function zAngleFromQ(q: [number, number, number, number]): number {
  const [qx, qy, qz, qw] = q;
  return Math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz));
}

// Project an OBB (centre cx,cy; half-extents hw,hh; rotation angle) onto axis (tx,ty).
function satInterval(cx: number, cy: number, hw: number, hh: number, angle: number, tx: number, ty: number): [number, number] {
  const cos = Math.cos(angle), sin = Math.sin(angle);
  const extent = hw * Math.abs(cos * tx + sin * ty) + hh * Math.abs(-sin * tx + cos * ty);
  const center = cx * tx + cy * ty;
  return [center - extent, center + extent];
}

function panelsOverlap(a: PanelData, b: PanelData): boolean {
  const angleA = zAngleFromQ(a.quaternion);
  const angleB = zAngleFromQ(b.quaternion);
  const axes: [number, number][] = [
    [Math.cos(angleA), Math.sin(angleA)],
    [-Math.sin(angleA), Math.cos(angleA)],
    [Math.cos(angleB), Math.sin(angleB)],
    [-Math.sin(angleB), Math.cos(angleB)],
  ];
  for (const [tx, ty] of axes) {
    const iA = satInterval(a.position[0], a.position[1], a.width / 2, a.height / 2, angleA, tx, ty);
    const iB = satInterval(b.position[0], b.position[1], b.width / 2, b.height / 2, angleB, tx, ty);
    if (iA[1] - iB[0] < 1e-4 || iB[1] - iA[0] < 1e-4) return false;
  }
  return true;
}

type SnapPose = {
  position: [number, number, number];
  quaternion: [number, number, number, number];
  /** World-space surface normal at the hit point — drives panel-model selection. */
  worldNormal: [number, number, number];
};

// Renders the loaded model. When `snap` props are provided, attaches pointer
// handlers so hovering or clicking the surface emits a BVH-accelerated snap
// pose. Placement is intentionally *not* committed on click — that goes
// through the explicit Confirm action — so accidentally clicking on existing
// panels never spawns extras. `yawRad` rotates the snap orientation around
// the surface normal so the user can spin the preview before committing.
function ModelMesh({
  object,
  snap,
}: {
  object: THREE.Object3D;
  snap?: {
    yawRad: number;
    onHover: (pose: SnapPose | null) => void;
    onAim: (pose: SnapPose) => void;
  };
}) {
  // Reusable scratch — created once, reused on every pointermove.
  const tmpNormalRef = useRef(new THREE.Vector3());

  const poseFromEvent = useCallback(
    (e: any, yawRad: number): SnapPose | null => {
      if (!e.face) return null;
      // face.normal is in the hit object's LOCAL space — promote to world.
      tmpNormalRef.current
        .copy(e.face.normal)
        .transformDirection(e.object.matrixWorld)
        .normalize();
      return surfacePoseFromHit(e.point, tmpNormalRef.current, yawRad);
    },
    [],
  );

  if (!snap) return <primitive object={object} />;

  return (
    <primitive
      object={object}
      onPointerMove={(e: any) => {
        e.stopPropagation();
        const pose = poseFromEvent(e, snap.yawRad);
        if (pose) snap.onHover(pose);
      }}
      onPointerOut={() => snap.onHover(null)}
      onClick={(e: any) => {
        // Don't place — only "aim". The Confirm action (button or Enter key)
        // commits a panel. This means clicks on placed Panel meshes never
        // accidentally drop a duplicate panel through them.
        if (e.button !== 0) return;
        // Bail if a non-model object (a placed Panel) is in front of us.
        const front = e.intersections?.[0]?.object as THREE.Object3D | undefined;
        if (front && !isDescendantOf(front, e.eventObject as THREE.Object3D)) return;
        e.stopPropagation();
        const pose = poseFromEvent(e, snap.yawRad);
        if (pose) snap.onAim(pose);
      }}
    />
  );
}

// Walk up the parent chain from `node` looking for `ancestor`.
function isDescendantOf(node: THREE.Object3D, ancestor: THREE.Object3D): boolean {
  let cur: THREE.Object3D | null = node;
  while (cur) {
    if (cur === ancestor) return true;
    cur = cur.parent;
  }
  return false;
}

// Translucent preview rendered at the cursor in free-placement mode.
// Box dimensions and color reflect the model that would actually be placed
// (flat vs slanted), so the user sees which panel they're about to drop.
function PreviewPanel({ pose }: { pose: SnapPose }) {
  const modelKey = pickPanelModel(new THREE.Vector3(...pose.worldNormal));
  const def = PANEL_MODELS[modelKey];
  return (
    <mesh position={pose.position} quaternion={pose.quaternion}>
      <boxGeometry args={[def.width, def.height, def.depth]} />
      <meshStandardMaterial
        color={def.color}
        emissive={def.color}
        emissiveIntensity={0.6}
        transparent
        opacity={0.55}
        depthWrite={false}
      />
    </mesh>
  );
}

function Panel({
  data,
  selected,
  onMove,
  onHoldingChange,
  onToggleSelect,
}: {
  data: PanelData;
  selected: boolean;
  onMove: (id: string, position: [number, number, number]) => void;
  onHoldingChange: (id: string | null) => void;
  onToggleSelect: (id: string) => void;
}) {
  const [hovered, setHovered] = useState(false);
  const [dragging, setDragging] = useState(false);
  const dragPlaneRef = useRef(new THREE.Plane());
  const dragOffsetRef = useRef(new THREE.Vector3());
  const intersectionRef = useRef(new THREE.Vector3());

  // Resolve the GLB url from the per-panel modelKey (default "flat" for legacy data).
  const modelKey: PanelModelKey = data.modelKey ?? "flat";
  const { scene } = useGLTF(PANEL_MODELS[modelKey].url);

  const [clonedScene, panelScale] = useMemo(() => {
    const clone = scene.clone(true);
    clone.traverse((obj) => {
      const mesh = obj as THREE.Mesh;
      if (mesh.isMesh) {
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        mesh.material = Array.isArray(mesh.material)
          ? mesh.material.map((m) => m.clone())
          : (mesh.material as THREE.Material).clone();
      }
    });
    const box = new THREE.Box3().setFromObject(clone);
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    clone.position.sub(center);
    const sx = size.x > 0 ? data.width / size.x : 1;
    const sy = size.y > 0 ? data.height / size.y : 1;
    return [clone, [sx, sy, sx] as [number, number, number]];
  }, [scene, data.width, data.height]);

  useEffect(() => {
    clonedScene.traverse((obj) => {
      const mesh = obj as THREE.Mesh;
      if (!mesh.isMesh) return;
      const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
      mats.forEach((m) => {
        const mat = m as THREE.MeshStandardMaterial;
        if (!mat.emissive) return;
        if (selected) {
          mat.emissive.set(0.05, 0.25, 0.7);
          mat.emissiveIntensity = 0.9;
        } else if (hovered) {
          mat.emissive.setScalar(0.12);
          mat.emissiveIntensity = 1.0;
        } else {
          mat.emissive.setScalar(0);
          mat.emissiveIntensity = 0;
        }
      });
    });
  }, [hovered, selected, clonedScene]);

  const startHolding = useCallback((e: any) => {
    e.stopPropagation();
    if (e.nativeEvent?.shiftKey ?? e.shiftKey) {
      onToggleSelect(data.id);
      return;
    }
    if (e.target && typeof e.target.setPointerCapture === "function") {
      e.target.setPointerCapture(e.pointerId);
    }
    const plane = dragPlaneRef.current;
    plane.set(new THREE.Vector3(0, 0, 1), -data.position[2]);
    const hit = e.ray.intersectPlane(plane, intersectionRef.current);
    if (hit) {
      dragOffsetRef.current.set(
        data.position[0] - hit.x,
        data.position[1] - hit.y,
        0,
      );
    } else {
      dragOffsetRef.current.set(0, 0, 0);
    }
    setDragging(true);
    onHoldingChange(data.id);
    document.body.style.cursor = "grabbing";
  }, [data.id, data.position, onHoldingChange, onToggleSelect]);

  const stopHolding = useCallback((e?: any) => {
    if (e?.target && typeof e.target.releasePointerCapture === "function") {
      try {
        e.target.releasePointerCapture(e.pointerId);
      } catch {
        // No-op: capture might already be released.
      }
    }
    setDragging(false);
    onHoldingChange(null);
    document.body.style.cursor = hovered ? "grab" : "auto";
  }, [hovered, onHoldingChange]);

  return (
    <group
      position={data.position}
      quaternion={data.quaternion}
      scale={panelScale}
      onPointerDown={startHolding}
      onPointerUp={(e) => stopHolding(e)}
      onPointerCancel={(e) => stopHolding(e)}
      onPointerOut={() => {
        setHovered(false);
        if (!dragging) document.body.style.cursor = "auto";
      }}
      onPointerMove={(e) => {
        if (!dragging) return;
        e.stopPropagation();
        const hit = e.ray.intersectPlane(dragPlaneRef.current, intersectionRef.current);
        if (!hit) return;
        onMove(data.id, [
          hit.x + dragOffsetRef.current.x,
          hit.y + dragOffsetRef.current.y,
          data.position[2],
        ]);
      }}
      onPointerOver={(e) => {
        e.stopPropagation();
        setHovered(true);
        document.body.style.cursor = dragging ? "grabbing" : "grab";
      }}
      onClick={(e) => {
        // Stop the click from bubbling through to the underlying roof —
        // otherwise free-placement mode would aim at the panel and the user
        // would inadvertently pile a new panel on top of this one.
        e.stopPropagation();
        onToggleSelect(data.id);
      }}
    >
      <primitive object={clonedScene} />
    </group>
  );
}

// Red semi-transparent overlay for a selected roof area (one plane per cell)
function RoofHighlight({ candidate }: { candidate: Candidate }) {
  const { center, w, h } = useMemo(() => {
    const o = candidate.outline;
    const cx = o.reduce((s, p) => s + p[0], 0) / o.length;
    const cy = o.reduce((s, p) => s + p[1], 0) / o.length;
    const cz = o.reduce((s, p) => s + p[2], 0) / o.length;
    const dU = new THREE.Vector3(...o[1]).sub(new THREE.Vector3(...o[0])).length();
    const dV = new THREE.Vector3(...o[3]).sub(new THREE.Vector3(...o[0])).length();
    return { center: [cx, cy, cz] as [number, number, number], w: dU, h: dV };
  }, [candidate]);

  return (
    <mesh position={center} quaternion={candidate.quaternion}>
      <planeGeometry args={[w, h]} />
      <meshBasicMaterial
        color="hsl(0, 100%, 55%)"
        transparent
        opacity={0.45}
        side={THREE.DoubleSide}
        depthWrite={false}
      />
    </mesh>
  );
}

// Selectable slot used in "pick your own" mode
function SlotPicker({
  slot,
  selected,
  onToggle,
}: {
  slot: PanelData;
  selected: boolean;
  onToggle: (id: string) => void;
}) {
  const [hovered, setHovered] = useState(false);
  const color = selected
    ? "hsl(140, 100%, 50%)"
    : hovered
      ? "hsl(180, 100%, 70%)"
      : "hsl(180, 100%, 55%)";
  return (
    <mesh
      position={slot.position}
      quaternion={slot.quaternion}
      onClick={(e) => {
        e.stopPropagation();
        onToggle(slot.id);
      }}
      onPointerOver={(e) => {
        e.stopPropagation();
        setHovered(true);
        document.body.style.cursor = "pointer";
      }}
      onPointerOut={() => {
        setHovered(false);
        document.body.style.cursor = "auto";
      }}
    >
      <boxGeometry args={[slot.width, slot.height, selected ? 0.06 : 0.02]} />
      <meshStandardMaterial
        color={color}
        transparent
        opacity={selected ? 1 : 0.55}
        emissive={color}
        emissiveIntensity={selected ? 0.7 : 0.3}
        depthWrite={selected}
      />
    </mesh>
  );
}

// 3D holographic coordinate grid system
function CoordinateGrid({
  size = GRID_SIZE,
  divisions = GRID_DIVISIONS,
  center = [0, 0, 0] as [number, number, number],
  rotationZDeg = 0,
}: {
  size?: number;
  divisions?: number;
  center?: [number, number, number];
  rotationZDeg?: number;
}) {
  const half = size / 2;
  const step = size / divisions;

  const lines: [number, number, number, number, number, number][] = [];

  for (let y = -half; y <= half; y += step) {
    for (let z = 0; z <= half; z += step) {
      lines.push([-half, y, z, half, y, z]);
    }
  }

  for (let x = -half; x <= half; x += step) {
    for (let z = 0; z <= half; z += step) {
      lines.push([x, -half, z, x, half, z]);
    }
  }

  for (let x = -half; x <= half; x += step) {
    for (let y = -half; y <= half; y += step) {
      lines.push([x, y, 0, x, y, half]);
    }
  }

  return (
    <group position={center} rotation={[0, 0, (rotationZDeg * Math.PI) / 180]}>
      {lines.map((line, idx) => (
        <line key={`grid-${idx}`}>
          <bufferGeometry>
            <bufferAttribute
              attach="attributes-position"
              count={2}
              array={new Float32Array(line)}
              itemSize={3}
            />
          </bufferGeometry>
          <lineBasicMaterial color={0x00ff88} transparent opacity={0.75} linewidth={3} fog={false} />
        </line>
      ))}

      {/* X axis - red */}
      <line>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={2}
            array={new Float32Array([-half, 0, 0, half, 0, 0])}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial color={0xff0000} linewidth={4} fog={false} />
      </line>

      {/* Y axis - green */}
      <line>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={2}
            array={new Float32Array([0, -half, 0, 0, half, 0])}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial color={0x00ff00} linewidth={4} fog={false} />
      </line>

      {/* Z axis - blue */}
      <line>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={2}
            array={new Float32Array([0, 0, 0, 0, 0, half])}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial color={0x0000ff} linewidth={4} fog={false} />
      </line>

      {/* Ground plane center marker */}
      <mesh position={[0, 0, 0]}>
        <sphereGeometry args={[0.5, 8, 8]} />
        <meshBasicMaterial color={0xffffff} />
      </mesh>
    </group>
  );
}

// Red translucent plane used to preview the current selection height.
function SelectionHeightPlane({
  size = GRID_SIZE,
  center,
  rotationZDeg,
  heightOffset,
}: {
  size?: number;
  center: [number, number, number];
  rotationZDeg: number;
  heightOffset: number;
}) {
  return (
    <mesh
      position={[center[0], center[1], center[2] + heightOffset + 0.02]}
      rotation={[0, 0, (rotationZDeg * Math.PI) / 180]}
    >
      <planeGeometry args={[size, size]} />
      <meshBasicMaterial
        color="hsl(0, 100%, 55%)"
        transparent
        opacity={0.18}
        side={THREE.DoubleSide}
        depthWrite={false}
      />
    </mesh>
  );
}

// A single clickable cell tile for the manual selection layer
function SelectableGridCell({
  col,
  row,
  size,
  x,
  y,
  selected,
  onToggle,
}: {
  col: number;
  row: number;
  size: number;
  x: number;
  y: number;
  selected: boolean;
  onToggle: (key: string) => void;
}) {
  const [hovered, setHovered] = useState(false);
  const meshRef = useRef<THREE.Mesh>(null);
  const key = `${col},${row}`;

  useFrame(({ clock }) => {
    if (!meshRef.current) return;
    if (hovered && !selected) {
      const pulse = 1 + 0.06 * Math.sin(clock.elapsedTime * 7.5);
      meshRef.current.scale.set(pulse, pulse, 1);
    } else {
      meshRef.current.scale.set(1, 1, 1);
    }
  });

  return (
    <mesh
      ref={meshRef}
      position={[x, y, 0.015]}
      onClick={(e) => {
        e.stopPropagation();
        onToggle(key);
      }}
      onPointerOver={(e) => {
        e.stopPropagation();
        setHovered(true);
        document.body.style.cursor = "pointer";
      }}
      onPointerOut={() => {
        setHovered(false);
        document.body.style.cursor = "auto";
      }}
    >
      <planeGeometry args={[size - 0.15, size - 0.15]} />
      <meshBasicMaterial
        color={
          selected
            ? "hsl(0, 100%, 48%)"
            : hovered
              ? "hsl(190, 100%, 78%)"
              : "hsl(190, 100%, 55%)"
        }
        transparent
        opacity={selected ? 0.88 : hovered ? 0.52 : 0.12}
        side={THREE.DoubleSide}
        depthTest={false}
        depthWrite={false}
      />
      {selected && (
        <>
          <mesh position={[0, 0, 0.01]}>
            <planeGeometry args={[size - 0.03, size - 0.03]} />
            <meshBasicMaterial
              color="hsl(0, 100%, 25%)"
              transparent
              opacity={0.24}
              side={THREE.DoubleSide}
              depthTest={false}
              depthWrite={false}
            />
          </mesh>
          <lineLoop position={[0, 0, 0.02]}>
            <bufferGeometry>
              <bufferAttribute
                attach="attributes-position"
                count={4}
                array={new Float32Array([
                  -(size / 2) + 0.04, -(size / 2) + 0.04, 0,
                  (size / 2) - 0.04, -(size / 2) + 0.04, 0,
                  (size / 2) - 0.04, (size / 2) - 0.04, 0,
                  -(size / 2) + 0.04, (size / 2) - 0.04, 0,
                ])}
                itemSize={3}
              />
            </bufferGeometry>
            <lineBasicMaterial color="hsl(0, 100%, 65%)" transparent opacity={0.9} />
          </lineLoop>
        </>
      )}
    </mesh>
  );
}

// Full 25×25 grid of selectable cells at the chosen height
function SelectableGridLayer({
  size = GRID_SIZE,
  divisions = GRID_DIVISIONS,
  center,
  rotationZDeg,
  heightOffset,
  selectedCells,
  onToggleCell,
}: {
  size?: number;
  divisions?: number;
  center: [number, number, number];
  rotationZDeg: number;
  heightOffset: number;
  selectedCells: Set<string>;
  onToggleCell: (key: string) => void;
}) {
  const step = size / divisions;
  const half = size / 2;
  return (
    <group
      position={[center[0], center[1], center[2] + heightOffset]}
      rotation={[0, 0, (rotationZDeg * Math.PI) / 180]}
    >
      {Array.from({ length: divisions }, (_, row) =>
        Array.from({ length: divisions }, (_, col) => {
          const x = -half + (col + 0.5) * step;
          const y = -half + (row + 0.5) * step;
          const key = `${col},${row}`;
          return (
            <SelectableGridCell
              key={key}
              col={col}
              row={row}
              size={step}
              x={x}
              y={y}
              selected={selectedCells.has(key)}
              onToggle={onToggleCell}
            />
          );
        })
      )}
    </group>
  );
}

function SceneContent({
  model,
  panels,
  onRemovePanel,
  onMovePanel,
  onRotatePanel,
  selectedPanelIds,
  onTogglePanelSelect,
  onMovePanelsZ,
  highlights,
  pickingSlots,
  selectedSlots,
  onToggleSlot,
  showCoordinateGrid,
  gridRotationZDeg,
  gridPanX,
  gridPanY,
  gridHeightOffset,
  gridSelectionMode,
  selectedGridCells,
  onToggleGridCell,
  freePlacementMode,
  hoverPose,
  previewYawRad,
  onSnapHover,
  onSnapAim,
}: {
  model: THREE.Object3D | null;
  panels: PanelData[];
  onRemovePanel: (id: string) => void;
  onMovePanel: (id: string, position: [number, number, number]) => void;
  onRotatePanel: (id: string, quaternion: [number, number, number, number]) => void;
  selectedPanelIds: Set<string>;
  onTogglePanelSelect: (id: string) => void;
  onMovePanelsZ: (ids: Set<string>, deltaZ: number) => void;
  highlights: Candidate[];
  pickingSlots: PanelData[] | null;
  selectedSlots: Set<string>;
  onToggleSlot: (id: string) => void;
  showCoordinateGrid: boolean;
  gridRotationZDeg: number;
  gridPanX: number;
  gridPanY: number;
  gridHeightOffset: number;
  gridSelectionMode: boolean;
  selectedGridCells: Set<string>;
  onToggleGridCell: (key: string) => void;
  freePlacementMode: boolean;
  hoverPose: SnapPose | null;
  previewYawRad: number;
  onSnapHover: (pose: SnapPose | null) => void;
  /** Aiming = updating the locked target without committing a panel. */
  onSnapAim: (pose: SnapPose) => void;
}) {
  const { camera } = useThree();
  const modelGridOriginRef = useRef<[number, number, number]>([0, 0, 0]);
  const [heldPanelId, setHeldPanelId] = useState<string | null>(null);

  useEffect(() => {
    if (!model) return;
    const box = new THREE.Box3().setFromObject(model);
    const size = box.getSize(new THREE.Vector3()).length();
    const center = box.getCenter(new THREE.Vector3());
    modelGridOriginRef.current = [center.x, center.y, box.min.z];
    camera.position.set(center.x + size, center.y + size * 0.6, center.z + size);
    camera.lookAt(center);
    (camera as THREE.PerspectiveCamera).near = size / 100;
    (camera as THREE.PerspectiveCamera).far = size * 100;
    camera.updateProjectionMatrix();
  }, [model, camera]);

  const gridCenter: [number, number, number] = [
    modelGridOriginRef.current[0] + gridPanX,
    modelGridOriginRef.current[1] + gridPanY,
    modelGridOriginRef.current[2],
  ];

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (!heldPanelId) return;
      const active = panels.find((p) => p.id === heldPanelId);
      if (!active) return;

      if (event.key === "ArrowUp" || event.key === "ArrowDown") {
        event.preventDefault();
        const delta = event.key === "ArrowUp" ? 0.1 : -0.1;
        const idsToMove = selectedPanelIds.size > 0 && selectedPanelIds.has(heldPanelId)
          ? selectedPanelIds
          : new Set([heldPanelId]);
        onMovePanelsZ(idsToMove, delta);
      }

      if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
        event.preventDefault();
        const angle = THREE.MathUtils.degToRad(event.key === "ArrowLeft" ? 15 : -15);
        const q = new THREE.Quaternion(...active.quaternion);
        q.premultiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), angle));
        onRotatePanel(heldPanelId, [q.x, q.y, q.z, q.w]);
      }

      if (event.key === "Backspace") {
        event.preventDefault();
        onRemovePanel(heldPanelId);
        setHeldPanelId(null);
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [heldPanelId, selectedPanelIds, onMovePanelsZ, onRotatePanel, onRemovePanel, panels]);

  return (
    <>
      <ambientLight intensity={0.6} />
      <directionalLight position={[10, 20, 10]} intensity={1.2} castShadow />
      <directionalLight position={[-10, 10, -10]} intensity={0.4} />
      <Environment preset="city" />
      {model && (
        <ModelMesh
          object={model}
          snap={
            freePlacementMode
              ? { yawRad: previewYawRad, onHover: onSnapHover, onAim: onSnapAim }
              : undefined
          }
        />
      )}
      {freePlacementMode && hoverPose && <PreviewPanel pose={hoverPose} />}
      {panels.map((p) => (
        <Panel
          key={p.id}
          data={p}
          selected={selectedPanelIds.has(p.id)}
          onMove={onMovePanel}
          onHoldingChange={setHeldPanelId}
          onToggleSelect={onTogglePanelSelect}
        />
      ))}
      {highlights.map((c) => (
        <RoofHighlight key={c.id} candidate={c} />
      ))}
      {pickingSlots?.map((s) => (
        <SlotPicker
          key={s.id}
          slot={s}
          selected={selectedSlots.has(s.id)}
          onToggle={onToggleSlot}
        />
      ))}
      {showCoordinateGrid && (
        <>
          <CoordinateGrid
            size={150}
            divisions={25}
            center={gridCenter}
            rotationZDeg={gridRotationZDeg}
          />
          <SelectionHeightPlane
            size={150}
            center={gridCenter}
            rotationZDeg={gridRotationZDeg}
            heightOffset={gridHeightOffset}
          />
        </>
      )}
      {gridSelectionMode && (
        <SelectableGridLayer
          center={gridCenter}
          rotationZDeg={gridRotationZDeg}
          heightOffset={gridHeightOffset}
          selectedCells={selectedGridCells}
          onToggleCell={onToggleGridCell}
        />
      )}
      {!model && (
        <Grid
          args={[20, 20]}
          cellColor="hsl(220, 15%, 70%)"
          sectionColor="hsl(220, 30%, 50%)"
          fadeDistance={30}
          infiniteGrid
        />
      )}
      <OrbitControls makeDefault enableDamping enabled={!heldPanelId} />
    </>
  );
}

// Converts selected grid cell keys to Candidate[] for the placement flow
function gridCellsToHighlights(
  cells: Set<string>,
  gridCX: number,
  gridCY: number,
  gridCZ: number,
  rotationZDeg: number,
  heightOffset: number,
): Candidate[] {
  const step = GRID_SIZE / GRID_DIVISIONS;
  const half = GRID_SIZE / 2;
  const angle = (rotationZDeg * Math.PI) / 180;
  const cosA = Math.cos(angle);
  const sinA = Math.sin(angle);
  const q = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0, angle));
  const quat: [number, number, number, number] = [q.x, q.y, q.z, q.w];
  const worldZ = gridCZ + heightOffset + 0.08;

  const panelCols = Math.floor((step + PANEL_GAP) / (PANEL_W + PANEL_GAP));
  const panelRows = Math.floor((step + PANEL_GAP) / (PANEL_H + PANEL_GAP));
  const totalW = panelCols * PANEL_W + (panelCols - 1) * PANEL_GAP;
  const totalH = panelRows * PANEL_H + (panelRows - 1) * PANEL_GAP;
  const startU = -totalW / 2 + PANEL_W / 2;
  const startV = -totalH / 2 + PANEL_H / 2;

  return Array.from(cells).map((key) => {
    const [c, r] = key.split(",").map(Number);
    const lx = -half + (c + 0.5) * step;
    const ly = -half + (r + 0.5) * step;
    const wx = gridCX + lx * cosA - ly * sinA;
    const wy = gridCY + lx * sinA + ly * cosA;

    const slots: PanelData[] = [];
    for (let ri = 0; ri < panelRows; ri++) {
      for (let ci = 0; ci < panelCols; ci++) {
        const u = startU + ci * (PANEL_W + PANEL_GAP);
        const v = startV + ri * (PANEL_H + PANEL_GAP);
        slots.push({
          id: crypto.randomUUID(),
          position: [wx + u * cosA - v * sinA, wy + u * sinA + v * cosA, worldZ],
          quaternion: quat,
          width: PANEL_W,
          height: PANEL_H,
        });
      }
    }

    const hw = step / 2;
    const outline: [number, number, number][] = (
      [[-hw, -hw], [hw, -hw], [hw, hw], [-hw, hw]] as [number, number][]
    ).map(([u, v]) => [wx + u * cosA - v * sinA, wy + u * sinA + v * cosA, worldZ]);

    return { id: `cell-${c}-${r}`, quaternion: quat, slots, outline, height: worldZ };
  });
}

type PlacementMode = null | "choose" | "picking";

export default function SolarViewer() {
  const [model, setModel] = useState<THREE.Object3D | null>(null);
  const [panels, setPanels] = useState<PanelData[]>([]);
  const [dragOver, setDragOver] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [highlights, setHighlights] = useState<Candidate[]>([]);
  const [showHighlights, setShowHighlights] = useState(false);
  const [placementMode, setPlacementMode] = useState<PlacementMode>(null);
  const [selectedSlots, setSelectedSlots] = useState<Set<string>>(new Set());
  const [showCoordinateGrid, setShowCoordinateGrid] = useState(false);
  const [gridRotationZDeg, setGridRotationZDeg] = useState(0);
  const [gridPanX, setGridPanX] = useState(0);
  const [gridPanY, setGridPanY] = useState(0);
  const [gridHeightOffset, setGridHeightOffset] = useState(0);
  const [gridRotationInput, setGridRotationInput] = useState("0");
  const [gridPanXInput, setGridPanXInput] = useState("0");
  const [gridPanYInput, setGridPanYInput] = useState("0");
  const [gridHeightInput, setGridHeightInput] = useState("0");
  const [gridConfigured, setGridConfigured] = useState(false);
  const [modelGroundZ, setModelGroundZ] = useState(0);
  const [modelCenterXY, setModelCenterXY] = useState<[number, number]>([0, 0]);
  const [gridSelectionMode, setGridSelectionMode] = useState(false);
  const [selectedGridCells, setSelectedGridCells] = useState<Set<string>>(new Set());
  const [recommendation, setRecommendation] = useState<DesignFromPromptResponse | null>(null);
  const [freePlacementMode, setFreePlacementMode] = useState(false);
  const [hoverPose, setHoverPose] = useState<SnapPose | null>(null);
  // Last seen hover pose — survives onPointerOut so the Propose button stays
  // clickable after you move the cursor off the canvas toward the sidebar.
  const [lastHoverPose, setLastHoverPose] = useState<SnapPose | null>(null);
  const [previewYawDeg, setPreviewYawDeg] = useState(0);
  // Estimated capacity of the roof patch around `lastHoverPose`, computed by
  // running a generous flood-fill against the BVH. null until measured.
  const [roofCapacity, setRoofCapacity] = useState<{
    count: number;
    /** Roof patch area available for placement (panel footprint + gap, ×count). */
    roofArea: number;
    /** Sum of panel footprints — what the panels themselves occupy. */
    panelCoverage: number;
  } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const addingMoreRef = useRef(false);

  const [selectedPanelIds, setSelectedPanelIds] = useState<Set<string>>(new Set());

  const togglePanelSelect = useCallback((id: string) => {
    setSelectedPanelIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setSelectedPanelIds(new Set());
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // Aim handler: a click on the model just locks the target without placing.
  // Lets the user nudge / rotate / verify the preview before pressing Confirm.
  const aimAtPose = useCallback((pose: SnapPose) => {
    setLastHoverPose(pose);
  }, []);

  // Commit a panel at the current locked target. Wired to the Confirm button
  // and to the Enter key while free-placement mode is active.
  const commitPlacement = useCallback(() => {
    if (!lastHoverPose) return;
    const pose = lastHoverPose;
    const modelKey = pickPanelModel(new THREE.Vector3(...pose.worldNormal));
    const def = PANEL_MODELS[modelKey];
    setPanels((prev) => [
      ...prev,
      {
        id: crypto.randomUUID(),
        position: pose.position,
        quaternion: pose.quaternion,
        width: def.width,
        height: def.height,
        depth: def.depth,
        modelKey,
      },
    ]);
  }, [lastHoverPose]);

  // Auto-layout: uses the current hover pose as the seed and grows a parallel
  // panel grid outward, sampling the BVH to skip non-roof cells. Replaces the
  // current panel set so the user can iterate on different seed points.
  // Helper: run the flood-fill once and convert poses into PanelData with the
  // right per-cell model. Returns the data plus a capacity summary.
  //
  // Gating chosen to stick to the seed's roof patch:
  //   - maxNormalDeg 12°    → won't bleed onto a neighbouring slope or wall
  //   - maxPlaneOffset 0.5m → handles photogrammetry noise but rejects ground
  //   - maxJump derived from cell step (auto), so legit neighbours always pass
  const buildLayoutFromSeed = useCallback(
    (seed: SnapPose, panelLimit: number, modelRoot: THREE.Object3D) => {
      const gap = 0.2;
      const cellArea = (PANEL_MODELS.flat.width + gap) * (PANEL_MODELS.flat.height + gap);
      const poses = proposeLayout(seed, panelLimit, modelRoot, {
        gap,
        maxNormalDeg: 12,
        maxPlaneOffset: 0.5,
        // maxJump intentionally omitted — proposeLayout defaults it from the cell step.
      });
      let panelCoverage = 0; // sum of panel footprints (real solar area)
      const data = poses.map<PanelData>((pose) => {
        const modelKey = pickPanelModel(new THREE.Vector3(...pose.worldNormal));
        const def = PANEL_MODELS[modelKey];
        panelCoverage += def.width * def.height;
        return {
          id: crypto.randomUUID(),
          position: pose.position,
          quaternion: pose.quaternion,
          width: def.width,
          height: def.height,
          depth: def.depth,
          modelKey,
        };
      });
      return {
        data,
        count: poses.length,
        // Roof patch area available for placement: each cell occupies its full
        // step (panel + gap), and that's the chunk of roof reserved per panel.
        roofArea: poses.length * cellArea,
        // What the panels actually cover (excluding gaps).
        panelCoverage,
      };
    },
    [],
  );

  // Measure: flood-fill with a generous cap (1000 panels) so we know how many
  // would fit on this roof patch from the chosen seed. Cached so the user can
  // see remaining capacity as they place panels.
  const measureRoof = useCallback(() => {
    if (!model) return;
    if (!lastHoverPose) {
      setError("Hover over the roof first, then press Measure.");
      return;
    }
    const { count, roofArea, panelCoverage } = buildLayoutFromSeed(lastHoverPose, 1000, model);
    if (count === 0) {
      setError("Couldn't find any panel slots from that point. Try a flatter spot on the roof.");
      setRoofCapacity(null);
      return;
    }
    setRoofCapacity({ count, roofArea, panelCoverage });
    setError(null);
  }, [model, lastHoverPose, buildLayoutFromSeed]);

  const proposeDesign = useCallback(() => {
    if (!model || !recommendation) {
      setError("Load a model and run a prompt first.");
      return;
    }
    if (!lastHoverPose) {
      setError("Hover over the roof in free-placement mode, then press Propose.");
      return;
    }
    const target = recommendation.design.panels_needed;
    // Run a generous fill first (so we learn capacity), then keep the closest N.
    const full = buildLayoutFromSeed(lastHoverPose, 1000, model);
    if (full.count === 0) {
      setError("Couldn't fit panels around that point. Try a flatter spot on the roof.");
      return;
    }
    setRoofCapacity({
      count: full.count,
      roofArea: full.roofArea,
      panelCoverage: full.panelCoverage,
    });
    const placed = full.data.slice(0, target);
    setPanels(placed);
    setError(
      placed.length < target
        ? `Roof fits ${full.count} panels here — placed ${placed.length} of the ${target} target.`
        : null,
    );
  }, [model, recommendation, lastHoverPose, buildLayoutFromSeed]);

  // Q / E rotate preview yaw, R resets, Enter commits at the locked target.
  // Only fires in free-placement mode; ignored when typing in an input.
  useEffect(() => {
    if (!freePlacementMode) return;
    const onKey = (event: KeyboardEvent) => {
      const t = event.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
      const k = event.key.toLowerCase();
      if (k === "q") {
        event.preventDefault();
        setPreviewYawDeg((d) => (d - 15 + 360) % 360);
      } else if (k === "e") {
        event.preventDefault();
        setPreviewYawDeg((d) => (d + 15) % 360);
      } else if (k === "r") {
        event.preventDefault();
        setPreviewYawDeg(0);
      } else if (event.key === "Enter") {
        event.preventDefault();
        commitPlacement();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [freePlacementMode, commitPlacement]);

  const snapHeightOffset = useCallback((value: number) => {
    const maxHeight = GRID_SIZE / 2;
    const clamped = Math.max(-maxHeight, Math.min(maxHeight, value));
    return Math.round(clamped / GRID_CELL_HEIGHT) * GRID_CELL_HEIGHT;
  }, []);

  useEffect(() => {
    setGridRotationInput(String(gridRotationZDeg));
  }, [gridRotationZDeg]);

  useEffect(() => {
    setGridPanXInput(String(gridPanX));
  }, [gridPanX]);

  useEffect(() => {
    setGridPanYInput(String(gridPanY));
  }, [gridPanY]);

  useEffect(() => {
    setGridHeightInput(String(gridHeightOffset));
  }, [gridHeightOffset]);

  const handleGridControlChange = useCallback(() => {
    setGridConfigured(false);
    setShowHighlights(false);
    setHighlights([]);
    setSelectedGridCells(new Set());
    setGridSelectionMode(false);
    setError(null);
  }, []);

  const loadGlb = useCallback(async (file: File) => {
    setLoading(true);
    setError(null);
    setPanels([]);
    setHighlights([]);
    setShowHighlights(false);
    setGridRotationZDeg(0);
    setGridPanX(0);
    setGridPanY(0);
    setGridHeightOffset(0);
    setGridConfigured(false);
    setGridSelectionMode(false);
    setSelectedGridCells(new Set());
    setPlacementMode(null);
    setSelectedSlots(new Set());
    setFreePlacementMode(false);
    setHoverPose(null);
    setLastHoverPose(null);
    setRoofCapacity(null);
    try {
      const arrayBuffer = await file.arrayBuffer();
      const gltf = await gltfLoader.parseAsync(arrayBuffer, "");
      gltf.scene.traverse((o) => {
        const m = o as THREE.Mesh;
        if (m.isMesh) {
          m.castShadow = true;
          m.receiveShadow = true;
        }
      });
      // BVH-accelerate raycasts (R3F pointer events) on the high-poly mesh.
      indexModelForBVH(gltf.scene);
      const box = new THREE.Box3().setFromObject(gltf.scene);
      const center = box.getCenter(new THREE.Vector3());
      setModelGroundZ(box.min.z);
      setModelCenterXY([center.x, center.y]);
      setModel(gltf.scene);
      setFileName(file.name);
    } catch (e) {
      console.error(e);
      setError("Could not load this file. Make sure it is a valid .glb model.");
    } finally {
      setLoading(false);
    }
  }, []);

  const handleFiles = useCallback(
    (files: FileList | null) => {
      if (!files || files.length === 0) return;
      const file = files[0];
      if (!file.name.toLowerCase().endsWith(".glb")) {
        setError("Only .glb files are supported.");
        return;
      }
      loadGlb(file);
    },
    [loadGlb]
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      handleFiles(e.dataTransfer.files);
    },
    [handleFiles]
  );

  const placeSelectedAreas = useCallback((sourceHighlights?: Candidate[]) => {
    const areas = sourceHighlights ?? highlights;
    if (areas.length === 0) return;
    const cellSize = GRID_SIZE / GRID_DIVISIONS;
    const boxDepth = cellSize * 0.95;
    const cellPanels = areas.map((c) => {
      const center: [number, number, number] = [
        c.outline.reduce((sum, p) => sum + p[0], 0) / c.outline.length,
        c.outline.reduce((sum, p) => sum + p[1], 0) / c.outline.length,
        c.outline.reduce((sum, p) => sum + p[2], 0) / c.outline.length + boxDepth / 2,
      ];
      return {
        id: crypto.randomUUID(),
        position: center,
        quaternion: c.quaternion,
        width: cellSize - 0.08,
        height: cellSize - 0.08,
        depth: boxDepth,
      } as PanelData;
    });
    if (addingMoreRef.current) {
      setPanels((prev) => [...prev, ...cellPanels]);
      addingMoreRef.current = false;
    } else {
      setPanels(cellPanels);
    }
    setPlacementMode(null);
    setShowHighlights(false);
    setSelectedSlots(new Set());
    setError(null);
  }, [highlights]);

  const confirmGridSelection = useCallback(() => {
    if (selectedGridCells.size === 0) return;
    const h = gridCellsToHighlights(
      selectedGridCells,
      modelCenterXY[0] + gridPanX,
      modelCenterXY[1] + gridPanY,
      modelGroundZ,
      gridRotationZDeg,
      gridHeightOffset,
    );
    setHighlights(h);
    setGridSelectionMode(false);
    setShowCoordinateGrid(false);
    setError(null);
    placeSelectedAreas(h);
  }, [
    selectedGridCells,
    modelCenterXY,
    gridPanX,
    gridPanY,
    modelGroundZ,
    gridRotationZDeg,
    gridHeightOffset,
    placeSelectedAreas,
  ]);

  const toggleGridCell = useCallback((key: string) => {
    setSelectedGridCells((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }, []);

  const openPlacement = useCallback(() => {
    if (!model) return;
    if (!gridConfigured) {
      setError("Set the grid first, then click Place Solar Panels.");
      return;
    }
    if (highlights.length === 0) {
      setShowCoordinateGrid(true);
      setSelectedGridCells(new Set());
      setGridSelectionMode(true);
      setError("Select roof cells, then click Confirm.");
      return;
    }
    placeSelectedAreas();
  }, [model, gridConfigured, highlights, placeSelectedAreas]);

  const openAddMore = useCallback(() => {
    if (!model || !gridConfigured) return;
    addingMoreRef.current = true;
    setHighlights([]);
    setShowHighlights(false);
    setShowCoordinateGrid(true);
    setSelectedGridCells(new Set());
    setGridSelectionMode(true);
    setError("Select roof cells to add more panels, then click Confirm.");
  }, [model, gridConfigured]);

  const toggleSlot = useCallback((id: string) => {
    setSelectedSlots((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const confirmPicked = useCallback(() => {
    const all = highlights.flatMap((c) => c.slots);
    const chosen = all.filter((s) => selectedSlots.has(s.id));
    setPanels(chosen.map((s) => ({ ...s, id: crypto.randomUUID() })));
    setPlacementMode(null);
    setSelectedSlots(new Set());
  }, [highlights, selectedSlots]);

  const cancelPicking = useCallback(() => {
    setPlacementMode(null);
    setSelectedSlots(new Set());
  }, []);

  const removePanel = useCallback((id: string) => {
    setPanels((prev) => prev.filter((p) => p.id !== id));
  }, []);

  const movePanel = useCallback((id: string, position: [number, number, number]) => {
    setPanels((prev) => {
      const moving = prev.find((p) => p.id === id);
      if (!moving) return prev;
      const candidate = { ...moving, position };
      if (prev.some((p) => p.id !== id && panelsOverlap(candidate, p))) return prev;
      return prev.map((p) => (p.id === id ? candidate : p));
    });
  }, []);

  const rotatePanel = useCallback((id: string, quaternion: [number, number, number, number]) => {
    setPanels((prev) => {
      const panel = prev.find((p) => p.id === id);
      if (!panel) return prev;
      const candidate = { ...panel, quaternion };
      if (prev.some((p) => p.id !== id && panelsOverlap(candidate, p))) return prev;
      return prev.map((p) => (p.id === id ? candidate : p));
    });
  }, []);

  const movePanelsZ = useCallback((ids: Set<string>, deltaZ: number) => {
    setPanels((prev) => prev.map((p) =>
      ids.has(p.id)
        ? { ...p, position: [p.position[0], p.position[1], p.position[2] + deltaZ] }
        : p
    ));
  }, []);

  const clearPanels = useCallback(() => {
    setPanels([]);
    setHighlights([]);
    setShowHighlights(false);
    setSelectedGridCells(new Set());
    setGridSelectionMode(false);
    setSelectedSlots(new Set());
    setSelectedPanelIds(new Set());
    setPlacementMode(null);
    setHoverPose(null);
    setError(null);
  }, []);

  const stats = useMemo(() => {
    // Bucket placed panels by model so the sidebar can show truthful
    // count / area / cost per type — flat and slanted carry different prices.
    const byModel: Record<PanelModelKey, { count: number; area: number; cost: number }> = {
      flat: { count: 0, area: 0, cost: 0 },
      slanted: { count: 0, area: 0, cost: 0 },
    };
    for (const p of panels) {
      const key = p.modelKey ?? "flat";
      const def = PANEL_MODELS[key];
      byModel[key].count += 1;
      byModel[key].area += p.width * p.height;
      byModel[key].cost += def.pricePerUnit;
    }
    return {
      count: panels.length,
      area: byModel.flat.area + byModel.slanted.area,
      cost: byModel.flat.cost + byModel.slanted.cost,
      byModel,
    };
  }, [panels]);

  const pickingSlots = placementMode === "picking"
    ? highlights.flatMap((c) => c.slots)
    : null;

  return (
    <div className="flex h-screen w-full bg-background text-foreground">
      {/* Viewer */}
      <div
        className="relative flex-1"
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
      >
        <Canvas
          shadows
          camera={{ position: [5, 5, 5], fov: 50 }}
          className="!absolute inset-0"
        >
          <Suspense fallback={null}>
            <SceneContent
              model={model}
              panels={panels}
              onRemovePanel={removePanel}
              onMovePanel={movePanel}
              onRotatePanel={rotatePanel}
              selectedPanelIds={selectedPanelIds}
              onTogglePanelSelect={togglePanelSelect}
              onMovePanelsZ={movePanelsZ}
              highlights={showHighlights ? highlights : []}
              pickingSlots={pickingSlots}
              selectedSlots={selectedSlots}
              onToggleSlot={toggleSlot}
              showCoordinateGrid={showCoordinateGrid}
              gridRotationZDeg={gridRotationZDeg}
              gridPanX={gridPanX}
              gridPanY={gridPanY}
              gridHeightOffset={gridHeightOffset}
              gridSelectionMode={gridSelectionMode}
              selectedGridCells={selectedGridCells}
              onToggleGridCell={toggleGridCell}
              freePlacementMode={freePlacementMode}
              hoverPose={hoverPose}
              previewYawRad={THREE.MathUtils.degToRad(previewYawDeg)}
              onSnapHover={(p) => {
                setHoverPose(p);
                if (p) setLastHoverPose(p);
              }}
              onSnapAim={aimAtPose}
            />
          </Suspense>
        </Canvas>

        {/* Drop overlay */}
        {(!model || dragOver) && (
          <div
            className={cn(
              "pointer-events-none absolute inset-0 flex items-center justify-center transition-colors",
              dragOver ? "bg-primary/10" : "bg-transparent"
            )}
          >
            {!model && (
              <Card className="pointer-events-auto flex flex-col items-center gap-4 border-2 border-dashed border-primary/40 bg-card/90 p-10 backdrop-blur">
                <Upload className="h-10 w-10 text-primary" />
                <div className="text-center">
                  <p className="text-lg font-semibold">Drop a .glb model here</p>
                  <p className="text-sm text-muted-foreground">or use the button to browse files</p>
                </div>
                <Button onClick={() => fileInputRef.current?.click()} disabled={loading}>
                  {loading ? "Loading..." : "Choose .glb file"}
                </Button>
              </Card>
            )}
          </div>
        )}

        <input
          ref={fileInputRef}
          type="file"
          accept=".glb"
          className="hidden"
          onChange={(e) => handleFiles(e.target.files)}
        />

        {/* Live placement HUD — only while free-placement is active and we have a target. */}
        {freePlacementMode && recommendation && (
          <PlacementHUD
            panelsPlaced={panels.length}
            targetPanels={recommendation.design.panels_needed}
            pricePerPanel={PRICE_PER_PANEL}
            baseCost={recommendation.design.recommended_battery_kwh * PRICE_PER_KWH_BATTERY}
          />
        )}

        {/* Top-left filename badge */}
        {fileName && (
          <div className="absolute left-4 top-4 rounded-md bg-card/80 px-3 py-1.5 text-sm shadow backdrop-blur">
            <span className="text-muted-foreground">Model:</span>{" "}
            <span className="font-medium">{fileName}</span>
          </div>
        )}

        {/* Picking mode toolbar */}
        {placementMode === "picking" && (
          <div className="absolute bottom-6 left-1/2 flex -translate-x-1/2 items-center gap-3 rounded-lg border border-border bg-card/95 px-4 py-3 shadow-lg backdrop-blur">
            <span className="text-sm">
              <span className="font-semibold text-primary">{selectedSlots.size}</span>{" "}
              slot{selectedSlots.size === 1 ? "" : "s"} selected
            </span>
            <Button size="sm" variant="ghost" onClick={cancelPicking}>
              Cancel
            </Button>
            <Button size="sm" onClick={confirmPicked} disabled={selectedSlots.size === 0}>
              Confirm placement
            </Button>
          </div>
        )}

        {error && (
          <div className="absolute bottom-4 left-1/2 -translate-x-1/2 rounded-md bg-destructive px-4 py-2 text-sm text-destructive-foreground shadow">
            {error}
          </div>
        )}

        {panels.length > 0 && (
          <div className="pointer-events-none absolute right-4 top-4 max-w-xs rounded-md border border-border bg-card/90 p-3 text-xs shadow-lg backdrop-blur">
            <p className="mb-1 font-semibold text-foreground">Panel controls</p>
            <p className="text-muted-foreground">
              Click and drag a panel to move on X/Y. While holding it: Arrow Up/Down to move on
              Z, Arrow Left/Right to rotate 15°, Backspace to delete.
            </p>
          </div>
        )}
      </div>

      {/* Sidebar */}
      <aside className="flex w-80 flex-col overflow-y-auto border-l border-border bg-card">
        <div className="border-b border-border p-5">
          <h1 className="flex items-center gap-2 text-xl font-bold">
            <Sun className="h-5 w-5 text-primary" />
            Solar Studio
          </h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Plan rooftop solar installations on any 3D model.
          </p>
        </div>

        <div className="border-b border-border p-5">
          <DesignFromPrompt onResult={setRecommendation} />
        </div>

        <div className="space-y-3 p-5">
          {/* Grid selection mode banner */}
          {gridSelectionMode && (
            <div className="rounded-md border border-primary bg-primary/10 p-3 text-sm">
              <p className="mb-1 font-medium">Click grid cells on the roof</p>
              <p className="mb-3 text-xs text-muted-foreground">
                {selectedGridCells.size} cell{selectedGridCells.size === 1 ? "" : "s"} selected
              </p>
              <div className="flex gap-2">
                <Button
                  size="sm"
                  className="flex-1"
                  onClick={confirmGridSelection}
                  disabled={selectedGridCells.size === 0}
                >
                  Confirm
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => setSelectedGridCells(new Set())}
                >
                  Clear
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => {
                    setGridSelectionMode(false);
                    setSelectedGridCells(new Set());
                  }}
                >
                  Cancel
                </Button>
              </div>
            </div>
          )}

          <Button
            variant="outline"
            className="w-full"
            onClick={() => setShowCoordinateGrid(!showCoordinateGrid)}
            disabled={!model || loading}
          >
            <Grid3x3 className="mr-2 h-4 w-4" />
            {showCoordinateGrid ? "Hide coordinate grid" : "Show coordinate grid"}
          </Button>

          {showCoordinateGrid && (
            <div className="space-y-2 rounded-md border border-border bg-background/60 p-3">
              <div className="flex items-center justify-between text-xs">
                <span className="font-medium text-muted-foreground">Grid rotation (Z)</span>
                <span className="font-semibold text-foreground">{gridRotationZDeg.toFixed(1)}°</span>
              </div>
              <input
                type="range"
                min={-180}
                max={180}
                step={0.1}
                value={gridRotationZDeg}
                onChange={(e) => {
                  setGridRotationZDeg(Number(e.target.value));
                  handleGridControlChange();
                }}
                className="w-full accent-primary"
              />
              <input
                type="number"
                min={-180}
                max={180}
                step={0.1}
                value={gridRotationInput}
                onChange={(e) => {
                  const raw = e.target.value;
                  setGridRotationInput(raw);
                  if (raw === "" || raw === "-" || raw === "." || raw === "-.") return;
                  const next = Number(raw);
                  if (Number.isNaN(next)) return;
                  setGridRotationZDeg(next);
                  handleGridControlChange();
                }}
                onBlur={() => {
                  const next = Number(gridRotationInput);
                  if (Number.isNaN(next)) {
                    setGridRotationInput(String(gridRotationZDeg));
                    return;
                  }
                  const clamped = Math.max(-180, Math.min(180, next));
                  setGridRotationZDeg(clamped);
                }}
                className="w-full rounded-md border border-border bg-background px-2 py-1 text-xs"
              />

              <div className="flex items-center justify-between text-xs">
                <span className="font-medium text-muted-foreground">Grid pan X</span>
                <span className="font-semibold text-foreground">{gridPanX.toFixed(1)} m</span>
              </div>
              <input
                type="range"
                min={-100}
                max={100}
                step={0.1}
                value={gridPanX}
                onChange={(e) => {
                  setGridPanX(Number(e.target.value));
                  handleGridControlChange();
                }}
                className="w-full accent-primary"
              />
              <input
                type="number"
                min={-100}
                max={100}
                step={0.1}
                value={gridPanXInput}
                onChange={(e) => {
                  const raw = e.target.value;
                  setGridPanXInput(raw);
                  if (raw === "" || raw === "-" || raw === "." || raw === "-.") return;
                  const next = Number(raw);
                  if (Number.isNaN(next)) return;
                  setGridPanX(next);
                  handleGridControlChange();
                }}
                onBlur={() => {
                  const next = Number(gridPanXInput);
                  if (Number.isNaN(next)) {
                    setGridPanXInput(String(gridPanX));
                    return;
                  }
                  const clamped = Math.max(-100, Math.min(100, next));
                  setGridPanX(clamped);
                }}
                className="w-full rounded-md border border-border bg-background px-2 py-1 text-xs"
              />

              <div className="flex items-center justify-between text-xs">
                <span className="font-medium text-muted-foreground">Grid pan Y</span>
                <span className="font-semibold text-foreground">{gridPanY.toFixed(1)} m</span>
              </div>
              <input
                type="range"
                min={-100}
                max={100}
                step={0.1}
                value={gridPanY}
                onChange={(e) => {
                  setGridPanY(Number(e.target.value));
                  handleGridControlChange();
                }}
                className="w-full accent-primary"
              />
              <input
                type="number"
                min={-100}
                max={100}
                step={0.1}
                value={gridPanYInput}
                onChange={(e) => {
                  const raw = e.target.value;
                  setGridPanYInput(raw);
                  if (raw === "" || raw === "-" || raw === "." || raw === "-.") return;
                  const next = Number(raw);
                  if (Number.isNaN(next)) return;
                  setGridPanY(next);
                  handleGridControlChange();
                }}
                onBlur={() => {
                  const next = Number(gridPanYInput);
                  if (Number.isNaN(next)) {
                    setGridPanYInput(String(gridPanY));
                    return;
                  }
                  const clamped = Math.max(-100, Math.min(100, next));
                  setGridPanY(clamped);
                }}
                className="w-full rounded-md border border-border bg-background px-2 py-1 text-xs"
              />

              <div className="flex items-center justify-between text-xs">
                <span className="font-medium text-muted-foreground">Selection height</span>
                <span className="font-semibold text-foreground">{gridHeightOffset.toFixed(1)} m</span>
              </div>
              <input
                type="range"
                min={-GRID_SIZE / 2}
                max={GRID_SIZE / 2}
                step={GRID_CELL_HEIGHT}
                value={gridHeightOffset}
                onChange={(e) => {
                  setGridHeightOffset(snapHeightOffset(Number(e.target.value)));
                  handleGridControlChange();
                }}
                className="w-full accent-primary"
              />
              <input
                type="number"
                min={-GRID_SIZE / 2}
                max={GRID_SIZE / 2}
                step={GRID_CELL_HEIGHT}
                value={gridHeightInput}
                onChange={(e) => {
                  const raw = e.target.value;
                  setGridHeightInput(raw);
                  if (raw === "" || raw === "-" || raw === "." || raw === "-.") return;
                  const next = Number(raw);
                  if (Number.isNaN(next)) return;
                  setGridHeightOffset(snapHeightOffset(next));
                  handleGridControlChange();
                }}
                onBlur={() => {
                  const next = Number(gridHeightInput);
                  if (Number.isNaN(next)) {
                    setGridHeightInput(String(gridHeightOffset));
                    return;
                  }
                  setGridHeightOffset(snapHeightOffset(next));
                }}
                className="w-full rounded-md border border-border bg-background px-2 py-1 text-xs"
              />

              <Button
                type="button"
                size="sm"
                className="w-full"
                disabled={!model || loading}
                onClick={() => {
                  setGridConfigured(true);
                  setShowCoordinateGrid(false);
                  setGridSelectionMode(false);
                  setError(null);
                }}
              >
                <Sparkles className="mr-2 h-3 w-3" />
                {gridConfigured ? "Grid Set" : "Set Grid"}
              </Button>

              <Button
                type="button"
                size="sm"
                variant="ghost"
                className="w-full"
                onClick={() => {
                  setGridRotationZDeg(0);
                  setGridPanX(0);
                  setGridPanY(0);
                  setGridHeightOffset(0);
                  handleGridControlChange();
                }}
              >
                Reset all grid controls
              </Button>
            </div>
          )}

          <Button
            className="w-full"
            onClick={openPlacement}
            disabled={!model || loading || placementMode === "picking" || !gridConfigured}
          >
            <Sun className="mr-2 h-4 w-4" />
            Place Solar Panels
          </Button>

          <Button
            variant={freePlacementMode ? "default" : "outline"}
            className="w-full"
            onClick={() => {
              setFreePlacementMode((v) => !v);
              setHoverPose(null);
              setLastHoverPose(null);
              setRoofCapacity(null);
              setPreviewYawDeg(0);
            }}
            disabled={!model || loading}
          >
            <MousePointerClick className="mr-2 h-4 w-4" />
            {freePlacementMode ? "Stop free placement" : "Free placement (hover & click)"}
          </Button>

          {freePlacementMode && (
            <p className="rounded-md border border-border bg-background/50 px-3 py-2 text-[11px] text-muted-foreground">
              Hover or click on the roof to aim, then press{" "}
              <span className="font-medium text-foreground">Enter</span> or
              {" "}<span className="font-medium text-foreground">Confirm</span> to drop a panel.
              {" "}<span className="font-medium text-foreground">Q / E</span> rotate ±15°,
              {" "}<span className="font-medium text-foreground">R</span> reset.
              Drag a placed panel; hold it and use{" "}
              <span className="font-medium text-foreground">←/→</span> to rotate,{" "}
              <span className="font-medium text-foreground">↑/↓</span> to lift,{" "}
              <span className="font-medium text-foreground">Backspace</span> to delete.
              {" "}Yaw: <span className="font-mono text-foreground">{previewYawDeg}°</span>
            </p>
          )}

          {freePlacementMode && (
            <Button
              variant="default"
              className="w-full"
              onClick={commitPlacement}
              disabled={!model || loading || !lastHoverPose}
            >
              <Sun className="mr-2 h-4 w-4" />
              Confirm placement (Enter)
            </Button>
          )}

          <Button
            variant="outline"
            className="w-full"
            onClick={measureRoof}
            disabled={!model || loading || !lastHoverPose}
          >
            <Ruler className="mr-2 h-4 w-4" />
            {roofCapacity
              ? `Roof patch: ${roofCapacity.count} panels · ${roofCapacity.roofArea.toFixed(1)} m²`
              : "Measure roof from this spot"}
          </Button>

          <Button
            variant="secondary"
            className="w-full"
            onClick={proposeDesign}
            disabled={!model || loading || !recommendation || !lastHoverPose}
          >
            <Sparkles className="mr-2 h-4 w-4" />
            Propose a design ({recommendation?.design.panels_needed ?? "?"} panels)
          </Button>

          <Button
            variant="outline"
            className="w-full"
            onClick={openAddMore}
            disabled={panels.length === 0 || !gridConfigured || loading || placementMode === "picking"}
          >
            <Sun className="mr-2 h-4 w-4" />
            Add more solar panels
          </Button>
          <Button
            variant="outline"
            className="w-full"
            onClick={() => fileInputRef.current?.click()}
          >
            <Upload className="mr-2 h-4 w-4" />
            Load .glb file
          </Button>
          {panels.length > 0 && (
            <Button variant="ghost" className="w-full" onClick={clearPanels}>
              <Trash2 className="mr-2 h-4 w-4" />
              Clear all panels
            </Button>
          )}
        </div>

        <div className="space-y-3 px-5">
          <StatRow
            icon={<Grid3x3 className="h-4 w-4" />}
            label="Panels placed"
            value={
              recommendation
                ? `${stats.count} / ${recommendation.design.panels_needed}`
                : stats.count.toString()
            }
          />
          {recommendation && (
            <>
              <StatRow
                icon={<Target className="h-4 w-4" />}
                label="Target roof area"
                value={`${recommendation.design.roof_space_sqm_needed.toFixed(1)} m²`}
              />
              <StatRow
                icon={<Battery className="h-4 w-4" />}
                label="Recommended battery"
                value={`${recommendation.design.recommended_battery_kwh} kWh`}
              />
            </>
          )}
          <StatRow
            icon={<Ruler className="h-4 w-4" />}
            label="Covered roof area"
            value={`${stats.area.toFixed(2)} m²`}
          />

          {(stats.byModel.flat.count > 0 || stats.byModel.slanted.count > 0) && (
            <div className="rounded-md border border-border bg-background/50 px-4 py-3 text-xs">
              <p className="mb-2 text-muted-foreground">By panel type</p>
              <div className="space-y-1.5">
                {(["flat", "slanted"] as const).map((k) => {
                  const b = stats.byModel[k];
                  if (b.count === 0) return null;
                  const def = PANEL_MODELS[k];
                  const perPanelArea = def.width * def.height;
                  return (
                    <div key={k} className="space-y-0.5">
                      <div className="flex items-center justify-between">
                        <span className="flex items-center gap-2">
                          <span
                            className="inline-block h-2 w-2 rounded-full"
                            style={{ backgroundColor: def.color }}
                          />
                          <span className="text-foreground">{def.label}</span>
                        </span>
                        <span className="font-mono tabular-nums text-foreground">
                          {b.count}
                        </span>
                      </div>
                      <div className="flex items-center justify-between pl-4 text-[11px] text-muted-foreground">
                        <span>
                          {def.width}×{def.height} m → {perPanelArea.toFixed(2)} m²/panel
                        </span>
                        <span className="font-mono tabular-nums">
                          € {def.pricePerUnit.toLocaleString("de-DE")}/each
                        </span>
                      </div>
                      <div className="flex items-center justify-between pl-4 text-[11px]">
                        <span className="text-muted-foreground">subtotal</span>
                        <span className="font-mono tabular-nums">
                          {b.area.toFixed(2)} m² · € {b.cost.toLocaleString("de-DE")}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {roofCapacity && (
            <div className="rounded-md border border-emerald-500/30 bg-emerald-500/5 px-4 py-3 text-xs">
              <p className="mb-1 text-muted-foreground">Roof patch (from seed)</p>
              <div className="flex items-center justify-between font-mono tabular-nums">
                <span className="text-muted-foreground">available</span>
                <span className="font-semibold text-foreground">
                  {roofCapacity.count} slots · {roofCapacity.roofArea.toFixed(1)} m²
                </span>
              </div>
              <div className="pl-4 text-[10px] text-muted-foreground">
                ↳ panels would cover {roofCapacity.panelCoverage.toFixed(1)} m² (the
                rest is gaps between panels)
              </div>
              <div className="mt-1 flex items-center justify-between font-mono tabular-nums">
                <span className="text-muted-foreground">placed</span>
                <span className="text-foreground">
                  {stats.count} · {stats.area.toFixed(1)} m² covered
                </span>
              </div>
              <div className="flex items-center justify-between font-mono tabular-nums">
                <span className="text-muted-foreground">remaining slots</span>
                <span
                  className={cn(
                    "font-semibold",
                    Math.max(0, roofCapacity.count - stats.count) === 0
                      ? "text-emerald-500"
                      : "text-amber-500",
                  )}
                >
                  {Math.max(0, roofCapacity.count - stats.count)} ·{" "}
                  {Math.max(0, roofCapacity.roofArea - (roofCapacity.roofArea / roofCapacity.count) * stats.count).toFixed(1)} m²
                </span>
              </div>
            </div>
          )}

          <StatRow
            icon={<Euro className="h-4 w-4" />}
            label={
              recommendation && stats.count === 0
                ? "ML cost estimate"
                : "Live placement cost"
            }
            value={`€ ${(recommendation && stats.count === 0
              ? recommendation.design.estimated_total_cost_euros
              : (recommendation
                  ? recommendation.design.recommended_battery_kwh * PRICE_PER_KWH_BATTERY
                  : 0) + stats.cost
            ).toLocaleString("de-DE", { maximumFractionDigits: 0 })}`}
            highlight
          />
        </div>

        <div className="mt-auto border-t border-border p-5 text-xs text-muted-foreground">
          <p className="mb-1 font-medium text-foreground">Tip</p>
          Show the coordinate grid, align it to your roof (pan, rotate, height), then
          click "Set Grid". After that, use "Place Solar Panels" to mark which cells are on
          the roof surface, then click Confirm to place panels immediately. Drag placed panels
          to reposition them.
        </div>
      </aside>

    </div>
  );
}

function StatRow({
  icon,
  label,
  value,
  highlight,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  highlight?: boolean;
}) {
  return (
    <div
      className={cn(
        "flex items-center justify-between rounded-md border border-border bg-background/50 px-4 py-3",
        highlight && "border-primary/40 bg-primary/5"
      )}
    >
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        {icon}
        {label}
      </div>
      <div className={cn("text-sm font-semibold", highlight && "text-primary")}>
        {value}
      </div>
    </div>
  );
}
