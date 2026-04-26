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
// Local "out" axis of the BoxGeometry-based panel preview (Z is the thin axis).
const PANEL_LOCAL_NORMAL = new THREE.Vector3(0, 0, 1);

const FLAT_PANEL_URL = "/solar_panels/solar_panel_flat.glb";
useGLTF.preload(FLAT_PANEL_URL);

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
};

type Candidate = {
  id: string;
  quaternion: [number, number, number, number];
  slots: PanelData[];
  outline: [number, number, number][];
  height: number;
};

type SnapPose = {
  position: [number, number, number];
  quaternion: [number, number, number, number];
};

// Renders the loaded model. When `snap` props are provided, attaches pointer
// handlers so hovering over the surface emits a BVH-accelerated snap pose and
// clicking commits a permanent panel.
function ModelMesh({
  object,
  snap,
}: {
  object: THREE.Object3D;
  snap?: {
    onHover: (pose: SnapPose | null) => void;
    onPlace: (pose: SnapPose) => void;
  };
}) {
  // Reusable scratch — created once, reused on every pointermove.
  const tmpQuatRef = useRef(new THREE.Quaternion());
  const tmpNormalRef = useRef(new THREE.Vector3());

  const poseFromEvent = useCallback((e: any): SnapPose | null => {
    if (!e.face) return null;
    // face.normal is in the hit object's LOCAL space — promote to world.
    tmpNormalRef.current
      .copy(e.face.normal)
      .transformDirection(e.object.matrixWorld)
      .normalize();
    tmpQuatRef.current.setFromUnitVectors(PANEL_LOCAL_NORMAL, tmpNormalRef.current);
    const q = tmpQuatRef.current;
    return {
      position: [e.point.x, e.point.y, e.point.z],
      quaternion: [q.x, q.y, q.z, q.w],
    };
  }, []);

  if (!snap) return <primitive object={object} />;

  return (
    <primitive
      object={object}
      onPointerMove={(e: any) => {
        e.stopPropagation();
        const pose = poseFromEvent(e);
        if (pose) snap.onHover(pose);
      }}
      onPointerOut={() => snap.onHover(null)}
      onClick={(e: any) => {
        // onClick fires only on a non-dragged left click — leaves drag-to-rotate intact.
        if (e.button !== 0) return;
        e.stopPropagation();
        const pose = poseFromEvent(e);
        if (pose) snap.onPlace(pose);
      }}
    />
  );
}

// Translucent preview rendered at the cursor in free-placement mode.
function PreviewPanel({ pose }: { pose: SnapPose }) {
  return (
    <mesh position={pose.position} quaternion={pose.quaternion}>
      <boxGeometry args={[PANEL_W, PANEL_H, 0.18]} />
      <meshStandardMaterial
        color="hsl(140, 100%, 55%)"
        emissive="hsl(140, 100%, 35%)"
        emissiveIntensity={0.7}
        transparent
        opacity={0.55}
        depthWrite={false}
      />
    </mesh>
  );
}

function Panel({
  data,
  onMove,
  onHoldingChange,
}: {
  data: PanelData;
  onMove: (id: string, position: [number, number, number]) => void;
  onHoldingChange: (id: string | null) => void;
}) {
  const [hovered, setHovered] = useState(false);
  const [dragging, setDragging] = useState(false);
  const dragPlaneRef = useRef(new THREE.Plane());
  const dragOffsetRef = useRef(new THREE.Vector3());
  const intersectionRef = useRef(new THREE.Vector3());

  const { scene } = useGLTF(FLAT_PANEL_URL);

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
        if (mat.emissive) {
          mat.emissive.setScalar(hovered ? 0.12 : 0);
          mat.emissiveIntensity = hovered ? 1.0 : 0;
        }
      });
    });
  }, [hovered, clonedScene]);

  const startHolding = useCallback((e: any) => {
    e.stopPropagation();
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
  }, [data.id, data.position, onHoldingChange]);

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
  onSnapHover,
  onSnapPlace,
}: {
  model: THREE.Object3D | null;
  panels: PanelData[];
  onRemovePanel: (id: string) => void;
  onMovePanel: (id: string, position: [number, number, number]) => void;
  onRotatePanel: (id: string, quaternion: [number, number, number, number]) => void;
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
  onSnapHover: (pose: SnapPose | null) => void;
  onSnapPlace: (pose: SnapPose) => void;
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

      if (event.key === "ArrowUp") {
        event.preventDefault();
        onMovePanel(heldPanelId, [
          active.position[0],
          active.position[1],
          active.position[2] + 0.1,
        ]);
      }

      if (event.key === "ArrowDown") {
        event.preventDefault();
        onMovePanel(heldPanelId, [
          active.position[0],
          active.position[1],
          active.position[2] - 0.1,
        ]);
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
  }, [heldPanelId, onMovePanel, onRotatePanel, onRemovePanel, panels]);

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
              ? { onHover: onSnapHover, onPlace: onSnapPlace }
              : undefined
          }
        />
      )}
      {freePlacementMode && hoverPose && <PreviewPanel pose={hoverPose} />}
      {panels.map((p) => (
        <Panel
          key={p.id}
          data={p}
          onMove={onMovePanel}
          onHoldingChange={setHeldPanelId}
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
  const fileInputRef = useRef<HTMLInputElement>(null);
  const addingMoreRef = useRef(false);

  // Commit a permanent panel from a hover pose — used by the BVH snap handler.
  const placeFromSnap = useCallback((pose: SnapPose) => {
    setPanels((prev) => [
      ...prev,
      {
        id: crypto.randomUUID(),
        position: pose.position,
        quaternion: pose.quaternion,
        width: PANEL_W,
        height: PANEL_H,
        depth: 0.18,
      },
    ]);
  }, []);

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
    setPanels((prev) => prev.map((panel) => (panel.id === id ? { ...panel, position } : panel)));
  }, []);

  const rotatePanel = useCallback((id: string, quaternion: [number, number, number, number]) => {
    setPanels((prev) => prev.map((panel) => (panel.id === id ? { ...panel, quaternion } : panel)));
  }, []);

  const clearPanels = useCallback(() => {
    setPanels([]);
    setHighlights([]);
    setShowHighlights(false);
    setSelectedGridCells(new Set());
    setGridSelectionMode(false);
    setSelectedSlots(new Set());
    setPlacementMode(null);
    setHoverPose(null);
    setError(null);
  }, []);

  const stats = useMemo(() => {
    const area = panels.reduce((s, p) => s + p.width * p.height, 0);
    return {
      count: panels.length,
      area,
      cost: panels.length * COST_PER_PANEL,
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
              onSnapHover={setHoverPose}
              onSnapPlace={placeFromSnap}
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
            }}
            disabled={!model || loading}
          >
            <MousePointerClick className="mr-2 h-4 w-4" />
            {freePlacementMode ? "Stop free placement" : "Free placement (hover & click)"}
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
            label="Selected roof area"
            value={`${stats.area.toFixed(2)} m²`}
          />
          <StatRow
            icon={<Euro className="h-4 w-4" />}
            label={recommendation ? "ML cost estimate" : "Estimated cost"}
            value={`€ ${(recommendation
              ? recommendation.design.estimated_total_cost_euros
              : stats.cost
            ).toLocaleString("de-DE")}`}
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
