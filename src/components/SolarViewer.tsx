import { useCallback, useMemo, useRef, useState, Suspense, useEffect } from "react";
import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls, Grid, Environment } from "@react-three/drei";
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
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Upload, Sun, Trash2, Euro, Ruler, Grid3x3, Sparkles, MousePointerClick } from "lucide-react";
import { cn } from "@/lib/utils";

// Cost per panel in euros (typical residential panel ~ €250)
const COST_PER_PANEL = 250;
// Typical residential panel size in meters
const PANEL_W = 1.0;
const PANEL_H = 1.65;
const PANEL_GAP = 0.05;

type PanelData = {
  id: string;
  position: [number, number, number];
  // Quaternion so panels can be oriented for any "up" axis
  quaternion: [number, number, number, number];
  width: number;
  height: number;
};

// A candidate flat surface with a precomputed grid of potential panel slots.
type Candidate = {
  id: string;
  quaternion: [number, number, number, number];
  // Centers + sizes for every slot in this candidate's grid
  slots: PanelData[];
  // Outline corners for the holographic overlay (4 world-space points)
  outline: [number, number, number][];
  // Plane height (for sorting)
  height: number;
};

function ModelMesh({ object }: { object: THREE.Object3D }) {
  return <primitive object={object} />;
}

function Panel({
  data,
  onRemove,
}: {
  data: PanelData;
  onRemove: (id: string) => void;
}) {
  const [hovered, setHovered] = useState(false);
  return (
    <mesh
      position={data.position}
      quaternion={data.quaternion}
      onClick={(e) => {
        e.stopPropagation();
        onRemove(data.id);
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
      <boxGeometry args={[data.width, data.height, 0.06]} />
      <meshStandardMaterial
        color={hovered ? "hsl(140, 100%, 65%)" : "hsl(140, 100%, 50%)"}
        metalness={0.3}
        roughness={0.4}
        emissive="hsl(140, 100%, 45%)"
        emissiveIntensity={hovered ? 0.9 : 0.55}
      />
    </mesh>
  );
}

// Holographic outline (semi-transparent glowing plane) for a candidate area
function HoloArea({ candidate }: { candidate: Candidate }) {
  // Compute centroid + dimensions from the slot bounds
  const { center, w, h } = useMemo(() => {
    const xs = candidate.slots.map((s) => s.position[0]);
    const ys = candidate.slots.map((s) => s.position[1]);
    const zs = candidate.slots.map((s) => s.position[2]);
    const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
    const cy = (Math.min(...ys) + Math.max(...ys)) / 2;
    const cz = (Math.min(...zs) + Math.max(...zs)) / 2;
    // Use outline points to compute size
    const o = candidate.outline;
    const dU = new THREE.Vector3(...o[1]).sub(new THREE.Vector3(...o[0])).length();
    const dV = new THREE.Vector3(...o[3]).sub(new THREE.Vector3(...o[0])).length();
    return { center: [cx, cy, cz] as [number, number, number], w: dU, h: dV };
  }, [candidate]);

  return (
    <mesh position={center} quaternion={candidate.quaternion}>
      <planeGeometry args={[w, h]} />
      <meshBasicMaterial
        color="hsl(180, 100%, 60%)"
        transparent
        opacity={0.35}
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

function SceneContent({
  model,
  panels,
  onRemovePanel,
  highlights,
  pickingSlots,
  selectedSlots,
  onToggleSlot,
}: {
  model: THREE.Object3D | null;
  panels: PanelData[];
  onRemovePanel: (id: string) => void;
  highlights: Candidate[];
  pickingSlots: PanelData[] | null;
  selectedSlots: Set<string>;
  onToggleSlot: (id: string) => void;
}) {
  const { camera } = useThree();

  useEffect(() => {
    if (!model) return;
    const box = new THREE.Box3().setFromObject(model);
    const size = box.getSize(new THREE.Vector3()).length();
    const center = box.getCenter(new THREE.Vector3());
    camera.position.set(center.x + size, center.y + size * 0.6, center.z + size);
    camera.lookAt(center);
    (camera as THREE.PerspectiveCamera).near = size / 100;
    (camera as THREE.PerspectiveCamera).far = size * 100;
    camera.updateProjectionMatrix();
  }, [model, camera]);

  return (
    <>
      <ambientLight intensity={0.6} />
      <directionalLight position={[10, 20, 10]} intensity={1.2} castShadow />
      <directionalLight position={[-10, 10, -10]} intensity={0.4} />
      <Environment preset="city" />
      {model && <ModelMesh object={model} />}
      {panels.map((p) => (
        <Panel key={p.id} data={p} onRemove={onRemovePanel} />
      ))}
      {highlights.map((c) => (
        <HoloArea key={c.id} candidate={c} />
      ))}
      {pickingSlots?.map((s) => (
        <SlotPicker
          key={s.id}
          slot={s}
          selected={selectedSlots.has(s.id)}
          onToggle={onToggleSlot}
        />
      ))}
      {!model && (
        <Grid
          args={[20, 20]}
          cellColor="hsl(220, 15%, 70%)"
          sectionColor="hsl(220, 30%, 50%)"
          fadeDistance={30}
          infiniteGrid
        />
      )}
      <OrbitControls makeDefault enableDamping />
    </>
  );
}

/**
 * Find ALL meaningful flat horizontal surfaces in the model.
 * Returns one Candidate per surface, each with its own grid of slots.
 */
function findCandidateSurfaces(model: THREE.Object3D): Candidate[] {
  type Tri = {
    area: number;
    centroid: THREE.Vector3;
    normal: THREE.Vector3;
  };
  const tris: Tri[] = [];

  model.updateMatrixWorld(true);

  model.traverse((obj) => {
    const mesh = obj as THREE.Mesh;
    if (!(mesh.isMesh && mesh.geometry)) return;
    const geom = mesh.geometry as THREE.BufferGeometry;
    const pos = geom.attributes.position as THREE.BufferAttribute | undefined;
    if (!pos) return;

    const idx = geom.index;
    const triCount = idx ? idx.count / 3 : pos.count / 3;
    const a = new THREE.Vector3();
    const b = new THREE.Vector3();
    const c = new THREE.Vector3();
    const ab = new THREE.Vector3();
    const ac = new THREE.Vector3();

    for (let i = 0; i < triCount; i++) {
      const i0 = idx ? idx.getX(i * 3) : i * 3;
      const i1 = idx ? idx.getX(i * 3 + 1) : i * 3 + 1;
      const i2 = idx ? idx.getX(i * 3 + 2) : i * 3 + 2;

      a.fromBufferAttribute(pos, i0).applyMatrix4(mesh.matrixWorld);
      b.fromBufferAttribute(pos, i1).applyMatrix4(mesh.matrixWorld);
      c.fromBufferAttribute(pos, i2).applyMatrix4(mesh.matrixWorld);

      ab.subVectors(b, a);
      ac.subVectors(c, a);
      const cross = new THREE.Vector3().crossVectors(ab, ac);
      const area = cross.length() * 0.5;
      if (area < 1e-6) continue;
      const normal = cross.normalize();

      const centroid = new THREE.Vector3()
        .add(a)
        .add(b)
        .add(c)
        .multiplyScalar(1 / 3);

      tris.push({ area, centroid, normal });
    }
  });

  if (tris.length === 0) return [];

  // Detect up axis
  let areaY = 0;
  let areaZ = 0;
  for (const t of tris) {
    areaY += Math.abs(t.normal.y) * t.area;
    areaZ += Math.abs(t.normal.z) * t.area;
  }
  const upAxis = new THREE.Vector3(0, 1, 0);
  if (areaZ > areaY) upAxis.set(0, 0, 1);

  const flatTris = tris.filter((t) => t.normal.dot(upAxis) > 0.95);
  if (flatTris.length === 0) return [];

  const upHeight = (v: THREE.Vector3) => v.dot(upAxis);

  // Cluster flat tris by height (greedy buckets)
  const sorted = [...flatTris].sort(
    (a, b) => upHeight(b.centroid) - upHeight(a.centroid)
  );
  const tol = 0.25;
  const clusters: Tri[][] = [];
  for (const t of sorted) {
    const h = upHeight(t.centroid);
    const c = clusters.find(
      (cl) => Math.abs(upHeight(cl[0].centroid) - h) <= tol
    );
    if (c) c.push(t);
    else clusters.push([t]);
  }

  // Build coordinate frame
  const axisU = new THREE.Vector3();
  const axisV = new THREE.Vector3();
  if (upAxis.y === 1) {
    axisU.set(1, 0, 0);
    axisV.set(0, 0, 1);
  } else {
    axisU.set(1, 0, 0);
    axisV.set(0, 1, 0);
  }

  const q = new THREE.Quaternion().setFromUnitVectors(
    new THREE.Vector3(0, 0, 1),
    upAxis
  );
  const quat: [number, number, number, number] = [q.x, q.y, q.z, q.w];

  const toWorld = (u: number, v: number, h: number): [number, number, number] => {
    const p = new THREE.Vector3()
      .addScaledVector(axisU, u)
      .addScaledVector(axisV, v)
      .addScaledVector(upAxis, h);
    return [p.x, p.y, p.z];
  };

  const lift = 0.08;
  const candidates: Candidate[] = [];

  for (const cluster of clusters) {
    const totalArea = cluster.reduce((s, t) => s + t.area, 0);
    if (totalArea < 2.0) continue; // ignore tiny patches

    const refH = upHeight(cluster[0].centroid);
    const us = cluster.map((t) => t.centroid.dot(axisU));
    const vs = cluster.map((t) => t.centroid.dot(axisV));
    const minU = Math.min(...us);
    const maxU = Math.max(...us);
    const minV = Math.min(...vs);
    const maxV = Math.max(...vs);
    const width = maxU - minU;
    const depth = maxV - minV;
    if (width < PANEL_W || depth < PANEL_H) continue;

    const cols = Math.floor((width + PANEL_GAP) / (PANEL_W + PANEL_GAP));
    const rows = Math.floor((depth + PANEL_GAP) / (PANEL_H + PANEL_GAP));
    if (cols < 1 || rows < 1) continue;

    const totalW = cols * PANEL_W + (cols - 1) * PANEL_GAP;
    const totalD = rows * PANEL_H + (rows - 1) * PANEL_GAP;
    const startU = (minU + maxU) / 2 - totalW / 2 + PANEL_W / 2;
    const startV = (minV + maxV) / 2 - totalD / 2 + PANEL_H / 2;

    const slots: PanelData[] = [];
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        slots.push({
          id: crypto.randomUUID(),
          position: toWorld(
            startU + c * (PANEL_W + PANEL_GAP),
            startV + r * (PANEL_H + PANEL_GAP),
            refH + lift
          ),
          quaternion: quat,
          width: PANEL_W,
          height: PANEL_H,
        });
      }
    }

    const u0 = (minU + maxU) / 2 - totalW / 2;
    const u1 = (minU + maxU) / 2 + totalW / 2;
    const v0 = (minV + maxV) / 2 - totalD / 2;
    const v1 = (minV + maxV) / 2 + totalD / 2;
    const outline: [number, number, number][] = [
      toWorld(u0, v0, refH + lift * 0.5),
      toWorld(u1, v0, refH + lift * 0.5),
      toWorld(u1, v1, refH + lift * 0.5),
      toWorld(u0, v1, refH + lift * 0.5),
    ];

    candidates.push({
      id: crypto.randomUUID(),
      quaternion: quat,
      slots,
      outline,
      height: refH,
    });
  }

  // Sort highest first
  candidates.sort((a, b) => b.height - a.height);
  return candidates;
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
  const fileInputRef = useRef<HTMLInputElement>(null);

  const loadGlb = useCallback(async (file: File) => {
    setLoading(true);
    setError(null);
    setPanels([]);
    setHighlights([]);
    setShowHighlights(false);
    setPlacementMode(null);
    setSelectedSlots(new Set());
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

  // Compute candidates lazily
  const ensureCandidates = useCallback((): Candidate[] => {
    if (!model) return [];
    if (highlights.length > 0) return highlights;
    const found = findCandidateSurfaces(model);
    setHighlights(found);
    return found;
  }, [model, highlights]);

  const toggleHighlights = useCallback(() => {
    if (!model) return;
    const found = ensureCandidates();
    if (found.length === 0) {
      setError("No flat horizontal surfaces detected on this model.");
      return;
    }
    setError(null);
    setShowHighlights((v) => !v);
  }, [model, ensureCandidates]);

  const openPlacement = useCallback(() => {
    if (!model) return;
    const found = ensureCandidates();
    if (found.length === 0) {
      setError("No flat horizontal surfaces detected on this model.");
      return;
    }
    setError(null);
    setPlacementMode("choose");
  }, [model, ensureCandidates]);

  const placeRecommended = useCallback(() => {
    const all = highlights.flatMap((c) => c.slots);
    setPanels(all.map((s) => ({ ...s, id: crypto.randomUUID() })));
    setPlacementMode(null);
    setShowHighlights(false);
  }, [highlights]);

  const startPicking = useCallback(() => {
    setSelectedSlots(new Set());
    setPlacementMode("picking");
    setShowHighlights(false);
  }, []);

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

  const clearPanels = useCallback(() => setPanels([]), []);

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
              highlights={showHighlights ? highlights : []}
              pickingSlots={pickingSlots}
              selectedSlots={selectedSlots}
              onToggleSlot={toggleSlot}
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
      </div>

      {/* Sidebar */}
      <aside className="flex w-80 flex-col border-l border-border bg-card">
        <div className="border-b border-border p-5">
          <h1 className="flex items-center gap-2 text-xl font-bold">
            <Sun className="h-5 w-5 text-primary" />
            Solar Studio
          </h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Plan rooftop solar installations on any 3D model.
          </p>
        </div>

        <div className="space-y-3 p-5">
          <Button
            variant="outline"
            className="w-full"
            onClick={toggleHighlights}
            disabled={!model || loading}
          >
            <Sparkles className="mr-2 h-4 w-4" />
            {showHighlights ? "Hide possible areas" : "Highlight possible areas"}
          </Button>
          <Button
            className="w-full"
            onClick={openPlacement}
            disabled={!model || loading || placementMode === "picking"}
          >
            <Sun className="mr-2 h-4 w-4" />
            Place Solar Panels
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
            label="Panels"
            value={stats.count.toString()}
          />
          <StatRow
            icon={<Ruler className="h-4 w-4" />}
            label="Roof area covered"
            value={`${stats.area.toFixed(2)} m²`}
          />
          <StatRow
            icon={<Euro className="h-4 w-4" />}
            label="Estimated cost"
            value={`€ ${stats.cost.toLocaleString("de-DE")}`}
            highlight
          />
        </div>

        <div className="mt-auto border-t border-border p-5 text-xs text-muted-foreground">
          <p className="mb-1 font-medium text-foreground">Tip</p>
          Click any green panel in the 3D view to remove it. Use "Highlight
          possible areas" to preview where panels can go.
        </div>
      </aside>

      {/* Placement mode chooser */}
      <Dialog
        open={placementMode === "choose"}
        onOpenChange={(o) => !o && setPlacementMode(null)}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Place Solar Panels</DialogTitle>
            <DialogDescription>
              Choose how you want to lay out panels on the detected surfaces.
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-3 sm:grid-cols-2">
            <button
              onClick={placeRecommended}
              className="flex flex-col items-start gap-2 rounded-lg border border-border bg-background/50 p-4 text-left transition-colors hover:border-primary hover:bg-primary/5"
            >
              <Sparkles className="h-5 w-5 text-primary" />
              <div className="font-semibold">Use recommended</div>
              <div className="text-xs text-muted-foreground">
                Auto-fill every detected flat surface with a full grid of panels.
              </div>
              <div className="mt-1 text-xs text-primary">
                {highlights.reduce((s, c) => s + c.slots.length, 0)} panels across{" "}
                {highlights.length} surface{highlights.length === 1 ? "" : "s"}
              </div>
            </button>
            <button
              onClick={startPicking}
              className="flex flex-col items-start gap-2 rounded-lg border border-border bg-background/50 p-4 text-left transition-colors hover:border-primary hover:bg-primary/5"
            >
              <MousePointerClick className="h-5 w-5 text-primary" />
              <div className="font-semibold">Pick your own</div>
              <div className="text-xs text-muted-foreground">
                Click individual grid slots in the 3D view, then confirm to place.
              </div>
            </button>
          </div>
          <DialogFooter>
            <Button variant="ghost" onClick={() => setPlacementMode(null)}>
              Cancel
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
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
