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
import { Upload, Sun, Trash2, Euro, Ruler, Grid3x3 } from "lucide-react";
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
  rotationY: number;
  width: number;
  height: number;
};

function ModelMesh({ object }: { object: THREE.Object3D }) {
  return <primitive object={object} />;
}

function Panel({ data, onRemove }: { data: PanelData; onRemove: (id: string) => void }) {
  const [hovered, setHovered] = useState(false);
  return (
    <mesh
      position={data.position}
      rotation={[-Math.PI / 2, 0, data.rotationY]}
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
      <boxGeometry args={[data.width, data.height, 0.04]} />
      <meshStandardMaterial
        color={hovered ? "hsl(220, 80%, 35%)" : "hsl(220, 70%, 22%)"}
        metalness={0.6}
        roughness={0.25}
        emissive={hovered ? "hsl(220, 90%, 40%)" : "hsl(220, 60%, 10%)"}
        emissiveIntensity={hovered ? 0.4 : 0.1}
      />
    </mesh>
  );
}

function SceneContent({
  model,
  panels,
  onRemovePanel,
}: {
  model: THREE.Object3D | null;
  panels: PanelData[];
  onRemovePanel: (id: string) => void;
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
 * Find the highest "flat horizontal" surface in the model and return
 * a list of panel positions to fill it.
 */
function generatePanelsForModel(model: THREE.Object3D): PanelData[] {
  // Collect upward-facing triangles, grouped by their height (Y value).
  type Tri = { y: number; area: number; centroid: THREE.Vector3 };
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
    const normal = new THREE.Vector3();

    for (let i = 0; i < triCount; i++) {
      const i0 = idx ? idx.getX(i * 3) : i * 3;
      const i1 = idx ? idx.getX(i * 3 + 1) : i * 3 + 1;
      const i2 = idx ? idx.getX(i * 3 + 2) : i * 3 + 2;

      a.fromBufferAttribute(pos, i0).applyMatrix4(mesh.matrixWorld);
      b.fromBufferAttribute(pos, i1).applyMatrix4(mesh.matrixWorld);
      c.fromBufferAttribute(pos, i2).applyMatrix4(mesh.matrixWorld);

      ab.subVectors(b, a);
      ac.subVectors(c, a);
      normal.crossVectors(ab, ac);
      const area = normal.length() * 0.5;
      if (area < 1e-6) continue;
      normal.normalize();

      // Mostly horizontal (normal pointing up), within ~15deg
      if (normal.y < 0.95) continue;

      const centroid = new THREE.Vector3()
        .add(a)
        .add(b)
        .add(c)
        .multiplyScalar(1 / 3);

      tris.push({ y: centroid.y, area, centroid });
    }
  });

  if (tris.length === 0) return [];

  // Bucket triangles by height (with tolerance) and pick the highest
  // bucket that has a meaningful total area.
  tris.sort((a, b) => b.y - a.y);

  const tol = 0.05; // 5cm
  let best: Tri[] | null = null;
  let bestY = 0;

  for (let i = 0; i < tris.length; i++) {
    const refY = tris[i].y;
    const bucket = tris.filter((t) => Math.abs(t.y - refY) <= tol);
    const totalArea = bucket.reduce((s, t) => s + t.area, 0);
    if (totalArea > 0.5) {
      best = bucket;
      bestY = refY;
      break;
    }
  }

  if (!best) {
    best = [tris[0]];
    bestY = tris[0].y;
  }

  // Compute XZ bounding box of the best surface
  const minX = Math.min(...best.map((t) => t.centroid.x));
  const maxX = Math.max(...best.map((t) => t.centroid.x));
  const minZ = Math.min(...best.map((t) => t.centroid.z));
  const maxZ = Math.max(...best.map((t) => t.centroid.z));

  const width = maxX - minX;
  const depth = maxZ - minZ;
  if (width < PANEL_W || depth < PANEL_H) {
    // Surface is too small for a typical panel; place a single small one
    return [
      {
        id: crypto.randomUUID(),
        position: [(minX + maxX) / 2, bestY + 0.025, (minZ + maxZ) / 2],
        rotationY: 0,
        width: Math.max(0.3, width * 0.8),
        height: Math.max(0.3, depth * 0.8),
      },
    ];
  }

  const cols = Math.floor((width + PANEL_GAP) / (PANEL_W + PANEL_GAP));
  const rows = Math.floor((depth + PANEL_GAP) / (PANEL_H + PANEL_GAP));

  const totalW = cols * PANEL_W + (cols - 1) * PANEL_GAP;
  const totalD = rows * PANEL_H + (rows - 1) * PANEL_GAP;
  const startX = (minX + maxX) / 2 - totalW / 2 + PANEL_W / 2;
  const startZ = (minZ + maxZ) / 2 - totalD / 2 + PANEL_H / 2;

  const panels: PanelData[] = [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      panels.push({
        id: crypto.randomUUID(),
        position: [
          startX + c * (PANEL_W + PANEL_GAP),
          bestY + 0.025,
          startZ + r * (PANEL_H + PANEL_GAP),
        ],
        rotationY: 0,
        width: PANEL_W,
        height: PANEL_H,
      });
    }
  }
  return panels;
}

export default function SolarViewer() {
  const [model, setModel] = useState<THREE.Object3D | null>(null);
  const [panels, setPanels] = useState<PanelData[]>([]);
  const [dragOver, setDragOver] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const loadGlb = useCallback(async (file: File) => {
    setLoading(true);
    setError(null);
    setPanels([]);
    try {
      const arrayBuffer = await file.arrayBuffer();
      const loader = gltfLoader;
      const gltf = await loader.parseAsync(arrayBuffer, "");
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

  const placePanels = useCallback(() => {
    if (!model) return;
    const newPanels = generatePanelsForModel(model);
    if (newPanels.length === 0) {
      setError("No flat horizontal surface detected on this model.");
      return;
    }
    setError(null);
    setPanels(newPanels);
  }, [model]);

  const removePanel = useCallback((id: string) => {
    setPanels((prev) => prev.filter((p) => p.id !== id));
  }, []);

  const clearPanels = useCallback(() => setPanels([]), []);

  const stats = useMemo(() => {
    const area = panels.reduce((s, p) => s + p.width * p.height, 0);
    return {
      count: panels.length,
      area: area,
      cost: panels.length * COST_PER_PANEL,
    };
  }, [panels]);

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
            <SceneContent model={model} panels={panels} onRemovePanel={removePanel} />
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
            className="w-full"
            onClick={placePanels}
            disabled={!model || loading}
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
          Click any blue panel in the 3D view to remove it. Drag, scroll, and
          right-click drag in the viewer to orbit, zoom, and pan.
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
