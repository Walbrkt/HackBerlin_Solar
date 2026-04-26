# Solar Roof Designer

An interactive 3D tool for planning rooftop solar installations. Drop any `.glb` building model into the canvas, describe the household in plain English, and the app uses a trained ML model to recommend a system size — then lets you place, arrange, and fine-tune solar panels directly on the roof surface.

---

## Features

### 3D Roof Viewer

- **Drag-and-drop `.glb` models** — load any 3D building model by dropping it onto the canvas or using the file picker. Supports Draco- and Meshopt-compressed files.
- **BVH-accelerated raycasting** — the loaded model is indexed into a Bounding Volume Hierarchy on load so all surface queries (hover, snap, flood-fill) stay fast regardless of mesh complexity.
- **Orbit camera** — drag to rotate, scroll to zoom; orbit is automatically locked while a panel is being dragged.

---

### Free-Placement Mode

Hover over any point on the roof and panels snap flush to the exact surface, correctly aligned to the face normal (flat or sloped).

| Action | Control |
|---|---|
| Preview panel on surface | Hover over the roof |
| Lock target ("aim") | Left-click on roof |
| Rotate preview in-plane | **Q** / **E** (15° steps) |
| Reset preview rotation | **R** |
| Commit panel at locked target | **Enter** or "Place panel" button |

- **Automatic model selection** — the preview and the placed panel automatically use the **flat GLB** on horizontal surfaces (≤ 10° tilt) and the **slanted GLB** on pitched surfaces, so the correct mounting style is always shown.
- **Color-coded preview** — flat panels preview in green, slanted panels in orange, so you can see which type will be placed before committing.
- The click only **aims** (locks the target), never immediately places — this prevents accidentally spawning panels when clicking through the model.

---

### Auto-Layout (BFS Flood-Fill)

Starting from a seed point on the roof, the app grows a parallel panel grid outward using a breadth-first flood-fill over the BVH:

- Each candidate cell is ray-cast from above; the hit is accepted only if:
  - **(a)** Its face normal is within **12°** of the seed's normal — rejects walls, dormers, and neighbouring slopes.
  - **(b)** The hit point lies within **0.5 m** of the seed's tangent plane — rejects ground, trees, and objects next to the building.
  - **(c)** It is reachable from its parent cell without a large positional jump — handles thin eave gaps where a ray would punch through to the ground.
- Accepted cells each get the correct panel model (flat/slanted) based on their individual face normal.
- Results are sorted by distance from the seed so the closest slots are filled first.

**Measure** — hover over any roof spot and press "Measure" to run the flood-fill with a 1000-panel cap. The sidebar shows how many panels fit on that patch, total roof area available, and actual panel coverage area.

**Propose** — when an ML recommendation is loaded, "Propose design" runs the flood-fill and places exactly the recommended number of panels starting from the hovered seed point. If the roof can't fit that many, it places as many as possible and shows the shortfall.

---

### Grid-Based Placement (Manual Flow)

An alternative to free-placement for users who prefer explicit cell selection:

1. Show the **coordinate grid** and align it to the roof — adjust pan (X/Y), rotation (Z°), and height offset with sliders.
2. Click **Set Grid** to lock the configuration.
3. Click individual grid cells to mark the roof region; the count updates live.
4. Press **Confirm** — panels fill every selected cell; the grid hides automatically.
5. Use **Add more solar panels** (active once panels are placed) to select a new area and append without losing existing panels.

---

### Panel Controls

All keyboard controls activate while a panel is **held** (clicked):

| Action | Control |
|---|---|
| Move on X / Y | Click and drag |
| Move on Z | Arrow Up / Arrow Down (0.1 m per press) |
| Rotate in-plane | Arrow Left / Arrow Right (15° snapping) |
| Delete | Backspace |
| Add to / remove from selection | Shift + Click |
| Clear selection | Escape |

- **Collision detection** — a 2D Separating Axis Theorem (OBB) check runs on every drag and rotation, silently blocking any move that would cause two panels to overlap.
- **Multi-select** — Shift-click any number of panels; they glow blue. Arrow Up / Down while holding any panel in the selection moves **all selected panels** together — the primary way to snap a row of panels down to the roof surface.

---

### AI-Powered System Design

- **Free-text prompt** — describe the household (e.g. *"180 sqm house, two EVs, ~7 MWh/year, no solar yet"*) and press "Design my system".
- **Gemini 2.5 Flash extraction** — with `GOOGLE_AI_API_KEY` set, the prompt is sent to Gemini with a structured `response_schema`. It extracts whichever fields it can confidently infer and leaves the rest null.
- **Regex fallback** — when no key is configured or Gemini fails, a deterministic keyword parser covers demand (Wh / kWh / MWh), house size (m² / sqft), EV, solar, battery, wallbox, electric heating, and negation ("no EV").
- **Per-field provenance badges** — every value shows `extracted` (Gemini), `regex` (fallback), or `defaulted` (training-data median) so you can see exactly what was inferred.

---

### ML Inference & Live Stats

- **RandomForest model** trained on ~1,100 real residential projects (scikit-learn, joblib).
- **Inference pipeline**: feature engineering → StandardScaler → RandomForest → battery kWh → physics heuristic for panel count and roof area → cost.
- **Live sidebar stats** — all update in real time as panels are placed or removed:
  - Panels placed vs. recommended target
  - Roof capacity (panels that fit on the measured patch)
  - Recommended battery capacity (kWh)
  - Target roof area (from ML)
  - Placed panel coverage area (m²)
  - Estimated total cost (€)

---

### Panel Models

Two GLB models are registered, both preloaded at startup:

| Key | File | Use case | Price |
|---|---|---|---|
| `flat` | `solar_panel_flat.glb` | Horizontal / low-pitch roofs | €1,500/panel |
| `slanted` | `solar_panel_slanted.glb` | Pitched roofs with tilt frame | €1,700/panel |

The correct model is chosen automatically per cell based on the surface normal. Adding a new model requires one entry in the `PANEL_MODELS` registry in `SolarViewer.tsx`; the component auto-scales it to fit the physical dimensions via bounding-box measurement.

---

## Stack

| Layer | Technology |
|---|---|
| 3D rendering | React Three Fiber, Three.js, `@react-three/drei` |
| UI | React, TypeScript, Vite, Tailwind CSS, shadcn/ui |
| ML backend | FastAPI, scikit-learn (RandomForest), pandas, joblib |
| AI extraction | Google Gemini 2.5 Flash (`google-genai` SDK) |
| Env loading | `python-dotenv` |

---

## Getting Started

### 1. Install dependencies

```bash
# Frontend
npm install

# Backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

Create a `.env` file in the project root (already gitignored):

```env
GOOGLE_AI_API_KEY=your_key_here   # https://aistudio.google.com/app/apikey
VITE_PORT=8080
API_HOST=0.0.0.0
API_PORT=8000
```

### 3. Run

```bash
# Terminal 1 — ML backend (port 8000)
.venv/bin/python3 src/inference_server.py

# Terminal 2 — Frontend (port 8080)
npm run dev
```

Open [http://localhost:8080](http://localhost:8080). Vite proxies all `/api/*` requests to FastAPI automatically.

---

## Project Structure

```
public/solar_panels/     GLB panel models (flat + slanted)
src/
  components/
    SolarViewer.tsx      All 3D viewer and placement logic
    DesignFromPrompt.tsx AI prompt UI and recommendation display
    PlacementHUD.tsx     Live stats overlay
  lib/
    bvh-setup.ts         BVH index builder for raycasting
  inference_server.py    FastAPI backend (ML inference + prompt endpoint)
  feature_extractor.py   Gemini / regex feature extraction pipeline
  train_model.py         Model retraining script
model/                   Trained artifacts (joblib)
data/                    Processed training datasets
CSV/                     Raw input CSVs
```
