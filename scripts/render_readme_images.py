"""
Generate placeholder + diagram images for the README.

Outputs to ../assets/screenshots/. Run:
    python scripts/render_readme_images.py

The screenshot images are styled placeholders — they keep the README from
showing broken image links until real captures are dropped in. The
architecture diagram is a real diagram, not a placeholder.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent.parent / "assets" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)

# Color system — matches the dark UI in the app.
BG = "#0f1419"
PANEL = "#161c24"
PANEL2 = "#1f2733"
BORDER = "#2a3441"
TEXT = "#e6edf3"
MUTED = "#8b96a5"
ACCENT_GREEN = "#4fb371"
ACCENT_ORANGE = "#e0934f"
ACCENT_BLUE = "#5e8ed1"
ACCENT_AMBER = "#d8a657"
ACCENT_RED = "#e06c75"
ACCENT_PURPLE = "#a980c4"


def card(
    path: Path,
    *,
    title: str,
    subtitle: str,
    accent: str,
    eyebrow: str,
    width: int = 1600,
    height: int = 900,
):
    """Render a branded placeholder card."""
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1600)
    ax.set_ylim(0, 900)
    ax.axis("off")

    # Accent bar on the left
    ax.add_patch(patches.Rectangle((0, 0), 14, 900, facecolor=accent, edgecolor="none"))

    # Subtle grid (very faint)
    for x in range(80, 1600, 80):
        ax.plot([x, x], [0, 900], color=BORDER, lw=0.4, alpha=0.3, zorder=0)
    for y in range(80, 900, 80):
        ax.plot([0, 1600], [y, y], color=BORDER, lw=0.4, alpha=0.3, zorder=0)

    # Eyebrow tag (uppercase pill)
    ax.text(
        90, 760,
        eyebrow.upper(),
        color=accent,
        fontsize=16,
        fontweight="bold",
        family="DejaVu Sans",
    )

    # Big title
    ax.text(
        90, 660,
        title,
        color=TEXT,
        fontsize=64,
        fontweight="bold",
        family="DejaVu Sans",
        verticalalignment="top",
    )

    # Subtitle (wrap manually to fit width)
    ax.text(
        90, 360,
        subtitle,
        color=MUTED,
        fontsize=22,
        family="DejaVu Sans",
        verticalalignment="top",
        wrap=True,
    )

    # Footer hint
    ax.text(
        90, 80,
        "PLACEHOLDER — replace with a real screenshot from the running app",
        color=MUTED,
        fontsize=14,
        family="DejaVu Sans",
        style="italic",
    )

    # Reonic-style corner mark
    ax.text(
        1500, 80,
        "REONIC × HackBerlin 2026",
        color=MUTED,
        fontsize=12,
        family="DejaVu Sans",
        ha="right",
        fontweight="bold",
    )

    fig.savefig(path, facecolor=BG, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"  wrote {path.relative_to(OUT.parent.parent)}")


def hero(path: Path):
    """Big hero card — first thing on the README."""
    fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1600)
    ax.set_ylim(0, 900)
    ax.axis("off")

    # Faint isometric grid effect (just diagonals)
    for k in range(-20, 25):
        ax.plot([k * 80, k * 80 + 900], [0, 900], color=BORDER, lw=0.4, alpha=0.18)

    # Accent bands at top + bottom
    ax.add_patch(patches.Rectangle((0, 870), 1600, 30, facecolor=ACCENT_GREEN, alpha=0.7))
    ax.add_patch(patches.Rectangle((0, 0), 1600, 12, facecolor=ACCENT_ORANGE, alpha=0.7))

    # Eyebrow
    ax.text(
        80, 770,
        "HACKBERLIN 2026",
        color=ACCENT_GREEN,
        fontsize=22,
        fontweight="bold",
        family="DejaVu Sans",
    )

    # Title
    ax.text(
        80, 670,
        "Solar Roof Designer",
        color=TEXT,
        fontsize=88,
        fontweight="bold",
        family="DejaVu Sans",
    )

    # Tagline
    ax.text(
        80, 540,
        "From “I think I want solar” to a clickable, priced,",
        color=MUTED,
        fontsize=32,
        family="DejaVu Sans",
    )
    ax.text(
        80, 480,
        "3D installation plan in under 60 seconds.",
        color=MUTED,
        fontsize=32,
        family="DejaVu Sans",
    )

    # Stat strip
    stats = [
        ("Photogrammetry .glb in", ACCENT_GREEN),
        ("Sized solar plan out", ACCENT_ORANGE),
        ("BVH-snapped 3D placement", ACCENT_BLUE),
        ("Live ML-priced HUD", ACCENT_PURPLE),
    ]
    x = 80
    for label, color in stats:
        # pill
        ax.add_patch(
            FancyBboxPatch(
                (x, 250), 360, 80,
                boxstyle="round,pad=4,rounding_size=10",
                facecolor=PANEL,
                edgecolor=color,
                linewidth=2,
            )
        )
        ax.text(
            x + 24, 290,
            label,
            color=TEXT,
            fontsize=18,
            family="DejaVu Sans",
            fontweight="bold",
        )
        x += 380

    # Footer
    ax.text(
        80, 60,
        "Built by team Reonic — drone .glb, FastAPI, Gemini, react-three-fiber, scikit-learn",
        color=MUTED,
        fontsize=16,
        family="DejaVu Sans",
        style="italic",
    )

    fig.savefig(path, facecolor=BG, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"  wrote {path.relative_to(OUT.parent.parent)}")


def architecture(path: Path):
    """Real architecture diagram — boxes + arrows."""
    fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1600)
    ax.set_ylim(0, 900)
    ax.axis("off")

    # Title
    ax.text(
        800, 850,
        "Architecture",
        color=TEXT, fontsize=36, fontweight="bold",
        family="DejaVu Sans", ha="center",
    )
    ax.text(
        800, 805,
        "Vite proxies /api/* to FastAPI — single page, two services",
        color=MUTED, fontsize=16, family="DejaVu Sans", ha="center",
    )

    def box(xy, w, h, label, sub=None, color=ACCENT_BLUE, fill=PANEL):
        x, y = xy
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h,
                boxstyle="round,pad=4,rounding_size=8",
                facecolor=fill, edgecolor=color, linewidth=2,
            )
        )
        ax.text(
            x + w / 2, y + h - 30,
            label, color=TEXT, fontsize=15,
            fontweight="bold", family="DejaVu Sans", ha="center",
        )
        if sub:
            for i, line in enumerate(sub):
                ax.text(
                    x + w / 2, y + h - 60 - i * 22,
                    line, color=MUTED, fontsize=12,
                    family="DejaVu Sans", ha="center",
                )

    # ── Vite container ───────────────────────────────
    ax.add_patch(
        FancyBboxPatch(
            (60, 380), 700, 380,
            boxstyle="round,pad=4,rounding_size=12",
            facecolor=PANEL, edgecolor=ACCENT_GREEN, linewidth=2.5,
        )
    )
    ax.text(
        80, 730, "VITE  :8080",
        color=ACCENT_GREEN, fontsize=13, fontweight="bold",
    )
    ax.text(
        80, 705, "React + react-three-fiber + drei + shadcn/ui",
        color=MUTED, fontsize=12, family="DejaVu Sans",
    )

    # frontend components
    box((90, 540), 200, 130, "SolarViewer.tsx",
        ["3D scene", "BVH raycast", "Free-place + auto"], ACCENT_GREEN)
    box((310, 540), 200, 130, "DesignFromPrompt",
        ["Prompt textarea", "Recommendation card", "Provenance badges"], ACCENT_PURPLE)
    box((530, 540), 200, 130, "PlacementHUD",
        ["Top overlay", "Progress bar", "Live cost"], ACCENT_AMBER)

    box((90, 410), 200, 100, "three-mesh-bvh",
        ["Mesh.raycast patch", "Sub-ms queries"], ACCENT_BLUE)
    box((310, 410), 420, 100, "feature_extractor (client UI)",
        ["Free text → POST /api/design-system/from-prompt"], ACCENT_RED)

    # ── FastAPI container ────────────────────────────
    ax.add_patch(
        FancyBboxPatch(
            (840, 200), 700, 560,
            boxstyle="round,pad=4,rounding_size=12",
            facecolor=PANEL, edgecolor=ACCENT_ORANGE, linewidth=2.5,
        )
    )
    ax.text(
        860, 730, "FASTAPI  :8000",
        color=ACCENT_ORANGE, fontsize=13, fontweight="bold",
    )
    ax.text(
        860, 705, "uvicorn + Pydantic v2",
        color=MUTED, fontsize=12, family="DejaVu Sans",
    )

    box((870, 560), 320, 130, "POST /api/design-system",
        ["Structured input (CustomerFeatures)", "→ ML pipeline"], ACCENT_BLUE)
    box((1210, 560), 320, 130, "POST /api/design-system/from-prompt",
        ["Free-text → extract → infer"], ACCENT_PURPLE)

    box((870, 400), 660, 130, "feature_extractor.py",
        ["Gemini 2.5 Flash (response_schema)",
         "↓ fallback when no key / error",
         "regex/keyword parser"], ACCENT_RED)

    box((870, 230), 660, 150, "scikit-learn pipeline",
        ["StandardScaler  →  RandomForestRegressor",
         "Multi-output: panels, battery_kwh, roof_sqm",
         "Persisted as model/*.joblib"], ACCENT_GREEN)

    # ── arrow proxy /api/* ───────────────────────────
    arrow = FancyArrowPatch(
        (760, 460), (840, 460),
        arrowstyle="-|>", mutation_scale=22,
        color=TEXT, linewidth=2,
    )
    ax.add_patch(arrow)
    ax.text(
        790, 480, "/api/*",
        color=TEXT, fontsize=12, family="DejaVu Sans",
        fontweight="bold", ha="center",
    )

    # data flow inside FastAPI (down)
    arrow2 = FancyArrowPatch(
        (1030, 560), (1030, 530),
        arrowstyle="-|>", mutation_scale=18,
        color=MUTED, linewidth=1.5,
    )
    ax.add_patch(arrow2)
    arrow3 = FancyArrowPatch(
        (1030, 400), (1030, 380),
        arrowstyle="-|>", mutation_scale=18,
        color=MUTED, linewidth=1.5,
    )
    ax.add_patch(arrow3)

    # ── outputs row ──────────────────────────────────
    ax.text(
        800, 110,
        "Output  →  panels_needed   roof_space_sqm_needed   recommended_battery_kwh   estimated_total_cost_euros",
        color=MUTED, fontsize=14, family="DejaVu Sans", ha="center", style="italic",
    )

    fig.savefig(path, facecolor=BG, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"  wrote {path.relative_to(OUT.parent.parent)}")


def main():
    print(f"Rendering README images into {OUT}")

    hero(OUT / "hero.png")
    architecture(OUT / "architecture.png")

    cards = [
        ("demo-flow.png",
         "DEMO FLOW",
         "Five steps to a priced solar plan",
         "Drop .glb  →  prompt  →  ML sizes  →  Propose layout  →  drag to fine-tune  →  done",
         ACCENT_GREEN),
        ("free-placement.png",
         "FREE PLACEMENT",
         "Hover, snap, click, Enter",
         "BVH-accelerated raycast finds the surface, the panel preview snaps to the local normal, click to lock the aim, Enter commits.",
         ACCENT_ORANGE),
        ("prompt-and-stats.png",
         "PROMPT → ML",
         "Plain English in, sized system out",
         "Gemini 2.5 Flash extracts features (or regex fallback). RandomForest sizes battery and panels. Each value tagged with provenance.",
         ACCENT_PURPLE),
        ("propose-design.png",
         "AUTO-LAYOUT",
         "BFS flood-fill across the BVH",
         "Sticks to one roof patch via normal-angle, tangent-plane offset, and reach-from-parent gates. Won't bleed onto walls or lawn.",
         ACCENT_BLUE),
        ("panel-models.png",
         "PANEL MODELS",
         "Flat for flat roofs, slanted for slopes",
         "Two GLBs in the registry; the surface tilt at each cell decides which gets placed. Adding a new model is one entry.",
         ACCENT_AMBER),
        ("live-hud.png",
         "LIVE HUD",
         "Progress, area, and cost in real time",
         "Top-of-canvas HUD tracks placed/target with a progress bar. Sidebar shows per-type counts, areas, and prices. Updates as you click.",
         ACCENT_RED),
    ]
    for fname, eyebrow, title, sub, accent in cards:
        card(OUT / fname, title=title, subtitle=sub, accent=accent, eyebrow=eyebrow)

    print(f"\nDone. {len(cards) + 2} images written.")


if __name__ == "__main__":
    main()
