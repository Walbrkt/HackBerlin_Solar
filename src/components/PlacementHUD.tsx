import { Battery, Coins, Target } from "lucide-react";
import { cn } from "@/lib/utils";

export type PlacementHUDProps = {
  panelsPlaced: number;
  targetPanels: number;
  pricePerPanel: number;
  baseCost: number; // ML-predicted battery cost (fixed)
  className?: string;
};

/**
 * Floating overlay shown over the 3D canvas while placing panels.
 * Tracks progress against the ML target and totals the live cost.
 */
export default function PlacementHUD({
  panelsPlaced,
  targetPanels,
  pricePerPanel,
  baseCost,
  className,
}: PlacementHUDProps) {
  const remaining = Math.max(0, targetPanels - panelsPlaced);
  const goalMet = panelsPlaced >= targetPanels && targetPanels > 0;
  const liveTotal = baseCost + panelsPlaced * pricePerPanel;
  const progress =
    targetPanels > 0 ? Math.min(1, panelsPlaced / targetPanels) : 0;

  return (
    <div
      className={cn(
        "pointer-events-none absolute left-1/2 top-4 z-10 -translate-x-1/2",
        "flex w-[440px] max-w-[calc(100vw-2rem)] flex-col gap-2 rounded-xl border border-border/80",
        "bg-card/90 px-4 py-3 shadow-xl backdrop-blur",
        className,
      )}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-sm font-semibold">
          <Target
            className={cn("h-4 w-4", goalMet ? "text-emerald-500" : "text-primary")}
          />
          <span>
            Panels placed:{" "}
            <span
              className={cn(
                "font-mono tabular-nums",
                goalMet ? "text-emerald-500" : "text-foreground",
              )}
            >
              {panelsPlaced} / {targetPanels}
            </span>
          </span>
        </div>
        <div
          className={cn(
            "text-xs font-medium",
            goalMet ? "text-emerald-500" : "text-muted-foreground",
          )}
        >
          {goalMet
            ? "✓ goal met"
            : `${remaining} ${remaining === 1 ? "panel" : "panels"} to go!`}
        </div>
      </div>

      {/* Progress bar */}
      <div className="h-1.5 overflow-hidden rounded-full bg-background/70">
        <div
          className={cn(
            "h-full transition-[width] duration-200",
            goalMet ? "bg-emerald-500" : "bg-primary",
          )}
          style={{ width: `${progress * 100}%` }}
        />
      </div>

      <div className="flex items-center justify-between text-xs">
        <div className="flex items-center gap-1.5 text-muted-foreground">
          <Battery className="h-3.5 w-3.5" />
          <span>Battery base: € {baseCost.toLocaleString("de-DE")}</span>
        </div>
        <div className="flex items-center gap-1.5 font-semibold">
          <Coins className="h-3.5 w-3.5 text-primary" />
          <span>
            Estimated price:{" "}
            <span className="font-mono tabular-nums">
              €{" "}
              {Math.round(liveTotal).toLocaleString("de-DE")}
            </span>
          </span>
        </div>
      </div>
    </div>
  );
}
