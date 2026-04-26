import { useState } from "react";
import { Loader2, Sparkles, Wand2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

export type DesignRecommendation = {
  panels_needed: number;
  roof_space_sqm_needed: number;
  recommended_battery_kwh: number;
  estimated_total_cost_euros: number;
};

type Provenance = "extracted" | "regex" | "defaulted";

export type DesignFromPromptResponse = {
  design: DesignRecommendation;
  inputs_used: Record<string, number>;
  provenance: Record<string, Provenance>;
};

const EXAMPLES = [
  "Single-family home, ~180 sqm, two EVs charging at home, ~7 MWh per year, no solar yet.",
  "Small apartment with EV.",
  "10 MWh per year, no EV, no existing solar, no battery, 250 m² house, electric heating uses about 2000 kWh/year.",
];

const FIELD_LABELS: Record<string, string> = {
  energy_demand_wh: "Annual demand",
  has_ev: "Electric vehicle",
  has_solar: "Existing solar",
  has_storage: "Existing battery",
  has_wallbox: "Wallbox",
  house_size_sqm: "House size",
  heating_existing_electricity_demand_kwh: "Electric heating",
};

function formatValue(field: string, value: number): string {
  switch (field) {
    case "energy_demand_wh":
      return `${(value / 1000).toLocaleString("en-US", { maximumFractionDigits: 0 })} kWh/yr`;
    case "house_size_sqm":
      return `${value} m²`;
    case "heating_existing_electricity_demand_kwh":
      return `${value} kWh/yr`;
    default:
      // Booleans
      return value === 1 ? "yes" : "no";
  }
}

function provenanceClass(p: Provenance): string {
  if (p === "defaulted")
    return "bg-amber-500/10 text-amber-500 border-amber-500/30";
  if (p === "regex") return "bg-sky-500/10 text-sky-400 border-sky-500/30";
  return "bg-emerald-500/10 text-emerald-400 border-emerald-500/30";
}

export type DesignFromPromptProps = {
  /** Called whenever a fresh recommendation lands. */
  onResult?: (result: DesignFromPromptResponse) => void;
  /** Optional initial prompt. */
  initialPrompt?: string;
};

export default function DesignFromPrompt({ onResult, initialPrompt }: DesignFromPromptProps) {
  const [prompt, setPrompt] = useState(initialPrompt ?? "");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<DesignFromPromptResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function submit() {
    const trimmed = prompt.trim();
    if (!trimmed) {
      setError("Type a prompt first.");
      return;
    }
    setError(null);
    setLoading(true);
    try {
      const r = await fetch("/api/design-system/from-prompt", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: trimmed }),
      });
      const data = await r.json();
      if (!r.ok) {
        throw new Error(typeof data.detail === "string" ? data.detail : `HTTP ${r.status}`);
      }
      setResult(data);
      onResult?.(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <Card className="space-y-3 p-4">
      <div className="flex items-center gap-2">
        <Wand2 className="h-4 w-4 text-primary" />
        <h2 className="text-sm font-semibold">Design from a prompt</h2>
      </div>
      <p className="text-xs text-muted-foreground">
        Describe the household. We extract the features (Gemini → regex fallback)
        and run them through the ML model to size the system.
      </p>

      <Textarea
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        placeholder="e.g. 180 sqm house, two EVs, ~7 MWh/year, no solar yet"
        className="min-h-[80px] resize-none text-sm"
        onKeyDown={(e) => {
          if ((e.metaKey || e.ctrlKey) && e.key === "Enter") submit();
        }}
      />

      <div className="flex flex-wrap gap-1.5">
        {EXAMPLES.map((ex) => (
          <button
            key={ex}
            type="button"
            onClick={() => setPrompt(ex)}
            className="rounded-full border border-border bg-background px-2.5 py-1 text-[11px] text-muted-foreground hover:border-primary hover:text-foreground"
          >
            {ex.length > 40 ? ex.slice(0, 38) + "…" : ex}
          </button>
        ))}
      </div>

      <Button className="w-full" onClick={submit} disabled={loading}>
        {loading ? (
          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
        ) : (
          <Sparkles className="mr-2 h-4 w-4" />
        )}
        {loading ? "Designing…" : "Design my system"}
      </Button>

      {error && (
        <p className="rounded-md bg-destructive/10 px-3 py-2 text-xs text-destructive">
          {error}
        </p>
      )}

      {result && <RecommendationCard result={result} />}
    </Card>
  );
}

function RecommendationCard({ result }: { result: DesignFromPromptResponse }) {
  const { design, inputs_used, provenance } = result;
  return (
    <div className="space-y-3 rounded-md border border-primary/30 bg-primary/5 p-3">
      <div className="grid grid-cols-2 gap-2">
        <Stat label="Recommended panels" value={`${design.panels_needed}`} suffix="× 400 W" />
        <Stat
          label="Roof space"
          value={design.roof_space_sqm_needed.toLocaleString("en-US")}
          suffix="m²"
        />
        <Stat
          label="Battery"
          value={design.recommended_battery_kwh.toLocaleString("en-US")}
          suffix="kWh"
        />
        <Stat
          label="Total cost"
          value={`€ ${design.estimated_total_cost_euros.toLocaleString("de-DE")}`}
        />
      </div>

      <details className="text-xs text-muted-foreground">
        <summary className="cursor-pointer select-none font-medium hover:text-foreground">
          Inputs used by the model
        </summary>
        <ul className="mt-2 space-y-1">
          {Object.keys(FIELD_LABELS).map((field) => (
            <li key={field} className="flex items-center justify-between gap-2">
              <span>{FIELD_LABELS[field]}</span>
              <span className="flex items-center gap-1.5">
                <span className="font-medium text-foreground">
                  {formatValue(field, inputs_used[field])}
                </span>
                <Badge
                  variant="outline"
                  className={cn(
                    "border px-1.5 py-0 text-[10px] uppercase tracking-wide",
                    provenanceClass(provenance[field]),
                  )}
                >
                  {provenance[field]}
                </Badge>
              </span>
            </li>
          ))}
        </ul>
      </details>
    </div>
  );
}

function Stat({
  label,
  value,
  suffix,
}: {
  label: string;
  value: string;
  suffix?: string;
}) {
  return (
    <div className="rounded-md bg-background/60 px-3 py-2">
      <div className="text-[10px] uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="mt-0.5 text-base font-semibold leading-tight">
        {value}
        {suffix && <span className="ml-1 text-xs font-normal text-muted-foreground">{suffix}</span>}
      </div>
    </div>
  );
}
