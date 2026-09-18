import type { AgentRun } from "@/lib/api";

const AGENT_LABELS: Record<string, string> = {
  intake: "Intake",
  document_analysis: "Documents",
  damage_analysis: "Damage",
  coverage: "Policy coverage",
  anomaly: "Anomaly model",
  network: "Network",
  adjudication: "Adjudication",
};

function statusDot(status: string): string {
  if (status === "failed") return "bg-reject";
  if (status === "skipped") return "bg-edge";
  return "bg-accent";
}

/** The audit trail: what each agent did, what it cost, what it concluded. */
export function AgentTimeline({ runs }: { runs: AgentRun[] }) {
  const ordered = [...runs].sort((a, b) => a.sequence - b.sequence);

  return (
    <ol className="relative space-y-4 border-l border-edge pl-6">
      {ordered.map((run) => (
        <li key={run.id} className="relative">
          <span
            className={`absolute -left-[1.6rem] top-1.5 h-2.5 w-2.5 rounded-full ring-4 ring-panel ${statusDot(
              run.status
            )}`}
          />
          <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
            <h3 className="text-sm font-semibold">
              {AGENT_LABELS[run.agent_name] ?? run.agent_name}
            </h3>
            <span className="font-mono text-[11px] text-muted">
              {run.duration_ms} ms · {run.llm_calls} LLM
              {run.risk_contribution > 0
                ? ` · +${run.risk_contribution.toFixed(2)} risk`
                : ""}
            </span>
            {run.status !== "ok" ? (
              <span className="chip border-edge bg-white/5 text-muted">{run.status}</span>
            ) : null}
          </div>
          <p className="mt-1 text-sm text-slate-300">{run.summary ?? "—"}</p>
          {run.error ? <p className="mt-1 text-xs text-reject">{run.error}</p> : null}
          <AgentDetail run={run} />
        </li>
      ))}
    </ol>
  );
}

function AgentDetail({ run }: { run: AgentRun }) {
  const out = run.output ?? {};
  const items: { label: string; value: string }[] = [];

  switch (run.agent_name) {
    case "intake":
      items.push(
        { label: "Completeness", value: fmtPct(out.completeness_score) },
        { label: "Severity", value: String(out.severity_band ?? "—") },
        { label: "Reporting delay", value: `${out.reporting_delay_days ?? 0} days` },
        { label: "Days since inception", value: `${out.days_since_policy_start ?? 0}` }
      );
      break;
    case "document_analysis":
      items.push(
        { label: "Documents", value: String(out.documents_reviewed ?? 0) },
        { label: "Consistency", value: fmtPct(out.consistency_score) },
        { label: "Mismatches", value: String((out.mismatches ?? []).length) },
        { label: "Invoice total", value: out.invoice_total ? String(out.invoice_total) : "—" }
      );
      break;
    case "damage_analysis":
      items.push(
        { label: "Severity", value: String(out.damage_severity ?? "—") },
        { label: "Plausibility", value: fmtPct(out.plausibility_score) },
        { label: "Vision backend", value: String(out.vision_backend ?? "—") },
        { label: "Regions", value: String((out.regions_detected ?? []).length) }
      );
      break;
    case "coverage":
      items.push(
        { label: "Covered", value: out.covered ? "yes" : "no" },
        { label: "Clauses retrieved", value: String(out.clauses_retrieved ?? 0) },
        { label: "Payable estimate", value: String(out.payable_estimate ?? "—") },
        { label: "Exclusions hit", value: String((out.exclusions_triggered ?? []).length) }
      );
      break;
    case "anomaly":
      items.push(
        { label: "Model", value: String(out.model ?? "—") },
        { label: "Probability", value: fmtPct(out.fraud_probability) },
        { label: "Severity", value: String(out.severity ?? "—") },
        { label: "Trained", value: out.model_trained ? "yes" : "cold start" }
      );
      break;
    case "network":
      items.push(
        { label: "Ring risk", value: String(out.ring_risk ?? "none") },
        { label: "Cluster size", value: String(out.ring_size ?? 0) },
        { label: "Cohesion", value: String(out.cohesion ?? 0) },
        { label: "Graph backend", value: String(out.graph_backend ?? "—") }
      );
      break;
    case "adjudication":
      items.push(
        { label: "Decision", value: String(out.decision ?? "—") },
        { label: "Confidence", value: fmtPct(out.confidence) },
        { label: "Model probability", value: fmtPct(out.model_probability) },
        { label: "Signal pressure", value: fmtPct(out.signal_pressure) }
      );
      break;
    default:
      return null;
  }

  return (
    <dl className="mt-3 grid grid-cols-2 gap-x-6 gap-y-2 sm:grid-cols-4">
      {items.map((item) => (
        <div key={item.label}>
          <dt className="text-[10px] uppercase tracking-wide text-muted">{item.label}</dt>
          <dd className="font-mono text-xs text-slate-200">{item.value}</dd>
        </div>
      ))}
    </dl>
  );
}

function fmtPct(value: unknown): string {
  const n = Number(value);
  return Number.isFinite(n) ? `${(n * 100).toFixed(0)}%` : "—";
}
