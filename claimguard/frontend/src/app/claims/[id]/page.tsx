import Link from "next/link";
import { notFound } from "next/navigation";

import { AgentTimeline } from "@/components/AgentTimeline";
import { InvestigateButton } from "@/components/InvestigateButton";
import { NetworkGraph } from "@/components/NetworkGraph";
import { ScoreBar } from "@/components/ScoreBar";
import { api, currency, decisionColor, type SubGraph } from "@/lib/api";

export const dynamic = "force-dynamic";

const SEVERITY_STYLE: Record<string, string> = {
  high: "border-reject/40 bg-reject/10 text-reject",
  medium: "border-review/40 bg-review/10 text-review",
  low: "border-edge bg-white/5 text-muted",
};

export default async function ClaimPage({ params }: { params: { id: string } }) {
  let claim;
  try {
    claim = await api.claim(params.id);
  } catch {
    notFound();
  }

  let subgraph: SubGraph = { nodes: [], links: [] };
  try {
    subgraph = await api.subgraph(params.id, 2);
  } catch {
    /* graph is optional context */
  }

  const inv = claim.latest_investigation;
  const signals = inv?.risk_signals ?? [];
  const clauses = (inv?.evidence ?? []).filter((e) => e.type === "policy_clause");

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <Link href="/claims" className="text-xs text-accent hover:underline">
            ← Claims queue
          </Link>
          <h1 className="mt-1 font-mono text-2xl font-semibold">{claim.claim_number}</h1>
          <p className="mt-1 text-sm text-muted">
            {claim.customer_name} · policy {claim.policy_number} ·{" "}
            {claim.incident_type.replace(/_/g, " ")} on {claim.incident_date.slice(0, 10)}
          </p>
        </div>
        <InvestigateButton claimId={claim.id} hasResult={Boolean(inv)} />
      </div>

      <div className="grid gap-6 lg:grid-cols-[1.35fr_1fr]">
        <div className="space-y-6">
          {/* ---------------------------------------------------- decision */}
          <section className="card-pad">
            {inv ? (
              <>
                <div className="flex flex-wrap items-center gap-3">
                  <span className={`chip px-3 py-1 text-sm ${decisionColor(inv.decision)}`}>
                    {inv.decision?.toUpperCase()}
                  </span>
                  <span className="font-mono text-sm text-muted">
                    fraud score {((inv.fraud_score ?? 0) * 100).toFixed(0)} / 100 · confidence{" "}
                    {((inv.confidence ?? 0) * 100).toFixed(0)}%
                  </span>
                  <span className="ml-auto font-mono text-sm">
                    payout {currency(inv.recommended_payout)}
                  </span>
                </div>
                <div className="mt-4">
                  <ScoreBar score={inv.fraud_score} />
                </div>
                <p className="mt-4 text-sm leading-relaxed text-slate-200">{inv.rationale}</p>
                <p className="mt-3 font-mono text-[11px] text-muted">
                  {inv.agent_runs.length} agents · {inv.llm_calls} LLM calls · {inv.duration_ms} ms ·
                  engine {inv.agent_summary?.engine ?? "—"}
                </p>
              </>
            ) : (
              <p className="text-sm text-muted">
                This claim has not been investigated yet. Run the pipeline to produce a
                recommendation.
              </p>
            )}
          </section>

          {/* ------------------------------------------------------ agents */}
          {inv ? (
            <section className="card-pad">
              <h2 className="mb-5 text-sm font-semibold">Agent trail</h2>
              <AgentTimeline runs={inv.agent_runs} />
            </section>
          ) : null}

          {/* ------------------------------------------------------- graph */}
          <section className="card-pad">
            <div className="flex items-baseline justify-between">
              <h2 className="text-sm font-semibold">Relationship neighbourhood</h2>
              {subgraph.ring?.size ? (
                <span className="font-mono text-[11px] text-muted">
                  cluster of {subgraph.ring.size} · cohesion {subgraph.ring.cohesion}
                </span>
              ) : null}
            </div>
            <div className="mt-4">
              <NetworkGraph data={subgraph} />
            </div>
          </section>
        </div>

        {/* ------------------------------------------------------- sidebar */}
        <div className="space-y-6">
          <section className="card-pad">
            <h2 className="text-sm font-semibold">Claim</h2>
            <dl className="mt-4 space-y-3 text-sm">
              <Row label="Claimed" value={currency(claim.claimed_amount)} />
              <Row label="Repair estimate" value={currency(claim.estimated_repair_cost)} />
              <Row label="Reported" value={claim.reported_date.slice(0, 10)} />
              <Row label="Witnesses" value={String(claim.witnesses)} />
              <Row label="Police report" value={claim.police_report ? "yes" : "no"} />
              <Row label="Injury claimed" value={claim.injury_claimed ? "yes" : "no"} />
              <Row label="Prior claims (12m)" value={String(claim.prior_claims_12m)} />
              <Row label="Repairer" value={claim.vendor_name ?? "—"} />
            </dl>
            <p className="mt-4 border-t border-edge pt-4 text-sm leading-relaxed text-slate-300">
              {claim.incident_description || "No description supplied."}
            </p>
          </section>

          <section className="card-pad">
            <h2 className="text-sm font-semibold">
              Risk signals{" "}
              <span className="font-normal text-muted">({signals.length})</span>
            </h2>
            <ul className="mt-4 space-y-3">
              {signals.length === 0 ? (
                <li className="text-sm text-muted">No signals raised.</li>
              ) : (
                signals.map((signal, i) => (
                  <li key={i} className="flex gap-3">
                    <span
                      className={`chip shrink-0 ${SEVERITY_STYLE[signal.severity] ?? SEVERITY_STYLE.low}`}
                    >
                      {signal.severity}
                    </span>
                    <div className="min-w-0">
                      <p className="text-sm text-slate-200">{signal.detail}</p>
                      <p className="mt-0.5 font-mono text-[10px] text-muted">
                        {signal.source} · {signal.code} · +{signal.weight}
                      </p>
                    </div>
                  </li>
                ))
              )}
            </ul>
          </section>

          <section className="card-pad">
            <h2 className="text-sm font-semibold">Policy clauses relied on</h2>
            <ul className="mt-4 space-y-3">
              {clauses.length === 0 ? (
                <li className="text-sm text-muted">No clauses retrieved.</li>
              ) : (
                clauses.map((clause, i) => (
                  <li key={i}>
                    <p className="font-mono text-xs text-accent">{clause.label}</p>
                    <p className="mt-1 text-xs leading-relaxed text-muted">
                      {String((clause.detail as any)?.excerpt ?? "").slice(0, 220)}…
                    </p>
                  </li>
                ))
              )}
            </ul>
          </section>

          {claim.documents.length > 0 ? (
            <section className="card-pad">
              <h2 className="text-sm font-semibold">Documents</h2>
              <ul className="mt-4 space-y-3">
                {claim.documents.map((doc) => (
                  <li key={doc.id}>
                    <p className="font-mono text-xs">{doc.filename}</p>
                    <p className="mt-0.5 text-[11px] text-muted">
                      {doc.doc_type} · total{" "}
                      {doc.extracted_fields?.total_amount
                        ? currency(Number(doc.extracted_fields.total_amount))
                        : "not extracted"}
                      {doc.extracted_fields?.invoice_number
                        ? ` · ${doc.extracted_fields.invoice_number}`
                        : ""}
                    </p>
                  </li>
                ))}
              </ul>
            </section>
          ) : null}
        </div>
      </div>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline justify-between gap-4">
      <dt className="text-muted">{label}</dt>
      <dd className="font-mono tabular-nums">{value}</dd>
    </div>
  );
}
