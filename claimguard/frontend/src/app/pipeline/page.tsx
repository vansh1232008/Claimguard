import { api } from "@/lib/api";

export const dynamic = "force-dynamic";

const NOTES: Record<string, string> = {
  intake:
    "Normalises the first notice of loss and raises the cheap flags (late reporting, incidents inside the inception window, thin narratives) before anything expensive runs.",
  document_analysis:
    "Extracts fields from invoices and reports, then cross-checks them against the claim form. Most padded claims fail here first.",
  damage_analysis:
    "Runs the vision module over the photographs and asks whether the damage shown plausibly costs what is being claimed.",
  coverage:
    "Retrieves the governing clauses from the policy wording and decides coverage against them, so a decline can cite the clause it rests on.",
  anomaly:
    "Scores the claim with the gradient-boosted model and turns the feature contributions into a written explanation.",
  network:
    "Walks the relationship graph for shared identities and repairers. Skipped when the claim is isolated and low-scoring.",
  adjudication:
    "Blends the model score with the rule-derived signals and produces the recommendation, the rationale and the payout.",
};

export default async function PipelinePage() {
  const info = await api.agents();
  const system = await api.systemInfo();

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-semibold">Pipeline</h1>
        <p className="mt-1 max-w-2xl text-sm text-muted">
          Seven agents on a {info.engine} state graph. The two conditional edges are what make it a
          graph rather than a loop: an unusable filing short-circuits to adjudication, and an
          isolated low-scoring claim skips the network agent entirely.
        </p>
      </div>

      <section className="card-pad">
        <h2 className="text-sm font-semibold">Execution order</h2>
        <ol className="mt-5 space-y-4">
          {info.order.map((name, i) => (
            <li key={name} className="flex gap-4">
              <span className="grid h-7 w-7 shrink-0 place-items-center rounded-lg border border-edge bg-white/5 font-mono text-xs">
                {i + 1}
              </span>
              <div>
                <h3 className="text-sm font-semibold capitalize">{name.replace(/_/g, " ")}</h3>
                <p className="mt-1 max-w-3xl text-sm text-muted">{NOTES[name] ?? ""}</p>
              </div>
            </li>
          ))}
        </ol>
      </section>

      <section className="card-pad">
        <h2 className="text-sm font-semibold">Runtime configuration</h2>
        <dl className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          <Item label="LLM" value={`${system.llm.mode} (${system.llm.model})`} />
          <Item
            label="Policy retrieval"
            value={`${system.vector_store.mode} · ${system.vector_store.clauses} clauses`}
          />
          <Item label="Graph store" value={system.graph_store.mode} />
          <Item
            label="Fraud model"
            value={`${system.fraud_model.mode} · ${system.fraud_model.features} features`}
          />
          <Item label="Database" value={system.database} />
          <Item label="Graph engine" value={system.pipeline.engine} />
        </dl>
        <div className="mt-6 border-t border-edge pt-4">
          <h3 className="text-xs uppercase tracking-wide text-muted">Decision thresholds</h3>
          <div className="mt-3 flex flex-wrap gap-6 font-mono text-sm">
            <span>
              <span className="text-approve">auto-approve</span> ≤{" "}
              {system.thresholds.auto_approve_below}
            </span>
            <span>
              <span className="text-reject">SIU referral</span> ≥{" "}
              {system.thresholds.auto_reject_above}
            </span>
            <span>
              <span className="text-review">manual sign-off above</span>{" "}
              {system.thresholds.high_value_claim_threshold.toLocaleString()}
            </span>
          </div>
        </div>
      </section>
    </div>
  );
}

function Item({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="label">{label}</dt>
      <dd className="mt-1 font-mono text-sm">{value}</dd>
    </div>
  );
}
