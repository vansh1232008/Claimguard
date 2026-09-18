import Link from "next/link";

import { ScoreBar } from "@/components/ScoreBar";
import { StatTile } from "@/components/StatTile";
import { api, currency, decisionColor } from "@/lib/api";

export const dynamic = "force-dynamic";

export default async function DashboardPage() {
  let stats;
  let recent;
  try {
    [stats, recent] = await Promise.all([api.stats(), api.investigations({ limit: 8 })]);
  } catch (err) {
    return (
      <div className="card-pad">
        <h1 className="text-lg font-semibold">Cannot reach the API</h1>
        <p className="mt-2 text-sm text-muted">
          Start the backend with <code className="font-mono">make api</code> (or{" "}
          <code className="font-mono">uvicorn app.main:app --app-dir backend</code>), then reload.
        </p>
        <pre className="mt-4 overflow-x-auto rounded-lg bg-black/40 p-3 text-xs text-reject">
          {String(err)}
        </pre>
      </div>
    );
  }

  const maxBucket = Math.max(...stats.score_histogram.map((b) => b.count), 1);
  const flaggedShare = stats.total_exposure
    ? stats.flagged_exposure / stats.total_exposure
    : 0;

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-semibold">Claims overview</h1>
        <p className="mt-1 text-sm text-muted">
          {stats.investigated.toLocaleString()} of {stats.total_claims.toLocaleString()} claims have
          been through the pipeline.
        </p>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatTile
          label="Straight-through approved"
          value={stats.approved.toLocaleString()}
          hint="score below the auto-approve threshold"
          tone="approve"
        />
        <StatTile
          label="Referred for review"
          value={stats.review.toLocaleString()}
          hint="needs an investigator"
          tone="review"
        />
        <StatTile
          label="Declined / SIU"
          value={stats.rejected.toLocaleString()}
          hint="not covered, or score above the referral line"
          tone="reject"
        />
        <StatTile
          label="Fraud rings detected"
          value={stats.rings_detected.toLocaleString()}
          hint="connected clusters of 3+ claims"
        />
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <StatTile
          label="Total exposure"
          value={currency(stats.total_exposure)}
          hint="sum of every claimed amount"
        />
        <StatTile
          label="Exposure under suspicion"
          value={currency(stats.flagged_exposure)}
          hint={`${(flaggedShare * 100).toFixed(1)}% of total exposure`}
          tone="review"
        />
        <StatTile
          label="Cost per claim"
          value={`${stats.avg_llm_calls.toFixed(1)} LLM calls`}
          hint={`${stats.avg_duration_ms.toFixed(0)} ms average end-to-end`}
        />
      </div>

      <div className="grid gap-6 lg:grid-cols-[minmax(340px,1fr)_1.4fr]">
        <section className="card-pad min-w-0">
          <h2 className="text-sm font-semibold">Fraud score distribution</h2>
          <p className="mt-1 text-xs text-muted">
            A healthy portfolio is heavily left-skewed: most claims are ordinary.
          </p>
          <div className="mt-5 space-y-1.5">
            {stats.score_histogram.map((bucket) => (
              <div key={bucket.bucket} className="flex items-center gap-3">
                <span className="w-16 font-mono text-[11px] text-muted">{bucket.bucket}</span>
                <div className="h-3 flex-1 overflow-hidden rounded bg-edge">
                  <div
                    className="h-full rounded bg-accent/70"
                    style={{ width: `${(bucket.count / maxBucket) * 100}%` }}
                  />
                </div>
                <span className="w-10 text-right font-mono text-[11px] tabular-nums text-muted">
                  {bucket.count}
                </span>
              </div>
            ))}
          </div>
        </section>

        <section className="card min-w-0">
          <div className="flex items-center justify-between border-b border-edge px-5 py-4">
            <h2 className="text-sm font-semibold">Latest decisions</h2>
            <Link href="/claims" className="text-xs text-accent hover:underline">
              All claims →
            </Link>
          </div>
          <div className="divide-y divide-edge">
            {recent.length === 0 ? (
              <p className="px-5 py-6 text-sm text-muted">
                Nothing investigated yet. Run <code className="font-mono">make demo</code> to
                process a batch.
              </p>
            ) : (
              recent.map((inv) => (
                <Link
                  key={inv.id}
                  href={`/claims/${inv.claim_id}`}
                  className="flex items-center gap-4 px-5 py-3 transition hover:bg-white/5"
                >
                  <span
                    className={`chip w-20 justify-center ${decisionColor(inv.decision)}`}
                  >
                    {inv.decision ?? "—"}
                  </span>
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-sm">{inv.rationale ?? "No rationale recorded."}</p>
                    <div className="mt-1.5 max-w-xs">
                      <ScoreBar score={inv.fraud_score} showTicks={false} />
                    </div>
                  </div>
                  <span className="font-mono text-sm tabular-nums">
                    {((inv.fraud_score ?? 0) * 100).toFixed(0)}
                  </span>
                </Link>
              ))
            )}
          </div>
        </section>
      </div>
    </div>
  );
}
