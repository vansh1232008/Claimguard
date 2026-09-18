import Link from "next/link";

import { api, currency } from "@/lib/api";

export const dynamic = "force-dynamic";

const ATTR_LABELS: Record<string, string> = {
  SHARED_PHONE: "phone",
  SHARED_BANK_ACCOUNT: "bank account",
  SHARED_ADDRESS: "address",
  SHARED_VENDOR: "repairer",
  SHARED_VEHICLE: "vehicle",
  SAME_AREA_SAME_WEEK: "same area, same week",
};

const STRONG = new Set(["SHARED_PHONE", "SHARED_BANK_ACCOUNT", "SHARED_ADDRESS"]);

export default async function RingsPage() {
  const rings = await api.rings(3);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Fraud rings</h1>
        <p className="mt-1 max-w-2xl text-sm text-muted">
          Connected clusters of three or more claims that share identifying attributes. Identity
          links (phone, bank account, address) are strong evidence of collusion; a shared repairer
          or postcode on its own usually is not.
        </p>
      </div>

      {rings.length === 0 ? (
        <div className="card-pad text-sm text-muted">
          No clusters found. Seed more claims, then rebuild the graph with{" "}
          <code className="font-mono">POST /api/graph/refresh</code>.
        </div>
      ) : null}

      <div className="space-y-4">
        {rings.map((ring) => {
          const strong = ring.shared_attributes.filter((a) => STRONG.has(a));
          return (
            <section key={ring.ring_id} className="card overflow-hidden">
              <div className="flex flex-wrap items-center gap-x-6 gap-y-2 border-b border-edge px-5 py-4">
                <h2 className="font-mono text-sm">{ring.ring_id}</h2>
                <span className="text-sm">
                  <span className="font-semibold">{ring.size}</span>{" "}
                  <span className="text-muted">claims</span>
                </span>
                <span className="text-sm">
                  <span className="font-semibold">{currency(ring.total_exposure)}</span>{" "}
                  <span className="text-muted">exposure</span>
                </span>
                <span className="text-sm text-muted">cohesion {ring.cohesion.toFixed(2)}</span>
                <div className="ml-auto flex flex-wrap gap-1.5">
                  {ring.shared_attributes.map((attr) => (
                    <span
                      key={attr}
                      className={`chip ${
                        STRONG.has(attr)
                          ? "border-reject/40 bg-reject/10 text-reject"
                          : "border-edge bg-white/5 text-muted"
                      }`}
                    >
                      {ATTR_LABELS[attr] ?? attr}
                    </span>
                  ))}
                </div>
              </div>
              {strong.length === 0 ? (
                <p className="border-b border-edge bg-white/[0.02] px-5 py-2 text-xs text-muted">
                  Linked only by incidental attributes — likely a busy repairer rather than a ring.
                </p>
              ) : null}
              <table className="w-full">
                <thead>
                  <tr>
                    <th className="th">Claim</th>
                    <th className="th">Claimant</th>
                    <th className="th text-right">Claimed</th>
                    <th className="th text-right">Score</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-edge">
                  {ring.members.slice(0, 12).map((member) => (
                    <tr key={member.claim_id} className="transition hover:bg-white/5">
                      <td className="td">
                        <Link
                          href={`/claims/${member.claim_id}`}
                          className="font-mono text-accent hover:underline"
                        >
                          {member.claim_number}
                        </Link>
                      </td>
                      <td className="td text-slate-300">{member.customer_name}</td>
                      <td className="td text-right font-mono tabular-nums">
                        {currency(member.claimed_amount)}
                      </td>
                      <td className="td text-right font-mono tabular-nums">
                        {member.fraud_score === null || member.fraud_score === undefined
                          ? "—"
                          : (member.fraud_score * 100).toFixed(0)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {ring.members.length > 12 ? (
                <p className="px-5 py-2 text-xs text-muted">
                  + {ring.members.length - 12} more claims in this cluster
                </p>
              ) : null}
            </section>
          );
        })}
      </div>
    </div>
  );
}
