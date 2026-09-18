import Link from "next/link";

import { api, currency } from "@/lib/api";

export const dynamic = "force-dynamic";

const STATUS_FILTERS = [
  { value: "", label: "All" },
  { value: "submitted", label: "Not yet investigated" },
  { value: "under_review", label: "Under review" },
  { value: "approved", label: "Approved" },
  { value: "rejected", label: "Declined" },
];

export default async function ClaimsPage({
  searchParams,
}: {
  searchParams: { status?: string; search?: string };
}) {
  const claims = await api.claims({
    status: searchParams.status,
    search: searchParams.search,
    limit: 100,
  });

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold">Claims queue</h1>
          <p className="mt-1 text-sm text-muted">{claims.length} claims shown</p>
        </div>
        <form className="flex items-center gap-2">
          {searchParams.status ? (
            <input type="hidden" name="status" value={searchParams.status} />
          ) : null}
          <input
            name="search"
            defaultValue={searchParams.search ?? ""}
            placeholder="Claim number or customer"
            className="rounded-lg border border-edge bg-panel px-3 py-1.5 text-sm outline-none
              placeholder:text-muted focus:border-accent"
          />
          <button className="btn" type="submit">
            Search
          </button>
        </form>
      </div>

      <div className="flex flex-wrap gap-2">
        {STATUS_FILTERS.map((filter) => {
          const active = (searchParams.status ?? "") === filter.value;
          const href = filter.value ? `/claims?status=${filter.value}` : "/claims";
          return (
            <Link
              key={filter.label}
              href={href}
              className={`chip ${
                active
                  ? "border-accent/50 bg-accent/15 text-accent"
                  : "border-edge bg-white/5 text-slate-300"
              }`}
            >
              {filter.label}
            </Link>
          );
        })}
      </div>

      <div className="card overflow-hidden">
        <table className="w-full">
          <thead className="border-b border-edge bg-white/[0.03]">
            <tr>
              <th className="th">Claim</th>
              <th className="th">Type</th>
              <th className="th">Incident</th>
              <th className="th text-right">Claimed</th>
              <th className="th">Status</th>
              <th className="th">Ring</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-edge">
            {claims.map((claim) => (
              <tr key={claim.id} className="transition hover:bg-white/5">
                <td className="td">
                  <Link href={`/claims/${claim.id}`} className="font-mono text-accent hover:underline">
                    {claim.claim_number}
                  </Link>
                </td>
                <td className="td capitalize text-slate-300">
                  {claim.incident_type.replace(/_/g, " ")}
                </td>
                <td className="td text-muted">{claim.incident_date.slice(0, 10)}</td>
                <td className="td text-right font-mono tabular-nums">
                  {currency(claim.claimed_amount)}
                </td>
                <td className="td">
                  <span className="chip border-edge bg-white/5 text-slate-300">
                    {claim.status.replace(/_/g, " ")}
                  </span>
                </td>
                <td className="td font-mono text-xs text-muted">{claim.fraud_ring_id ?? "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {claims.length === 0 ? (
          <p className="px-5 py-8 text-center text-sm text-muted">
            No claims match. Seed some with <code className="font-mono">make seed</code>.
          </p>
        ) : null}
      </div>
    </div>
  );
}
