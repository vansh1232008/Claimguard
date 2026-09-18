export function StatTile({
  label,
  value,
  hint,
  tone = "default",
}: {
  label: string;
  value: string;
  hint?: string;
  tone?: "default" | "approve" | "review" | "reject";
}) {
  const toneClass = {
    default: "text-slate-100",
    approve: "text-approve",
    review: "text-review",
    reject: "text-reject",
  }[tone];

  return (
    <div className="card-pad">
      <div className="label">{label}</div>
      <div className={`mt-2 text-2xl font-semibold tabular-nums ${toneClass}`}>{value}</div>
      {hint ? <div className="mt-1 text-xs text-muted">{hint}</div> : null}
    </div>
  );
}
