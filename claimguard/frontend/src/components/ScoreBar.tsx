import { scoreColor } from "@/lib/api";

/** Fraud score with the two decision thresholds marked on the track. */
export function ScoreBar({
  score,
  approveBelow = 0.25,
  rejectAbove = 0.85,
  showTicks = true,
}: {
  score: number | null | undefined;
  approveBelow?: number;
  rejectAbove?: number;
  showTicks?: boolean;
}) {
  const value = Math.max(0, Math.min(1, score ?? 0));
  return (
    <div className="w-full">
      <div className="relative h-2 w-full overflow-hidden rounded-full bg-edge">
        <div
          className={`h-full rounded-full ${scoreColor(score)}`}
          style={{ width: `${value * 100}%` }}
        />
        {showTicks ? (
          <>
            <span
              className="absolute top-0 h-full w-px bg-white/40"
              style={{ left: `${approveBelow * 100}%` }}
            />
            <span
              className="absolute top-0 h-full w-px bg-white/40"
              style={{ left: `${rejectAbove * 100}%` }}
            />
          </>
        ) : null}
      </div>
      {showTicks ? (
        <div className="mt-1 flex justify-between text-[10px] text-muted">
          <span>auto-approve ≤ {approveBelow}</span>
          <span>SIU referral ≥ {rejectAbove}</span>
        </div>
      ) : null}
    </div>
  );
}
