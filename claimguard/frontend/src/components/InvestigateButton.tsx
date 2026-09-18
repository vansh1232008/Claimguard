"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

import { api } from "@/lib/api";

export function InvestigateButton({ claimId, hasResult }: { claimId: string; hasResult: boolean }) {
  const router = useRouter();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function run() {
    setBusy(true);
    setError(null);
    try {
      await api.investigate(claimId, true);
      router.refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex flex-col items-end gap-1">
      <button className="btn-accent" onClick={run} disabled={busy}>
        {busy ? "Running the pipeline…" : hasResult ? "Re-investigate" : "Investigate claim"}
      </button>
      {error ? <span className="text-xs text-reject">{error}</span> : null}
    </div>
  );
}
