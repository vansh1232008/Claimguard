import { api } from "@/lib/api";

/**
 * Shows which backend each swappable subsystem is running on.
 * Offline mode is a legitimate state, not an error - so it is labelled, not hidden.
 */
export async function SystemChips() {
  let info;
  try {
    info = await api.systemInfo();
  } catch {
    return (
      <span className="chip border-reject/40 bg-reject/10 text-reject">API unreachable</span>
    );
  }

  const chips = [
    { label: "LLM", value: info.llm.mode === "gemini" ? info.llm.model : "offline stub" },
    { label: "RAG", value: `${info.vector_store.mode} · ${info.vector_store.clauses} clauses` },
    { label: "Graph", value: info.graph_store.mode },
    { label: "Model", value: info.fraud_model.mode },
    { label: "Engine", value: info.pipeline.engine },
  ];

  return (
    <div className="flex flex-wrap items-center gap-1.5">
      {chips.map((chip) => (
        <span key={chip.label} className="chip border-edge bg-white/5 text-slate-300">
          <span className="text-muted">{chip.label}</span>
          {chip.value}
        </span>
      ))}
    </div>
  );
}
