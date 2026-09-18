/** Typed client for the ClaimGuard API. */

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? (typeof window === "undefined" ? "http://127.0.0.1:8000" : "");

export type Decision = "approve" | "review" | "reject";

export interface RiskSignal {
  source: string;
  code: string;
  detail: string;
  severity: "low" | "medium" | "high";
  weight: number;
}

export interface EvidenceItem {
  type: string;
  label: string;
  detail: Record<string, unknown>;
}

export interface AgentRun {
  id: string;
  agent_name: string;
  sequence: number;
  status: string;
  summary: string | null;
  output: Record<string, any> | null;
  risk_contribution: number;
  llm_calls: number;
  duration_ms: number;
  error: string | null;
}

export interface Investigation {
  id: string;
  claim_id: string;
  status: string;
  decision: Decision | null;
  fraud_score: number | null;
  confidence: number | null;
  rationale: string | null;
  recommended_payout: number | null;
  risk_signals: RiskSignal[] | null;
  evidence: EvidenceItem[] | null;
  agent_summary: Record<string, any> | null;
  llm_calls: number;
  duration_ms: number;
  error: string | null;
  started_at: string;
  finished_at: string | null;
  agent_runs: AgentRun[];
}

export interface Claim {
  id: string;
  claim_number: string;
  customer_id: string;
  policy_id: string;
  incident_type: string;
  incident_date: string;
  reported_date: string;
  claimed_amount: number;
  estimated_repair_cost: number;
  status: string;
  is_fraud_label: boolean | null;
  fraud_ring_id: string | null;
  created_at: string;
}

export interface ClaimDetail extends Claim {
  incident_description: string;
  incident_postal_code: string | null;
  witnesses: number;
  police_report: boolean;
  injury_claimed: boolean;
  prior_claims_12m: number;
  customer_name: string | null;
  policy_number: string | null;
  vendor_name: string | null;
  documents: {
    id: string;
    doc_type: string;
    filename: string;
    extracted_fields: Record<string, any> | null;
    uploaded_at: string;
  }[];
  latest_investigation: Investigation | null;
}

export interface FraudRing {
  ring_id: string;
  size: number;
  total_exposure: number;
  shared_attributes: string[];
  cohesion: number;
  members: {
    claim_id: string;
    claim_number: string;
    customer_name: string;
    claimed_amount: number;
    fraud_score: number | null;
  }[];
}

export interface Stats {
  total_claims: number;
  investigated: number;
  pending: number;
  approved: number;
  review: number;
  rejected: number;
  avg_fraud_score: number;
  total_exposure: number;
  flagged_exposure: number;
  rings_detected: number;
  avg_duration_ms: number;
  avg_llm_calls: number;
  decisions_by_day: Record<string, any>[];
  score_histogram: { bucket: string; count: number }[];
}

export interface SystemInfo {
  llm: { mode: string; model: string };
  vector_store: { mode: string; clauses: number };
  graph_store: { mode: string };
  fraud_model: { mode: string; features: number };
  pipeline: { engine: string };
  database: string;
  thresholds: Record<string, number>;
}

export interface SubGraph {
  nodes: {
    id: string;
    label: string;
    customer: string;
    amount: number;
    fraud_score: number | null;
    focus: boolean;
  }[];
  links: { source: string; target: string; kinds: string[]; weight: number; value: string }[];
  ring?: FraudRing & { members: any[] };
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}/api${path}`, {
    cache: "no-store",
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status} ${res.statusText} - ${body.slice(0, 200)}`);
  }
  return (await res.json()) as T;
}

export const api = {
  stats: () => request<Stats>("/stats"),
  systemInfo: () => request<SystemInfo>("/system/info"),
  claims: (params: Record<string, string | number | undefined> = {}) => {
    const qs = new URLSearchParams();
    Object.entries(params).forEach(([k, v]) => {
      if (v !== undefined && v !== "") qs.set(k, String(v));
    });
    return request<Claim[]>(`/claims?${qs.toString()}`);
  },
  claim: (id: string) => request<ClaimDetail>(`/claims/${id}`),
  investigate: (id: string, force = true) =>
    request<Investigation>(`/claims/${id}/investigate`, {
      method: "POST",
      body: JSON.stringify({ force, include_damage_analysis: true }),
    }),
  investigations: (params: Record<string, string | number | undefined> = {}) => {
    const qs = new URLSearchParams();
    Object.entries(params).forEach(([k, v]) => {
      if (v !== undefined && v !== "") qs.set(k, String(v));
    });
    return request<Investigation[]>(`/investigations?${qs.toString()}`);
  },
  rings: (minSize = 3) => request<FraudRing[]>(`/graph/rings?min_size=${minSize}`),
  subgraph: (claimId: string, depth = 2) =>
    request<SubGraph>(`/graph/claims/${claimId}?depth=${depth}`),
  agents: () =>
    request<{ engine: string; order: string[]; agents: { name: string; description: string }[] }>(
      "/agents"
    ),
};

export function currency(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  // Plain thousands grouping: the reference dataset is French motor insurance,
  // so an en-IN lakh/crore grouping would be wrong for these amounts.
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 }).format(value);
}

export function decisionColor(decision: string | null | undefined): string {
  switch (decision) {
    case "approve":
      return "text-approve border-approve/40 bg-approve/10";
    case "reject":
      return "text-reject border-reject/40 bg-reject/10";
    case "review":
      return "text-review border-review/40 bg-review/10";
    default:
      return "text-muted border-edge bg-white/5";
  }
}

export function scoreColor(score: number | null | undefined): string {
  if (score === null || score === undefined) return "bg-edge";
  if (score >= 0.85) return "bg-reject";
  if (score >= 0.25) return "bg-review";
  return "bg-approve";
}
