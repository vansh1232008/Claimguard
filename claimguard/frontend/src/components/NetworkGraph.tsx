import type { SubGraph } from "@/lib/api";

/**
 * Static SVG of the claim's neighbourhood.
 *
 * Deliberately not a force simulation: the layout is a deterministic radial one
 * so the same claim always draws the same picture. An investigator comparing
 * two screenshots of the same cluster should see the same shape.
 */
export function NetworkGraph({ data }: { data: SubGraph }) {
  const width = 640;
  const height = 360;
  const cx = width / 2;
  const cy = height / 2;

  if (data.nodes.length === 0) {
    return (
      <p className="text-sm text-muted">
        This claim has no links to any other claim in the portfolio.
      </p>
    );
  }

  const focus = data.nodes.find((n) => n.focus) ?? data.nodes[0];
  const others = data.nodes.filter((n) => n.id !== focus.id);

  const positions = new Map<string, { x: number; y: number }>();
  positions.set(focus.id, { x: cx, y: cy });

  // Two rings: direct links inside, everything else outside.
  const direct = new Set(
    data.links
      .filter((l) => l.source === focus.id || l.target === focus.id)
      .map((l) => (l.source === focus.id ? l.target : l.source))
  );
  const inner = others.filter((n) => direct.has(n.id));
  const outer = others.filter((n) => !direct.has(n.id));

  const place = (list: typeof others, radius: number, offset: number) => {
    list.forEach((node, i) => {
      const angle = offset + (2 * Math.PI * i) / Math.max(list.length, 1);
      positions.set(node.id, {
        x: cx + radius * Math.cos(angle),
        y: cy + radius * Math.sin(angle) * 0.72,
      });
    });
  };
  place(inner, 110, -Math.PI / 2);
  place(outer, 165, -Math.PI / 2 + 0.4);

  const linkColor = (kinds: string[]) => {
    if (kinds.some((k) => k === "SHARED_BANK_ACCOUNT" || k === "SHARED_PHONE")) return "#d0524a";
    if (kinds.includes("SHARED_ADDRESS")) return "#d99a2b";
    return "#3b4a5e";
  };

  const nodeColor = (score: number | null) => {
    if (score === null || score === undefined) return "#4b5a6e";
    if (score >= 0.85) return "#d0524a";
    if (score >= 0.25) return "#d99a2b";
    return "#2f9e6e";
  };

  return (
    <div className="space-y-3">
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full">
        {data.links.map((link, i) => {
          const a = positions.get(link.source);
          const b = positions.get(link.target);
          if (!a || !b) return null;
          return (
            <line
              key={i}
              x1={a.x}
              y1={a.y}
              x2={b.x}
              y2={b.y}
              stroke={linkColor(link.kinds)}
              strokeWidth={link.weight >= 0.9 ? 2 : 1}
              strokeOpacity={0.75}
            />
          );
        })}
        {data.nodes.map((node) => {
          const pos = positions.get(node.id);
          if (!pos) return null;
          const r = node.focus ? 11 : 7;
          return (
            <g key={node.id}>
              <circle
                cx={pos.x}
                cy={pos.y}
                r={r}
                fill={nodeColor(node.fraud_score)}
                stroke={node.focus ? "#ffffff" : "#0d1117"}
                strokeWidth={node.focus ? 2 : 1.5}
              />
              <text
                x={pos.x}
                y={pos.y + r + 12}
                textAnchor="middle"
                className="fill-slate-400"
                style={{ fontSize: 9, fontFamily: "ui-monospace, monospace" }}
              >
                {node.label?.slice(-8)}
              </text>
            </g>
          );
        })}
      </svg>
      <div className="flex flex-wrap gap-4 text-[11px] text-muted">
        <span className="flex items-center gap-1.5">
          <span className="h-0.5 w-5" style={{ background: "#d0524a" }} /> shared phone / bank
          account
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-0.5 w-5" style={{ background: "#d99a2b" }} /> shared address
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-0.5 w-5" style={{ background: "#3b4a5e" }} /> shared vendor / area
        </span>
      </div>
    </div>
  );
}
