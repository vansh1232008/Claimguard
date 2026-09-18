import type { Metadata } from "next";
import Link from "next/link";

import "./globals.css";
import { SystemChips } from "@/components/SystemChips";

export const metadata: Metadata = {
  title: "ClaimGuard",
  description: "Autonomous insurance claims investigation platform",
};

const NAV = [
  { href: "/", label: "Dashboard" },
  { href: "/claims", label: "Claims" },
  { href: "/rings", label: "Fraud rings" },
  { href: "/pipeline", label: "Pipeline" },
];

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="min-h-screen">
          <header className="sticky top-0 z-20 border-b border-edge bg-ink/85 backdrop-blur">
            <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-x-8 gap-y-3 px-6 py-3">
              <Link href="/" className="flex items-center gap-2 text-lg font-semibold">
                <span className="grid h-7 w-7 place-items-center rounded-md bg-accent text-sm font-bold text-white">
                  CG
                </span>
                ClaimGuard
              </Link>
              <nav className="flex items-center gap-1 text-sm">
                {NAV.map((item) => (
                  <Link
                    key={item.href}
                    href={item.href}
                    className="rounded-lg px-3 py-1.5 text-slate-300 transition hover:bg-white/5 hover:text-white"
                  >
                    {item.label}
                  </Link>
                ))}
              </nav>
              <div className="ml-auto">
                <SystemChips />
              </div>
            </div>
          </header>
          <main className="mx-auto max-w-7xl px-6 py-8">{children}</main>
          <footer className="mx-auto max-w-7xl px-6 pb-10 text-xs text-muted">
            Synthetic data. Decisions shown here are model recommendations for investigators, not
            final settlement decisions.
          </footer>
        </div>
      </body>
    </html>
  );
}
