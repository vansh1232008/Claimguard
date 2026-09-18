# ClaimGuard — interview notes

Simple answers you can say out loud. Short sentences. No jargon unless the interviewer uses
it first.

---

## 1. "Tell me about this project" (60 seconds)

> Insurance investigators check claims by hand. They open the claim form, the repair
> invoice, the damage photos and the policy document, and compare them. It is slow, and
> organised fraud rings still get through, because each claim looks fine on its own.
>
> ClaimGuard automates the first pass. Seven agents run one after another. They read the
> claim, check the invoice against the claim form, look at the damage photos, find the right
> policy clauses, run a fraud model, and look for links to other claims. At the end it says
> approve, review or reject, and it gives the reason and the evidence.
>
> I built the backend in FastAPI, the agents on LangGraph, the fraud model in XGBoost, the
> graph in Neo4j, and the dashboard in Next.js.

---

## 2. "Why seven agents? Why not one big prompt?"

> Three reasons.
>
> One, each agent is small, so I can test it alone. If the document agent breaks, the rest
> still run.
>
> Two, they need different tools. The policy agent needs search over the policy document.
> The damage agent needs a vision model. The anomaly agent needs XGBoost. One prompt cannot
> do all of that.
>
> Three, I can skip work. If a claim is low risk and not connected to any other claim, the
> graph agent is skipped. That saves one model call on most claims.

---

## 3. "Why a graph, and not just a rule?"

> Because a fraud ring is invisible one claim at a time. Each claim is small and the story
> is fine. But four different people sharing one bank account, all repairing at the same
> garage, is a pattern you can only see when you put the claims side by side.
>
> So I build a graph. Claims are nodes. An edge means the two claims share something — a
> phone number, a bank account, an address, a repair garage. Then I look for connected
> groups.
>
> I weight the links. Sharing a bank account is strong evidence. Sharing a garage is weak,
> because a popular garage has hundreds of honest customers. In fact I cap it: if a garage
> appears on more than 25 claims, I do not create those edges at all. That one rule removed
> most of my false rings. There is a test for it.

---

## 4. "What did the graph actually buy you?" (the ablation)

Run `python ml/evaluate.py`. It trains the same model three ways and prints the table.

> I measured it instead of assuming it. I trained the same model three ways: claim features
> only, then plus hand-made graph features, then plus GraphSAGE embeddings.
>
> Adding graph features raised PR-AUC from about 0.28 to about 0.43. More importantly, ring
> capture went from about half the rings to almost all of them. Ring capture means: of the
> fraud rings in the test period, how many had at least one member flagged. That is the
> number that matters in practice, because once you catch one member you investigate the
> whole cluster.

**Use the numbers your own run prints, not the numbers above.** They shift with the seed and
the dataset size.

---

## 5. "How do you know the numbers are real?"

This is the question that separates a real project from a demo. Three honest answers:

> First, the split is by time, not random. I sort claims by incident date and hold out the
> last 20%. If I split randomly, members of the same fraud ring end up in both train and
> test, and the model looks much better than it is.
>
> Second, I made the data hard on purpose. About a fifth of the fraudulent claims look
> completely normal, and about one in seven honest claims looks suspicious. My first version
> did not have that overlap and the model got ROC-AUC 1.0 — which told me the data was
> broken, not that the model was good.
>
> Third, I pick the threshold on a validation fold, never on test. And I report several
> operating points, because the right threshold is a business decision about how many
> investigators you have.

---

## 6. "Where did the data come from?"

> It is synthetic, and I say that openly. There is no public motor-insurance dataset with
> labelled fraud rings — nobody publishes that.
>
> So I generated it. The population is shaped like freMTPL2, the French motor insurance
> dataset: vehicle power, vehicle age, driver age, bonus-malus, region. On top of that I
> added two fraud layers. One is individual — padded invoices, late reporting, thin
> descriptions. One is organised — rings whose members share identity details.
>
> The honest framing is: the *detection system* is the project. The data is the test bench I
> built to prove the system works.

---

## 7. "Why RAG for the policy? Why not put the policy in the prompt?"

> Two reasons.
>
> The wording is long, and most of it is irrelevant to any one claim. Retrieval gets the
> four clauses that matter.
>
> But the real reason is the citation. I chunk by clause, not by paragraph, so every chunk
> keeps its clause ID. When the system declines a claim it says "excluded under MC-3.5,
> water ingress". A decline letter has to point at a clause. A summary of the policy is not
> good enough.

---

## 8. "How do you stop the LLM from making up a decision?"

> Three layers.
>
> The LLM never sees a free-form question. Every call asks for JSON with a fixed schema, and
> the context is only what the agents found.
>
> The score is not the LLM's. It comes from XGBoost plus the signals the agents raised, with
> a fixed formula. The LLM writes the explanation; it does not choose the number.
>
> And two rules sit above it that it cannot override. If the policy does not cover the loss,
> the claim is never approved, whatever the model says. And any claim above the high-value
> ceiling always goes to a human.

---

## 9. "What if Gemini is down, or you have no key?"

> The whole thing still runs. Every external service has a fallback: a rule-based reasoner
> instead of Gemini, an in-memory index instead of Qdrant, NetworkX instead of Neo4j, SQLite
> instead of Postgres, a logistic scorecard instead of the trained model.
>
> The fallbacks are real logic, not fake responses, so the output is meaningful and
> reproducible. And the API tells you which backend is live, so nobody mistakes offline
> output for production output.
>
> I did this because a portfolio project that needs five API keys to start is a project
> nobody ever runs.

---

## 10. Bugs I actually hit (good to mention — it proves you built it)

Pick one or two. Interviewers like these more than the architecture.

**The address collision.** My graph had 155,000 edges for 20,000 claims. Far too dense. The
cause was the data generator: addresses were drawn from a pool of about 1,400, so thousands
of unrelated customers shared one, and the graph filled with meaningless "shared address"
links that buried the real rings. I made addresses near-unique and the edge count dropped to
about 20,000, which is what a real portfolio looks like.

**The GraphSAGE loss that would not go down.** It sat flat around 3.7. I was L2-normalising
the embeddings inside the forward pass but ignoring that step when I computed the gradient,
so the gradient did not match the loss. I moved normalisation to inference only. The loss
dropped from 6.4 to 2.9 and link-prediction accuracy went to about 0.89. Caching the first
layer at the same time took training from 79 seconds to 2.5.

**The LangGraph node that never ran.** All seven agents silently did nothing. LangGraph reads
a node function's type hints to decide what to pass it, and I had annotated my wrapper with
my own TypedDict — so it handed the node a filtered state without the claim ID. Removing the
annotation fixed it. Lesson: a framework that inspects your type hints will surprise you.

**Perfect accuracy is a bug report.** My first model got ROC-AUC 1.0. I did not celebrate; I
went looking for the leak. Fraudulent claims were drawn from a different pool of description
texts, so the model just learned the vocabulary. I made both classes share one text pool.

---

## 11. Numbers to know

Run these before the interview and write the real answers here:

| Thing | Where to get it | Your number |
|---|---|---|
| Claims in benchmark | `make dataset` | 150,000 |
| Fraud rate | generator output | ~8% |
| ROC-AUC | `make evaluate` | |
| PR-AUC (base → graph) | `make evaluate` | |
| Ring capture rate | `make evaluate` | |
| Policy clauses indexed | `GET /api/system/info` | 34 |
| Agents per claim | claim detail page | 6–7 |
| LLM calls per claim | claim detail page | |
| Latency per claim | claim detail page | |

---

## 12. Honest CV wording

Write the bullet from the numbers `ml/evaluate.py` gives you. A template that stays true:

> Built a seven-agent claims-investigation pipeline (LangGraph, Gemini, Qdrant policy RAG,
> XGBoost anomaly model, Neo4j relationship graph) that returns explainable
> approve/review/reject recommendations with clause-level citations; graph features lifted
> PR-AUC from **X** to **Y** and ring capture from **A%** to **B%** on a held-out
> time-based split of a 150,000-claim benchmark.

Two rules for this bullet:

1. **Never quote a number you cannot reproduce on your laptop in front of them.** If they
   ask "how did you measure that" and you cannot rerun it, the whole CV becomes suspect.
2. **Name the ablation, not just the headline.** "Graph features lifted PR-AUC from 0.28 to
   0.43" is far more convincing than "94% recall", because it shows you measured a
   contribution rather than reporting a single flattering number.
