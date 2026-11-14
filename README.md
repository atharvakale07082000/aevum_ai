# Aevum AI

Decision analysis for the Aevum platform, built on [Strands Agents](https://strandsagents.com).

It powers four panels in the product: the **AI Summary** rail, the **Live
Insight** card, the **Consensus report** on a decision's history, and the
**Ask about this decision** box.

---

## The one idea worth reading before the code

**Every number is measured. The model only writes the words.**

The designs call for lines like *"80% of senior architects favour Hybrid Cloud"*
and *"84% probability of consensus"*. A language model will produce sentences
like that fluently and, often, wrongly. In a decision platform that is a
specific and serious failure: the analysis lands inside a permanent record, it
reads as official, and people act on it.

So the service is split down the middle:

| | Where | What it does |
|---|---|---|
| **Arithmetic** | `analysis.py` | Vote shares, turnout, lead margin, role divergence, risk level — computed from real ballots and real roles. No model involved. |
| **Language** | `agents/analyst.py` | Receives those figures as text and writes prose around them. The system prompt forbids producing any number that was not passed in. |

Two consequences fall out of that split, both good:

- With no model configured the service still answers. The figures are identical;
  the prose comes from templates and the response is marked
  `source: "deterministic"`. The interface says which it got. **Degraded, never
  wrong.**
- Risk levels ship with the reasons that produced them, so a reader can
  disagree with the judgement instead of taking it on faith.

The "likelihood of consensus" is a **stated heuristic, not a prediction**:

```
0.50 baseline
  + margin / 200                 a 100-point lead adds 0.50
  + (participation − 0.5) / 5    full turnout adds 0.10, none removes 0.10
  − 0.15                         when senior and member votes diverge
clamped to [0.05, 0.95]
```

The formula travels with the number in `likelihoodBasis`, and the UI shows it on
hover. A heuristic that explains itself is useful; one that doesn't quietly
acquires authority it never earned.

---

## Security posture

**This service has no identity of its own.** Every request carries the bearer
token and `X-Organization-Id` that arrived on the inbound call, and it reads the
decision back out of the Aevum REST API as that person.

That is deliberate. A service account with its own credentials would be a second
path into tenant data sitting outside `TenantFilter`, and the first bug in it
would be a cross-tenant leak. Borrowing the caller's token means **there is no
new boundary to get wrong** — if the backend refuses them, it refuses us, and
that refusal is passed straight back with its original error code.

The agent is given no tools. State is assembled by the caller and handed over
whole, so the model cannot reach for data outside what the caller was granted.

Per-voter ballots come from `GET /api/decisions/{id}/votes/all`, which is
**participant-only**: a tally is a group fact, but who voted for what is a
personal one. If that call is refused, the role-divergence claim is simply
omitted rather than estimated.

---

## Endpoints

All take `Authorization: Bearer <jwt>` and `X-Organization-Id`. Errors use the
Aevum backend's own `ApiError` envelope, so the frontend parses one contract.

| Method | Path | Panel |
|---|---|---|
| `GET` | `/health` | Reports `mode: model \| deterministic` |
| `POST` | `/ai/decisions/{id}/summary` | AI Summary rail — consensus risk, key considerations, metrics |
| `POST` | `/ai/decisions/{id}/insight` | The dark Live Insight card |
| `POST` | `/ai/decisions/{id}/report` | Consensus report on the timeline |
| `POST` | `/ai/decisions/{id}/ask` | Ask about this decision |

`POST` for reads is deliberate: these are expensive, non-idempotent generations,
not cacheable resources, and a `GET` would invite proxies and browsers to cache
an analysis that is only true for the minute it was made.

---

## Running it

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env           # then fill in AEVUM_AI_OPENROUTER_API_KEY
aevum-ai                       # or: uvicorn aevum_ai.app:app --port 8090
```

With no `AEVUM_AI_MODEL_PROVIDER` set it runs in deterministic mode: real
numbers, template prose, no API key needed. That is the right default for CI,
which should never need a key to run the suite.

`GET /health` says which mode it is in, and on which model:

```json
{ "status": "UP", "modelConfigured": true, "mode": "model",
  "provider": "openrouter", "model": "nvidia/nemotron-3-ultra-550b-a55b:free" }
```

### OpenRouter

The default provider. OpenRouter's endpoint is OpenAI-compatible.

```bash
AEVUM_AI_MODEL_PROVIDER=openrouter
AEVUM_AI_OPENROUTER_API_KEY=sk-or-v1-...
```

`openrouter.py` subclasses Strands' `OpenAIModel` against OpenRouter and adds
two throttles a free-tier key needs — a concurrency cap, because a decision room
opens four panels at once, and a sliding-window bucket sized to the free tier's
20 requests/minute. Without them a busy afternoon returns 429 on all four panels
simultaneously, which reads as the product being broken rather than as a quota.

Things that are not obvious and were expensive to learn:

1. **The model is pinned for a reason.** Every job here asks the agent for
   *structured output* — a typed narrative, never free text. Models that chat
   perfectly well can still fail Strands' structured output and **degrade
   silently rather than raising**. Validate a replacement against a real panel
   call, not against a prompt.
2. **Not a chain-of-thought model.** One will spend the entire token budget
   deliberating and return nothing at all.
3. **`enable_thinking: false`**, and a 3000-token budget rather than 1200.
   Thinking tokens never reach the reader, and on a structured-output call they
   can crowd out the JSON entirely; running out mid-turn fails the whole request
   instead of shortening the answer.

One thing is *not* carried over. The ai-tutor version wraps only `stream()`.
Everything here goes through `structured_output()`, so wrapping `stream` alone
would have left every real call unthrottled while looking correct. Both are
overridden, and a test asserts it.

Anthropic and Bedrock still work (`AEVUM_AI_MODEL_PROVIDER=anthropic` with
`AEVUM_AI_ANTHROPIC_API_KEY`, or `bedrock` with the standard AWS credential
chain). Every provider produces identical figures; only the prose changes.

### Why the analysis jobs are `async`

Strands' synchronous `Agent.__call__` runs each invocation in a **fresh event
loop on a new thread**, then blocks on the result. Called from an async FastAPI
endpoint that stalls the whole uvicorn worker for the length of a generation —
one slow panel freezes every other request in the process — and it puts the rate
limiter's `asyncio` primitives on a different loop each call, which raises
"Future attached to a different loop" rather than throttling anything.

So the four jobs are coroutines and use `agent.invoke_async(...)`. One loop, no
thread bridge, and a rate limiter that actually limits.

### Where the frontend points

The Next.js app reaches this service through its own server-side proxy at
`/api/ai/*`, which attaches the caller's token from an httpOnly cookie. Set
`AI_BASE_URL=http://localhost:8090` in the frontend's environment. The browser
never talks to this service directly.

---

## Tests

```bash
python -m pytest
```

No test needs an API key. The suite covers `analysis.py` exhaustively, because if the
arithmetic is wrong, every panel is confidently wrong and nothing downstream
will catch it, because the model is explicitly told to trust these figures.

---

## Layout

```
src/aevum_ai/
  analysis.py        every number the product shows. Pure functions, no model.
  schemas.py         wire contracts; errors reuse the backend's ApiError shape
  aevum_client.py    read-only client that borrows the caller's identity
  openrouter.py      the OpenRouter model, rate-limited on both entry points
  config.py          environment
  app.py             FastAPI surface
  agents/analyst.py  the Strands agent: system prompt, four jobs, fallbacks
tests/
  test_analysis.py   19 tests over the arithmetic
  test_openrouter.py 11 over throttling, wiring, and the deterministic fallback
```

---

## Known gaps

1. **Nothing is cached.** Each panel call re-reads the decision and re-runs the
   agent. Fine at current scale; a busy room will want a short-lived cache keyed
   on the decision's `updatedAt`.
2. **No streaming.** The Ask box waits for a complete answer. Strands supports
   streaming; the UI would need an incremental renderer.
3. **Ranked voting is not modelled.** The backend accepts a `rank` and discards
   it, so the analysis treats every ballot as a single choice.
4. **Considerations are drawn only from option descriptions.** An option with no
   description produces a thin consideration. Comments would be the natural
   richer source, and those are Phase 1.
