# How MCTS next-step prediction helps the current turn's response

*Plain-English explainer with a concrete worked example. Read this first if
you're new to the planner and you want to understand what MCTS actually
contributes to the agent's reply.*

## The short answer

MCTS doesn't insert clever text into the response prompt. **It changes
*which action* the prompt is written about, and which supporting context
that action drags in.** The downstream LLM gets a sharper, more constrained
prompt because the planner already eliminated bad options by simulating
their futures.

If you skip MCTS and let a baseline planner pick the action via a single
LLM call based only on the current turn, you get an action that *looks*
right now. With MCTS, you get an action that *also* tends to lead to good
outcomes 2-3 turns later. Everything downstream — the must-say phrases for
the chosen action, the retrieved style precedents, the pre-fetched external
data — gets pulled in based on whatever action was picked.

A weak action choice drags weak supporting context with it. A good action
choice drags good context.

## What the response LLM actually sees

The response generator's user prompt (from `backend/app/llm/prompts.py`,
function `response_gen_user_prompt`) has this structure:

```
ROLE: <agent role>
GOAL: <conversation goal>
KNOWLEDGE: <SOP knowledge block>

CHOSEN ACTION: <action name> - <action description>     ← The MCTS pick

MUST INCLUDE (proven phrases — adapt naturally):        ← Tied to the action
  - <must_say phrase 1>
  - <must_say phrase 2>

AVOID (negative-lift patterns):                          ← Tied to the action
  - <must_not_say phrase>

RELEVANT PAST AGENT RESPONSES (style reference):         ← Retrieved by cohort+state
  [precedent 1: action=X, cohort=Y, outcome=success]
  [precedent 2: action=X, cohort=Y, outcome=success]

HISTORY:
  <user/agent turns so far>

Write the agent's next utterance now.
```

Five things in this prompt come *from MCTS*:

1. **The `CHOSEN ACTION` line itself.** Not picked by gut feel — picked
   because rollouts simulated 8 alternate futures (or however many parallel
   rollouts × iterations the config calls for) and this action's average
   reward across all of those futures was highest.
2. **The action's `MUST INCLUDE` phrases.** Each SOP action declares
   phrases that historically worked. Once MCTS picks the action, the right
   phrase set is pulled into the prompt.
3. **The action's `AVOID` patterns.** Same idea, for what NOT to say.
4. **The `RELEVANT PAST AGENT RESPONSES` block.** Retrieved by similarity
   to the cohort+state that MCTS's classifier identified, filtered to
   precedents that used the same chosen action and ended in `success`.
5. *(When the speculative data-prefetch pipeline is enabled)* **Pre-fetched
   external data payloads** (e.g., the customer's account record, current
   competitor pricing) injected as additional context. The fetches were
   scheduled because earlier rollouts predicted this action would happen.

The response LLM doesn't see MCTS rollouts directly. It sees the
*consequences* of MCTS having run: a specific action, its tied supporting
material, and possibly data that's now ready to quote.

## Worked example: credit-card activation, turn 3

### Setup

The agent is on an outbound call to activate a new credit card. The
conversation is at turn 3. The user just said:

> *"I don't know, the annual fee makes me nervous. I'd rather think about it."*

Cohort classifier says: `PriceSensitive`.
User-state classifier says: `RaisingObjection`.

### What a baseline planner does

A baseline planner picks an action via one LLM call that looks only at the
current turn. The literal phrase "the annual fee makes me nervous" is the
most obvious match for the action called `ReassureFees`. Baseline picks
that.

Response prompt becomes:

```
CHOSEN ACTION: ReassureFees - Address customer concerns about pricing.
HISTORY: ...
```

Generated response (plausible):

> *"The fee is waived for the first year, and you can always cancel — no
> pressure!"*

This sounds fine in isolation. But in *aggregate*, sessions where the agent
picks `ReassureFees` at this kind of objection turn tend to end badly: at
turn 4 the user typically asks *"and what about year 2?"* — to which the
agent has nothing prepared — and by turn 5 the call has abandoned with the
card unactivated. The baseline planner can't see this because it doesn't
look ahead.

### What MCTS does

MCTS expands 3 candidate actions at turn 3, and for each runs ~8 rollouts
that simulate 3 turns into the future. The simulator generates a
hypothetical user response, the agent picks a follow-up action, and so on,
until a depth limit or terminal marker. Each rollout's outcome (success /
failure / how-far-it-got) becomes a reward signal back to the candidate at
turn 3.

Stylized result:

| Candidate at T3 | Simulated future (depth 3) | Avg reward |
|---|---|---|
| `ReassureFees` | user → *"but year 2?"* → agent vague → user abandons | 0.10 |
| `HandleObjection` | user → asks about benefits → agent quotes offer → user agrees | **0.65** |
| `PivotToBenefits` | user → *"you didn't answer me"* → user frustrated → abandons | 0.00 |

MCTS picks `HandleObjection`. Cohort+state classifier confirms
`PriceSensitive` / `RaisingObjection`. Three precedents are retrieved with
`(cohort=PriceSensitive, action=HandleObjection, terminal_outcome=success)`.
Meanwhile, the speculative-prefetch pipeline, which schedules fetches
based on earlier-turn rollout predictions, has already fetched
`tailored_offer` (a 4.2-second CRM API call) and `competitor_compare`
(a 4.8-second KG query) in the background. They're sitting in the per-
session cache.

Now the response prompt becomes:

```
ROLE: Outbound bank representative for ACME Bank
GOAL: Activate the customer's newly issued credit card during this call.
KNOWLEDGE: Activation requires confirming the cardholder's identity. The card
has a $0 annual fee for the first year and 2% cashback on groceries.

CHOSEN ACTION: HandleObjection - Address a concern the user raised
                                  (cost, security, time).

MUST INCLUDE (proven phrases — adapt naturally, don't quote verbatim):
  - acknowledge the concern explicitly before answering
  - quantify the benefit in dollar terms, not percentages
  - leave the cancellation option open

AVOID (negative-lift patterns):
  - dismissive language like "no pressure" or "totally fine"
  - over-promising about future years

RELEVANT PAST AGENT RESPONSES (style reference, do not copy):
  [PriceSensitive cohort, HandleObjection action, terminal=success]
    "Totally hear you on the fee. Here's what made it worth it for most
    customers in your situation: the cashback alone covered the fee within
    4 months..."
  [PriceSensitive cohort, HandleObjection action, terminal=success]
    "The fee concern is the most common one we hear — let me show you the
    math..."

PREFETCHED DATA AVAILABLE:
  - tailored_offer: "OFFER: tier=Platinum; $0 annual fee year-1, 2% grocery
                     cashback, 0% APR for 12mo on purchases, $200 sign-up
                     bonus if spend $1k in 90d"
  - competitor_compare: "COMPETITORS: Chase Sapphire(2.0% groc, $95 fee,
                         60k SUB), Citi DoubleCash(1.5% all, $0 fee), Amex
                         Everyday(2.0% groc cap $6k/yr)"

HISTORY:
  ... [previous 3 turns] ...
  user: "I don't know, the annual fee makes me nervous. I'd rather think
         about it."

Write the agent's next utterance now.
```

Generated response with all of that context available:

> *"I totally hear you on the fee — that's the question most customers in
> your situation ask first. Year one it's actually $0 — and the math works
> out: at 2% cashback on groceries, the average household earns about
> $260 a year right there, plus a $200 sign-up bonus if you put $1,000 on
> it in the first 90 days. By year 2 you'll have hard numbers in your hand,
> and you can absolutely close it then if it doesn't pencil out. Want me
> to walk you through what happens at the year-one mark?"*

This response now:

- **Acknowledges first, answers second** (from the must-say list).
- **Quantifies in dollars** ($260, $200), not percentages (must-say).
- **Quotes specific real numbers** from the prefetched `tailored_offer`
  payload — instantly, no live API wait.
- **Pre-answers the next-turn question explicitly** ("by year 2 you'll
  have hard numbers"), because MCTS rollouts predicted the year-2 follow-
  up was coming.
- **Avoids "no pressure" phrasing** (from the avoid list).
- **Keeps the cancellation option open** without sounding dismissive
  (must-say).

## What MCTS *actually contributed*, line by line

If you separate the contributions strictly:

| Contribution | Where it shows up in the response prompt | Without it |
|---|---|---|
| Picked `HandleObjection` over `ReassureFees` | `CHOSEN ACTION: HandleObjection …` | Would have been `ReassureFees` |
| Confirmed cohort = `PriceSensitive` | Filtered precedents block to `PriceSensitive` successes only | Generic or wrong-cohort precedents |
| Selected the must-say/avoid list for `HandleObjection` | The `MUST INCLUDE` and `AVOID` sections | Different action's must-say (less applicable) |
| Predicted next user state ≈ "WantsToHearBenefits" | Triggered prefetch of `tailored_offer` | Either no prefetch, or 4.2 s of live-fetch dead air, or no quantified offer at all |

The response-gen LLM is doing the same kind of writing work in both cases.
**The leverage is in everything around it** — which action's must-say list
shapes the response, which precedents anchor its style, which data is
sitting in cache ready to quote.

## The simplest mental model

> **MCTS is a smarter "what should I do next" decision. Everything
> downstream — the must-say phrases, the retrieved style, the prefetched
> data — is filed under whatever action MCTS picks. A weak action choice
> drags weak supporting context with it. A good action choice drags good
> context.**

The deeper the rollouts (e.g., depth 4-5 instead of 1), the more the
action picked accounts for what comes *after* this response — not just
what feels right *to* this response. That's where the lookahead value
lives.

## Where to dig further

- `backend/app/planner/responder.py` — how the response prompt is
  assembled and called.
- `backend/app/llm/prompts.py` — the literal `response_gen_user_prompt`
  template that defines the LLM input shape.
- `backend/app/planner/mcts.py` — how candidates are scored via rollouts,
  including bandit / NEE / WU-PUCT optimizations.
- `backend/app/planner/data_prefetch.py` — how the speculative-prefetch
  pipeline turns rollout predictions into background fetches.
- `notes/2026-05-23-first-successful-conversation-and-78pct-state-prediction.md`
  — measured numbers from real sessions: state-prediction accuracy,
  prefetch hit rates, what worked and what didn't.
