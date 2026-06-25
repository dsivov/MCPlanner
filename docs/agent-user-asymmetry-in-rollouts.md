# Agent/user asymmetry in MCTS rollouts (why this isn't chess)

*The most conceptually load-bearing distinction in the planner. If you came
from a game-tree-MCTS background, the search shape will look familiar but
the player relationship is fundamentally different. This doc explains how
and why, what we model today, and where the gaps are.*

## The setup

Standard game-tree MCTS (chess, Go, board games):

- Both players bound by the same formal rules.
- Both players want the *opposite* outcome (zero-sum or near-zero-sum).
- The search alternates max → min → max → min, with each side
  optimizing against the other.

Our planner:

- The **agent** is bound by the SOP — formal action vocabulary,
  edge-restricted transitions, declared `success_markers` and
  `failure_markers`.
- The **user** has no formal action vocabulary, no SOP-defined moves, no
  explicit "win condition." The user is a *participant in a conversation*,
  not an opponent in a game.
- In many real scenarios, the user and the agent want the *same* outcome
  (e.g., insurance renewal — both prefer to renew, just maybe on different
  terms). It's not zero-sum at all.

So the search alternation is `agent_action → user_response → agent_action
→ user_response → …`, but neither side is `min` against the other.
Calling this a "game tree" obscures more than it explains.

## How the user actually shows up in our rollouts

Three layers do the user-modeling work:

### Layer 1 — `task.user_profile` (the natural-language identity)

Every user-sim call in a rollout receives:

```
YOU ARE THE USER. PROFILE:
  <task.user_profile.description>
Demographics: <task.user_profile.demographics>

Agent is trying to: <task.conversation_profile.goal>
HISTORY (you are 'user'):
  <history so far, real + simulated>
Write your next reply as the user.
```

The profile's prose description carries the user's *intent stance*
implicitly. For `car_insurance_renewal`, the profile reads roughly:

> *"A current SafeWay policyholder receiving an outbound renewal call.
> Their default disposition is to renew but they may have specific
> concerns about price, coverage, or recent life changes."*

That sentence is doing a lot of work. It's where the *agent and user
share an intent* (renew the policy) is encoded. The user-sim reads this,
internalizes "I'm someone who probably wants to renew," and behaves
accordingly. The agent's goal is also shown explicitly so the simulated
user can react sensibly to it.

### Layer 2 — Cohort assignment (the disposition label)

At the start of each turn, a cohort classifier maps the user to one of
the SOP's cohort vocabulary. For car insurance:

| Cohort | Intent alignment with agent goal |
|---|---|
| `LoyalCustomer` | Aligned — wants to renew |
| `PriceShopper` | Partially aligned — wants to renew at the right price |
| `LifeChange` | Mixed — has circumstantial obstacles |
| `CoverageConcerned` | Aligned but cautious |
| `Skeptical` | Neutral / needs persuasion |
| `Hurried` | Aligned but deferring |
| `WrongPerson` | Blocked — can't transact |

The cohort flows into the rest of the system mainly via **precedent
retrieval**: when we retrieve "similar past sessions" for the response-gen
prompt, we filter by cohort. So the cohort indirectly shapes the agent's
response style by selecting which past trajectories anchor it.

### Layer 3 — `task.user_states` (the state vocabulary)

The user-simulator's free-form text gets passed to a *state classifier*
that maps it to one of the SOP's `user_states` vocabulary. State
membership is then checked against `success_markers` and `failure_markers`
— and *that's* what terminates the rollout and assigns the reward.

So the user is:

- Free-form at the **text** level (anything they could plausibly say).
- Constrained at the **state** level (any text maps to one of a fixed
  set of states).
- Categorical at the **disposition** level (cohort label, classified
  separately).

Three different vocabularies, three different grain sizes, all working
together.

## What this gives us — scenarios we handle

| Scenario | How rollouts cope |
|---|---|
| User aligned with agent goal | Cohort `LoyalCustomer`. User-sim's profile encodes cooperative disposition. Rollouts converge on `success_markers` quickly. |
| User aligned but with conditions | Cohort `PriceShopper`. Rollouts simulate price objections, hit `HandlePriceObjection` paths, then `AgreedToRenew`. |
| User pulled away by external factor | Cohort `LifeChange`. Rollouts go through life-change-reporting paths, may end in success or escalation depending on simulated trajectory. |
| User can't transact | Cohort `WrongPerson`. Rollouts terminate on the `WrongPerson` failure marker. |
| User wants to defer | Cohort `Hurried`. Rollouts converge on `ScheduleCallback` (soft outcome). |

The key win here: we don't need to write *rules* for each user disposition.
The cohort label + profile prose + precedent retrieval combine to handle
it implicitly.

## What we DON'T handle (honest gaps)

1. **No formal "user intent" field separate from the profile prose.** The
   alignment-vs-misalignment signal lives in natural language, not in a
   structured field MCTS can reason about. There's no
   `user_profile.alignment: "aligned" | "mixed" | "opposed"` schema entry.

2. **No adversarial sampling.** Chess MCTS considers the opponent's
   *best* response (min). Our rollouts sample the *average* user-sim
   response. There's no "what if this user is more obstinate than usual?"
   stress-test mode. The user is always the median user from the profile,
   biased toward continuity.

3. **No mid-rollout cohort updates.** If a `Skeptical` user becomes
   convinced mid-rollout, the cohort label doesn't update. Only the
   state does. In practice the dynamic signal lives in state transitions,
   but the cohort being frozen is a modelling gap that compounds with
   point #2 — every rollout in a search shares the same cohort, so
   diverse-user simulations don't exist.

4. **No asymmetric search-depth weighting.** A chess engine spends equal
   search depth on its moves and the opponent's. We alternate at fixed
   ratio (one agent action per one user response), but never explore
   "what's the *worst* user response we could see here?" That's a
   meaningful tool for risk-averse planning — we don't have it.

## Where this asymmetry surfaces in the architecture

Three places:

### Reward composition

`combine_reward(rationality, progress)` weighs the agent's progress toward
`success_markers` (the goal) with a "user rationality" score (how
plausible the simulated user behaviour was). **The user isn't a reward-
maximizer.** It's a reward *gateway*. The user's job in a rollout is to
react plausibly so the agent's outcome can be scored.

### Action policy

`rollout_action_policy` ∈ {`bandit`, `llm_top1`, `llm_topk`} only governs
the **agent** side. The user side is always the same
`simulate_user_with_state` LLM call. There's no concept of a "bandit
policy for the user-simulator" — we don't pretend the user is optimizing
anything.

### The state-prediction null result

The asymmetry note from 2026-05-23 documented a measured null result:
**100% accuracy on stable phases, 0% on transitions**, with top-K
aggregation producing the *same* numbers as modal because rollouts
emit degenerate (point-mass) state distributions.

This null result is downstream of *exactly* the asymmetry being discussed.
Because:

- The user-sim is profile-conditioned and continuity-biased (point #2
  above).
- Eight parallel rollouts at the same step receive the same profile and
  the same agent action → they all produce essentially the same predicted
  user response.
- Distribution collapses to a point mass.
- Top-K hedging has nothing to hedge across.

The fix isn't aggregation — it's on the user-sim side. Higher
temperature, multi-sample, or explicit disposition-diversity prompts.
This is the *next* research target.

## How to do this better (production-design implications)

These belong in a production conversation but flow directly from the
asymmetry:

1. **Make cohort persistent inside rollouts.** Today cohort is classified
   at the root and used for precedent retrieval. It is *not* injected
   into each rollout's user-sim prompt. Doing so ("You are a
   PriceShopper-cohort user — be specifically price-resistant") would
   keep simulated users in character across rollout depth.
2. **Sample disposition diversity per rollout.** At rollout start, draw
   a "user mood" from a distribution conditioned on observed cohort, and
   *freeze it* for that rollout. Different rollouts get different mood
   samples. This produces the non-degenerate distributions that top-K
   hedging needs.
3. **Add an explicit `user_profile.goal_vs_agent` field to the SOP
   schema.** First-class "aligned / mixed / opposed" indicator that
   MCTS can reason about as a categorical variable. Maps to whether
   `success_markers` are *easy* or *hard* for this scenario.
4. **Optional: adversarial rollout mode.** For risk-averse planning, run
   a subset of rollouts with an explicit "be more skeptical / be more
   hurried" prompt. Worst-case-against-adversary style. Most useful when
   the cost of getting a transition wrong is high (e.g., voice agents
   where a stall is fine but a wrong commitment is costly).

## The mental model

> **The agent plays a game with formal rules. The user is a participant
> in that game with no formal rules — only an implicit profile-encoded
> disposition. The cohort classifier translates that disposition into a
> categorical label that flows into precedent retrieval. The
> success/failure markers define what "winning" means for the agent; the
> user has no parallel concept. So the asymmetry isn't a bug to fight
> — it's a different game shape entirely from chess, and most production
> conversation-planner work boils down to modelling that shape well.**

## Related research findings

Several of the dated research notes touch on this asymmetry:

- `notes/2026-05-23-stable-vs-transition-state-prediction-asymmetry.md`
  — the null result that motivated this doc: top-K aggregation didn't
  help because the rollout user-sim is deterministic-per-prompt, which
  is itself a symptom of profile-only user modelling without disposition
  sampling.
- `notes/2026-05-23-first-successful-conversation-and-78pct-state-prediction.md`
  — first successful session: a 78% state-prediction headline that was
  inflated by stable phases, while transition accuracy was 0%. The
  transitions are exactly the moments where user-disposition asymmetry
  matters most.
- `notes/2026-05-23-voice-agent-production-architecture.md`
  — the production architecture's "top-K branches in the blackboard" hedge
  requires the user-side diversity we don't yet have. The asymmetry is
  the upstream blocker on that hedge being meaningful.

## Where to dig further

- `backend/app/planner/user_sim.py` — the three user-simulator entry
  points (basic, with-state, end-of-rollout). All are profile-conditioned
  LLM calls.
- `backend/app/llm/prompts.py` — the `user_sim_user_prompt` and related
  templates. This is where the agent's-goal hint and the profile are
  woven in.
- `backend/app/planner/mcts.py` — the rollout loop, where user-sim calls
  are made per step and the result is fed into the next agent-action
  decision.
- `data/sops/*.json` — the SOPs themselves, including each one's
  `user_profile`, cohort vocabulary, and success/failure markers.
- Companion docs: [how-mcts-helps-current-turn.md](how-mcts-helps-current-turn.md),
  [how-prefetch-reads-mcts-rollouts.md](how-prefetch-reads-mcts-rollouts.md).
