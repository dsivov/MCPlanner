# How prefetch reads MCTS rollouts (when MCTS only returns one action)

*Plain-English explainer with a chess analogy and the actual code wiring.
Companion to* `how-mcts-helps-current-turn.md` *— that doc covers what
MCTS contributes to the current reply; this one covers how the prefetch
system gets multi-turn lookahead from a planner that only "returns" one
action.*

## The natural question

MCTS's job is to pick the best action for the *current* turn. The signature
of the planner call is roughly:

```python
chosen_action: str = await run_mcts(...)
```

It returns a single action label. So how does the prefetch system — which
needs to schedule fetches for the next 1-3 turns ahead — know what actions
are coming?

## The trick: we don't read MCTS's return value, we read its by-products

MCTS picks its action by **simulating dozens of multi-turn futures** and
choosing whichever first-action leads to the best average outcome. Those
multi-turn simulations are called *rollouts*. Each rollout is a list of
hypothetical actions the agent would take if this branch were chosen.

When MCTS finishes, those rollouts are still sitting in memory. The
prefetch system reads them.

## The chess analogy

Imagine asking a chess engine "best move?" and getting `"Nf3"` back.
That's all the engine *returns*. But to pick `Nf3` it considered ten
thousand game continuations internally. We peek at those internal lines and
see: *"after Nf3, opponent likely plays Nc6, then I play d4, then..."* We
use that lookahead to set up our broader game plan — even though the engine
itself only emitted one move.

Same with our planner:

- MCTS *returns* `"HandleObjection"`.
- During the search, it generated 8 rollouts. Each is a trajectory like
  `["HandleObjection", "PitchActivation", "RequestActivation"]`.
- The prefetch system reads those trajectories *after the search finishes*
  to know which actions are likely upcoming.
- For each likely upcoming action, the SOP declares which external data
  it needs. Prefetch fires those fetches in the background.

## What the rollouts actually look like in memory

After MCTS finishes, the `ExperimentLogger` holds something like:

```python
logger.rollouts = [
    RolloutEntry(planned_actions=["HandleObjection", "PitchActivation",  "RequestActivation"], reward=0.65, ...),
    RolloutEntry(planned_actions=["HandleObjection", "PitchActivation",  "ConfirmActivation"], reward=0.60, ...),
    RolloutEntry(planned_actions=["HandleObjection", "ScheduleCallback"],                       reward=0.20, ...),
    RolloutEntry(planned_actions=["ReassureFees",    "ScheduleCallback"],                       reward=0.10, ...),
    RolloutEntry(planned_actions=["PivotToBenefits", "HandleObjection"],                        reward=0.00, ...),
    ...
]
```

Each `RolloutEntry` has a `planned_actions` list and a `reward` (how well
that imagined future went). The return value of MCTS — `"HandleObjection"`
— was picked because the rollouts that started with `HandleObjection`
averaged the highest reward. But the lists themselves are still here to
read.

## How prefetch reads them — five steps

Right after MCTS picks `HandleObjection`, before the response is generated,
the prefetch system runs in the background. The logic, in plain language:

```
1. Look at all rollouts whose first action == "HandleObjection"
   (the chosen action — the rollouts of paths we won't take are
   discarded because they're no longer reachable).

2. For each surviving rollout, walk forward through planned_actions:
     planned_actions[1] is the predicted next action  (offset = 1)
     planned_actions[2] is the predicted action after that  (offset = 2)
     ...

3. Vote-count: at each offset, sum how many rollouts predict each action.
   Weight each vote by the rollout's reward.

     offset=1: {PitchActivation: 0.65 + 0.60 = 1.25, ScheduleCallback: 0.20}
                → PitchActivation wins, with 1.25/1.45 = ~86% probability
     offset=2: {RequestActivation: 0.65, ConfirmActivation: 0.60}
                → RequestActivation wins, with 0.65/1.25 = ~52% probability

4. For each likely future action, look up its data_dependencies in the SOP:
     PitchActivation     → needs ["customer_account", "tailored_offer"]
     RequestActivation   → needs ["activation_template"]

5. Schedule background fetches for those deps. They run in parallel with
   response_gen and the user's typing/thinking time.
```

The closing step glues two things together that on their own would be
useless:

- **The rollouts know which *actions* are likely** (this offset-N is
  PitchActivation with 86% confidence) but say nothing about data.
- **The SOP knows which *data* each action needs** (PitchActivation needs
  customer_account, tailored_offer) but says nothing about likelihood.

Combine them and you get: *fetch customer_account + tailored_offer right
now, in the background, because there's an 86% chance the agent will need
them next turn.*

## The minimal code wiring

```python
# After MCTS picks chosen_action (in chat.py)
rollouts_snapshot = list(logger.rollouts)           # the by-product

# The predictor reads rollouts and emits (action, offset, probability) tuples
predictor = MctsTrajectoryPredictor(
    rollouts=rollouts_snapshot,
    chosen_action=chosen_action,
)
predictions = await predictor.predict(max_offset=3)
# → [TrajectoryPrediction(action="PitchActivation", offset=1, probability=0.86, ...), ...]

# Plan-build: for each predicted action, look up the SOP's data_dependencies
plan = build_prefetch_plan_from_predictions(predictions, task=task)
# → [PrefetchPlanItem(dep="customer_account", offset=1, action="PitchActivation", confidence=0.86), ...]

# Schedule background fetches against the registered fetcher per dep.kind
await data_prefetch_manager.schedule(experiment_id=exp.id, plan=plan, ...)
```

That's the whole story — about 5 lines of glue. The hard work was done by
MCTS *during* the search; the prefetch system just harvests the search
tree's by-products.

## The simplest mental model

> **MCTS *returns* one action. But to pick it, MCTS had to imagine 8
> multi-turn futures. Those imagined futures are the prefetch system's
> input — the rollouts know "if HandleObjection now, then PitchActivation
> next, then RequestActivation," and the SOP knows PitchActivation needs
> a CRM call. So while the agent is responding to *this* turn, the
> background is already fetching the CRM data for the *next* turn.**

## A wrinkle worth knowing about

This works only as well as the rollouts are *predictive*. Two failure modes
to watch for:

1. **All rollouts agree on the wrong future.** If the rollout user-simulator
   is biased toward a particular continuation, every rollout will sample
   the same path. The prefetch system confidently fetches the wrong things.
   The cure is sampling diversity in the rollouts (see the research note
   on *stable-vs-transition state-prediction asymmetry* for the current
   state of this).
2. **Rollouts disagree wildly.** Each (offset, action) prediction has low
   confidence, so few make it past the `min_confidence` threshold. Few
   fetches scheduled. This is fine for cost — we just don't get much
   latency hiding when MCTS itself is uncertain.

The right way to think about prefetch confidence is: it inherits from the
underlying rollouts' agreement. If the rollouts are confident-and-right,
prefetch is fast-and-accurate. If they're confident-and-wrong, prefetch is
wasteful. If they're uncertain, prefetch is cautious.

## Where to dig further

- `backend/app/planner/mcts.py` — where the rollouts are generated. See
  `RolloutOutcome` and the main rollout loop in `_rollout()`.
- `backend/app/planner/trajectory_predictor.py` — `MctsTrajectoryPredictor`
  and the `build_prefetch_plan_from_predictions` helper.
- `backend/app/planner/data_prefetch.py` — `DataPrefetchManager` (the
  per-session background queue) and the fetcher registry.
- `backend/app/schemas.py` — `DataDependency` declaration and how each
  `agent_action` declares its `data_dependencies`.
- Companion doc: [how-mcts-helps-current-turn.md](how-mcts-helps-current-turn.md)
  — what MCTS contributes to the response of the *current* turn.
