"""TrajectoryPredictor — pluggable source of "what actions will the agent take at
future turn offsets, given the just-chosen action".

Decoupling this from the rollout-mode-specific data path means the speculative
data-prefetch pipeline works under value-mode (which produces no MCTS trajectories)
as well as simulate/hybrid mode.

Two implementations ship here:

- MctsTrajectoryPredictor — uses the in-memory rollouts already produced by simulate /
  hybrid mode. Same signal as the original derive_prefetch_plan.
- EmpiricalTrajectoryPredictor — queries precedent_traces for empirical "action at
  turn N → action at turn N+offset" transitions, grouped by cohort + sop_ref. Works
  with any rollout mode, including value-mode. Becomes increasingly accurate as
  precedent data accumulates.

The chat-route picks the right predictor based on what's available (router-pattern):
  - simulate/hybrid + at least one deep rollout → MctsTrajectoryPredictor
  - else if enough precedent data → EmpiricalTrajectoryPredictor
  - else: no plan (cold start)
"""

from __future__ import annotations
import asyncio
import math
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from sqlalchemy import select, func, text
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import PrecedentTrace, TurnRecord


@dataclass
class TrajectoryPrediction:
    """One predicted (action, offset, probability) tuple.

    `probability` is the predictor's estimate that this action will be the actual
    agent move at `current_turn_index + offset`. Used to weight prefetch scheduling.
    """
    action: str
    offset: int                    # 1, 2, 3 … turns ahead of "now"
    probability: float             # in [0, 1] (or [0, 1+] for unbounded reward signals)
    source: str = "unknown"        # which predictor produced this — useful for analysis
    # Modal predicted user state at this offset (the rollout's conditioning context for
    # this action). Empirical inherits the MCTS-published hint inside Union mode.
    predicted_user_state: str | None = None
    # Full per-state distribution at this (offset, action): {state_name: normalized share}.
    # Sums to 1 across states that voted on this (offset, action). The modal field above is
    # argmax(predicted_user_state_dist). Surfaced so consumers can hedge across the top-K
    # plausible state branches — critical for prefetch hedging on transition turns where the
    # mode is biased toward continuity.
    predicted_user_state_dist: dict[str, float] = field(default_factory=dict)
    # Q6b: the rollout-derived simulated user utterance at this (offset, action). Used as
    # the seed text for query-aware data prefetch (rendered into DataDependency.query_template).
    # Picked as the highest-reward rollout's text when multiple rollouts agree on this
    # (offset, action). Empty when no rollouts produced text (e.g., value mode).
    predicted_user_text: str | None = None


class TrajectoryPredictor(ABC):
    """Returns a flat list of predictions across (action, offset) pairs."""

    name: str = "abstract"

    @abstractmethod
    async def predict(self, *, max_offset: int = 3) -> list[TrajectoryPrediction]:
        ...


# ----------------------------- MCTS in-memory ------------------------------


class MctsTrajectoryPredictor(TrajectoryPredictor):
    """Reads predictions from the rollouts list left in the ExperimentLogger after
    a turn finishes. Confidence = mean rollout reward × exp(-decay·offset) over
    rollouts whose first_action == the just-chosen action.

    The empirical "probability" output is the normalized share of action votes at
    each offset, not the raw confidence — that way scoring across predictors is
    apples-to-apples.
    """

    name = "mcts"

    def __init__(self, rollouts: list, chosen_action: str, decay_lambda: float = 0.3):
        self.rollouts = rollouts or []
        self.chosen_action = chosen_action
        self.decay_lambda = decay_lambda

    # Top-K = how many state branches per (offset, action) to publish in the distribution.
    # K=3 fits the typical rollout count (8) without exposing one-rollout noise tails.
    TOP_K_STATES = 3

    async def predict(self, *, max_offset: int = 3) -> list[TrajectoryPrediction]:
        # Aggregate (offset, action) → cumulative reward-weighted score; track per-state
        # weight so we can recover both the modal state AND the full top-K distribution
        # that conditioned each (offset, action).
        cumul: dict[tuple[int, str], float] = defaultdict(float)
        weight_per_offset: dict[int, float] = defaultdict(float)
        state_weight: dict[tuple[int, str, str], float] = defaultdict(float)  # (offset, action, state) → weight
        # Q6b: track the highest-reward rollout's user_text per (offset, action). We pick
        # the best (rather than concatenating) so the query template gets a coherent
        # utterance, not a mash-up.
        best_text: dict[tuple[int, str], tuple[float, str]] = {}  # → (best_weight_so_far, text)
        for r in self.rollouts:
            planned = getattr(r, "planned_actions", None) or []
            if not planned or planned[0] != self.chosen_action:
                continue
            states = getattr(r, "planned_states", None) or []
            user_texts = getattr(r, "planned_user_texts", None) or []
            reward = float(getattr(r, "reward", 0.0) or 0.0)
            for offset, action_name in enumerate(planned[1:max_offset + 1], start=1):
                discount = math.exp(-self.decay_lambda * offset)
                w = reward * discount
                cumul[(offset, action_name)] += w
                weight_per_offset[offset] += w
                if offset < len(states) and states[offset]:
                    state_weight[(offset, action_name, states[offset])] += w
                # The user text aligned with this action lives at planned_user_texts[offset-1]
                # (it's the user-sim reply produced AFTER agent took planned[offset]).
                text_idx = offset - 1
                if 0 <= text_idx < len(user_texts) and user_texts[text_idx]:
                    prev = best_text.get((offset, action_name))
                    if prev is None or w > prev[0]:
                        best_text[(offset, action_name)] = (w, user_texts[text_idx])
        # Build per-(offset, action) state distribution (normalized).
        # Modal field stays for back-compat; new dist field carries the top-K tail too.
        state_dist: dict[tuple[int, str], dict[str, float]] = defaultdict(dict)
        for (offset, action_name, state), w in state_weight.items():
            state_dist[(offset, action_name)][state] = w
        modal_state: dict[tuple[int, str], str | None] = {}
        top_k_dist: dict[tuple[int, str], dict[str, float]] = {}
        for key, d in state_dist.items():
            total = sum(d.values()) or 1.0
            norm = {s: round(w / total, 4) for s, w in d.items()}
            modal_state[key] = max(norm, key=norm.get) if norm else None
            # Truncate to top-K, preserving descending order
            top_k_dist[key] = dict(sorted(norm.items(), key=lambda x: -x[1])[: self.TOP_K_STATES])

        out: list[TrajectoryPrediction] = []
        for (offset, action), w in cumul.items():
            denom = weight_per_offset[offset] or 1.0
            prob = w / denom
            txt_entry = best_text.get((offset, action))
            out.append(TrajectoryPrediction(
                action=action, offset=offset, probability=prob, source=self.name,
                predicted_user_state=modal_state.get((offset, action)),
                predicted_user_state_dist=top_k_dist.get((offset, action), {}),
                predicted_user_text=(txt_entry[1] if txt_entry else None),
            ))
        out.sort(key=lambda p: (p.offset, -p.probability))
        return out


# ----------------------------- Empirical from DB ------------------------------


class EmpiricalTrajectoryPredictor(TrajectoryPredictor):
    """Queries precedent_traces for the empirical distribution of "action at turn
    N+offset" given "action at turn N == chosen_action" (filtered by sop_ref + cohort).

    Each offset is computed independently — this is a marginal-distribution
    approximation, not a full Markov chain. Cheap to compute, no LLM call. Becomes
    accurate as precedents accumulate.

    Falls back to cohort-agnostic if no precedents match the (sop_ref, cohort, action)
    triple. Returns empty list if there's also no SOP-level data.
    """

    name = "empirical"

    def __init__(
        self,
        db: AsyncSession,
        *,
        sop_ref: str,
        cohort: str,
        chosen_action: str,
        min_supporting: int = 3,
        mood: str | None = None,
    ):
        self.db = db
        self.sop_ref = sop_ref
        self.cohort = cohort
        self.chosen_action = chosen_action
        self.min_supporting = min_supporting
        # Phase-2 mood conditioning. When provided, the SQL primarily restricts to
        # precedents with matching (cohort, mood). Falls back through cohort-only,
        # sop-only on sparse hits — see `_lookup_distribution` and the fallback chain
        # inside `predict()`.
        self.mood = mood or None

    async def predict(
        self,
        *,
        max_offset: int = 3,
        state_hints: dict[int, str] | None = None,
        state_hints_topk: dict[int, list[str]] | None = None,
    ) -> list[TrajectoryPrediction]:
        """Predict (action, offset) distributions from precedent_traces.

        When `state_hints_topk` is provided (per-offset list of likely states from the
        MCTS-side of Union), runs the SQL once per hinted state and emits one prediction
        per (offset, action, hinted_state). This is the multi-branch hedge that lets
        prefetch cover the transition cases where the modal state would silently miss.

        When `state_hints` is provided (legacy single-hint path) or nothing is provided,
        falls back to the simpler one-query-per-offset behaviour.

        Each offset tries the conditioned query first; if it returns < min_supporting rows
        the offset falls back to the state-blind query so a sparse hint doesn't kill recall.
        """
        out: list[TrajectoryPrediction] = []
        for offset in range(1, max_offset + 1):
            # Build the list of hinted states for this offset, in order of preference.
            hints: list[str | None] = []
            if state_hints_topk and state_hints_topk.get(offset):
                hints = [s for s in state_hints_topk[offset] if s]
            elif state_hints and state_hints.get(offset):
                hints = [state_hints[offset]]
            if not hints:
                hints = [None]

            offset_emitted = False
            for hint in hints:
                # Fallback chain (most-specific first), per (offset, state-hint):
                #   1. cohort + state + mood
                #   2. cohort + state          (drop mood)
                #   3. cohort                  (drop state)
                #   4. sop only                (drop cohort)
                dist: list[tuple[str, int, float]] = []
                if hint and self.mood:
                    dist = await self._lookup_distribution(
                        offset, cohort_required=True, state_hint=hint, use_mood=True,
                    )
                    if sum(c for _, c, _ in dist) < self.min_supporting:
                        dist = []
                if not dist and hint:
                    dist = await self._lookup_distribution(
                        offset, cohort_required=True, state_hint=hint, use_mood=False,
                    )
                    if sum(c for _, c, _ in dist) < self.min_supporting:
                        dist = []
                if not dist and hint is None:
                    # No state hint — try cohort-only, then sop-only.
                    dist = await self._lookup_distribution(offset, cohort_required=True, state_hint=None, use_mood=False)
                    if not dist:
                        dist = await self._lookup_distribution(offset, cohort_required=False, state_hint=None, use_mood=False)
                # Recall gate on raw count; probability on success-weighted sum.
                count_total = sum(c for _, c, _ in dist)
                if count_total < self.min_supporting:
                    continue
                weight_total = sum(w for _, _, w in dist) or 1.0
                for action, _count, weight in dist:
                    prob = weight / weight_total
                    out.append(TrajectoryPrediction(
                        action=action, offset=offset, probability=prob, source=self.name,
                        predicted_user_state=hint if hint else None,
                    ))
                offset_emitted = True
            # If every hinted state was too sparse, fall back to a state-blind query once.
            if not offset_emitted and any(h is not None for h in hints):
                dist = await self._lookup_distribution(offset, cohort_required=True, state_hint=None, use_mood=False)
                if not dist:
                    dist = await self._lookup_distribution(offset, cohort_required=False, state_hint=None, use_mood=False)
                count_total = sum(c for _, c, _ in dist)
                if count_total >= self.min_supporting:
                    weight_total = sum(w for _, _, w in dist) or 1.0
                    for action, _count, weight in dist:
                        prob = weight / weight_total
                        out.append(TrajectoryPrediction(
                            action=action, offset=offset, probability=prob, source=self.name,
                            predicted_user_state=None,
                        ))
        out.sort(key=lambda p: (p.offset, -p.probability))
        return out

    async def _lookup_distribution(
        self,
        offset: int,
        *,
        cohort_required: bool,
        state_hint: str | None = None,
        use_mood: bool = False,
    ) -> list[tuple[str, int, float]]:
        """Returns [(future_action, freq)] for sessions where turn N had
        action=chosen_action, looking at the action at turn N+offset within the same session.

        When state_hint is given, restricts to precedents whose user was in that state at
        the future turn (`next_p.immediate_state`). This is the joint-distribution lookup
        that lets empirical match MCTS's state-conditioning power.

        When use_mood is True AND self.mood is set, also restricts to precedents whose
        mood matches the current turn's classified mood. This is the Phase-2 mood
        conditioning — sharpest filter, fired first in the fallback chain."""
        params: dict[str, object] = {
            "sop_ref": self.sop_ref,
            "chosen_action": self.chosen_action,
            "offset": offset,
        }
        cohort_clause = ""
        if cohort_required and self.cohort:
            cohort_clause = "AND p.cohort = :cohort"
            params["cohort"] = self.cohort
        state_clause = ""
        if state_hint:
            state_clause = "AND next_p.immediate_state = :state_hint"
            params["state_hint"] = state_hint
        mood_clause = ""
        if use_mood and self.mood:
            mood_clause = "AND p.mood = :mood"
            params["mood"] = self.mood
        # Success-weighting (2026-06-07): back-prop writes terminal_reward to every trace
        # on session end (success=1.0, abandoned=0.25, failure=0.0). The empirical
        # predictor used to count raw frequency (COUNT(*)), treating successful and failed
        # action choices equally — so it never actually learned from the back-propagated
        # outcome. We now also SUM the future action's terminal_reward as `wsum`. The
        # min_supporting recall gate still uses the raw count (so sparse-but-real branches
        # aren't dropped just because they were low-reward), but the emitted probability is
        # weight-normalized: actions that appeared in successful sessions dominate. NULL
        # reward (current in-progress session) gets a neutral default so it neither
        # dominates nor vanishes.
        params["neutral_reward"] = 0.3
        sql = text(f"""
            SELECT next_p.action,
                   COUNT(*) AS freq,
                   SUM(COALESCE(next_p.terminal_reward, :neutral_reward)) AS wsum
            FROM precedent_traces p
            JOIN turns t                ON t.id = p.turn_id
            JOIN turns next_t           ON next_t.experiment_id = t.experiment_id
                                       AND next_t.turn_index = t.turn_index + :offset
            JOIN precedent_traces next_p ON next_p.turn_id = next_t.id
            WHERE p.sop_ref = :sop_ref
              AND p.action   = :chosen_action
              {cohort_clause}
              {mood_clause}
              {state_clause}
            GROUP BY next_p.action
            ORDER BY wsum DESC
        """)
        res = await self.db.execute(sql, params)
        return [(row[0], int(row[1]), float(row[2] or 0.0)) for row in res.all()]


# ----------------------------- Union of MCTS + Empirical ------------------------------


class UnionTrajectoryPredictor(TrajectoryPredictor):
    """Run two predictors in parallel and merge their (action, offset) predictions.

    Merge rule per (offset, action):
      - emitted by both predictors  → source="both", probability=1-(1-p_mcts)(1-p_emp)
        (independent-sources union — monotonic in either input, bounded in [0,1])
      - emitted by only one         → source inherits from that predictor, probability unchanged

    Result probabilities are NOT renormalized per offset — that's intentional. Two predictors
    agreeing on an (action, offset) is a stronger prefetch signal than either alone, and we
    want the downstream confidence (probability × decay) to reflect that. Plan items still
    pass through `min_confidence` filtering and per-session outstanding caps, so worst case
    is more queue churn, not unbounded fetcher load.
    """

    name = "union"

    def __init__(self, mcts: MctsTrajectoryPredictor, empirical: EmpiricalTrajectoryPredictor) -> None:
        self.mcts = mcts
        self.empirical = empirical

    # Top-K = how many state branches per offset to feed into the empirical predictor.
    TOP_K_STATES = 3

    async def predict(self, *, max_offset: int = 3) -> list[TrajectoryPrediction]:
        # Two-phase: MCTS first (cheap, in-memory), then empirical conditioned on the
        # per-offset TOP-K state distribution that MCTS predicts. This is the hedge against
        # the modal-aggregation failure mode where minority-but-real state branches get
        # discarded (see notes/2026-05-23-stable-vs-transition-state-prediction-asymmetry.md).
        mcts_preds = await self.mcts.predict(max_offset=max_offset)

        # Build per-offset state distribution by accumulating MCTS predictions' state dists
        # weighted by the action's own probability. Result: P(state at offset N) marginalized
        # over actions, sourced from rollouts. Then take top-K per offset.
        per_offset_state_weight: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))  # type: ignore[arg-type]
        for p in mcts_preds:
            if not p.predicted_user_state_dist:
                continue
            for s, share in p.predicted_user_state_dist.items():
                per_offset_state_weight[p.offset][s] += share * p.probability
        state_hints_topk: dict[int, list[str]] = {}
        for offset, d in per_offset_state_weight.items():
            ordered = sorted(d.items(), key=lambda x: -x[1])[: self.TOP_K_STATES]
            state_hints_topk[offset] = [s for s, _ in ordered]

        emp_preds = await self.empirical.predict(max_offset=max_offset, state_hints_topk=state_hints_topk)

        # Bucket by (offset, action). Empirical may produce multiple predictions for the
        # same (offset, action) when state_hints_topk supplies multiple branches — we keep
        # max probability across hints, and inherit the MCTS-side state distribution
        # whole-cloth (the per-state branching info MCTS already published).
        @dataclass
        class _Bucket:
            p_mcts: float = 0.0
            p_emp: float = 0.0
            state_mcts: str | None = None
            state_emp: str | None = None
            state_dist_mcts: dict[str, float] = field(default_factory=dict)
            # Q6b: carry the MCTS-side predicted user text through Union. Empirical never
            # emits text (it's not in the precedent_traces schema), so MCTS owns this.
            text_mcts: str | None = None

        by_key: dict[tuple[int, str], _Bucket] = defaultdict(_Bucket)
        for p in mcts_preds:
            b = by_key[(p.offset, p.action)]
            if p.probability > b.p_mcts:
                b.p_mcts = p.probability
                b.state_mcts = p.predicted_user_state
                b.state_dist_mcts = dict(p.predicted_user_state_dist or {})
                b.text_mcts = p.predicted_user_text
        for p in emp_preds:
            b = by_key[(p.offset, p.action)]
            if p.probability > b.p_emp:
                b.p_emp = p.probability
                b.state_emp = p.predicted_user_state

        out: list[TrajectoryPrediction] = []
        for (offset, action), b in by_key.items():
            if b.p_mcts > 0 and b.p_emp > 0:
                prob = 1.0 - (1.0 - b.p_mcts) * (1.0 - b.p_emp)
                source = "both"
                state = b.state_mcts or b.state_emp
            elif b.p_mcts > 0:
                prob = b.p_mcts
                source = "mcts"
                state = b.state_mcts
            else:
                prob = b.p_emp
                source = "empirical"
                state = b.state_emp
            out.append(TrajectoryPrediction(
                action=action, offset=offset, probability=prob, source=source,
                predicted_user_state=state,
                # Carry MCTS's top-K state distribution through Union. Empirical's hints
                # came FROM this distribution, so it's the canonical source.
                predicted_user_state_dist=b.state_dist_mcts,
                # Q6b: the rollout-derived text seed (only ever from MCTS-side).
                predicted_user_text=b.text_mcts,
            ))
        out.sort(key=lambda p: (p.offset, -p.probability))
        return out


# ----------------------------- Helper for plan-building ------------------------------


def build_prefetch_plan_from_predictions(
    predictions: list[TrajectoryPrediction],
    *,
    task,
    decay_lambda: float = 0.3,
    cohort: str = "",
    mood: str = "",
    user_text_override: str | None = None,
) -> list:
    """Turns a flat list of TrajectoryPredictions into the (dependency, offset, action,
    confidence) plan items that DataPrefetchManager.schedule consumes.

    Confidence = probability × exp(-decay · offset) (time-decay on top of predictor's
    own probability so far-future predictions get progressively less budget).

    Q6b: when a DataDependency declares a `query_template`, this function renders it
    using the prediction's `(user_text, state, action)` plus the runtime-classified
    `(cohort, mood)` passed by the caller. The rendered query lands on PrefetchPlanItem
    and is forwarded to the fetcher at schedule time. Template uses Python str.format
    placeholders. When no user_text is available (value mode, sparse rollouts), a stub
    is synthesised from (cohort, mood, action, state) so the template still renders.
    """
    from .data_prefetch import PrefetchPlanItem
    deps_by_action: dict[str, list[str]] = {a.name: list(a.data_dependencies or []) for a in task.agent_actions}
    if not any(deps_by_action.values()):
        return []
    # Index DataDependency by name so we can pull query_template per dep.
    dep_by_name = {d.name: d for d in task.data_dependencies}
    scores: dict[tuple[str, int, str], float] = defaultdict(float)
    sources: dict[tuple[str, int, str], set[str]] = defaultdict(set)
    states: dict[tuple[str, int, str], str | None] = {}            # one published hint per plan key
    state_dists: dict[tuple[str, int, str], dict[str, float]] = {} # top-K distribution per plan key
    texts: dict[tuple[str, int, str], str | None] = {}             # Q6b: best user_text per plan key
    for pred in predictions:
        deps = deps_by_action.get(pred.action, [])
        if not deps:
            continue
        score = pred.probability * math.exp(-decay_lambda * pred.offset)
        for dep_name in deps:
            key = (dep_name, pred.offset, pred.action)
            scores[key] += score
            sources[key].add(pred.source or "unknown")
            if states.get(key) is None and pred.predicted_user_state:
                states[key] = pred.predicted_user_state
            if pred.predicted_user_state_dist and len(pred.predicted_user_state_dist) > len(state_dists.get(key, {})):
                state_dists[key] = dict(pred.predicted_user_state_dist)
            if texts.get(key) is None and pred.predicted_user_text:
                texts[key] = pred.predicted_user_text
    items: list[PrefetchPlanItem] = []
    for key, score in scores.items():
        dep_name, offset, action_name = key
        src_set = sources[key]
        if "both" in src_set or ("mcts" in src_set and "empirical" in src_set):
            source = "both"
        elif "mcts" in src_set:
            source = "mcts"
        elif "empirical" in src_set:
            source = "empirical"
        else:
            source = "unknown"
        # Q6b: render the query from the dep's template, if present. Template absence
        # ⇒ legacy action-keyed fetch (today's behaviour).
        rendered_query: str | None = None
        dep_obj = dep_by_name.get(dep_name)
        tmpl = getattr(dep_obj, "query_template", None) if dep_obj is not None else None
        if tmpl:
            state_lbl = states.get(key) or ""
            # Priority for the generative {user_text} slot:
            #   1. user_text_override — a cheap-LLM predicted next utterance (the recommended
            #      source; replaces MCTS rollout text at ~85x lower cost).
            #   2. texts.get(key) — MCTS rollout-derived text (legacy Q6b path).
            #   3. a structured stub from (cohort, mood, state, action) — sufficient on
            #      coarse corpora, weak on fine-grained ones.
            user_text = user_text_override or texts.get(key)
            if not user_text:
                user_text = (f"a {mood or 'neutral'} customer in cohort {cohort or 'unknown'} "
                             f"expected to be in state {state_lbl or 'unknown'}, "
                             f"about to receive agent action {action_name}")
            try:
                rendered_query = tmpl.format(
                    user_text=user_text, cohort=cohort or "", mood=mood or "",
                    state=state_lbl, action=action_name,
                )
            except (KeyError, IndexError):
                # Template had an unknown placeholder; fall back to action-keyed fetch.
                rendered_query = None
        items.append(PrefetchPlanItem(
            dependency_name=dep_name,
            action_name=action_name,
            confidence=round(score, 4),
            predicted_turn_offset=offset,
            predictor_source=source,
            predicted_user_state=states.get(key),
            predicted_user_state_dist=state_dists.get(key, {}),
            rendered_query=rendered_query,
        ))
    items.sort(key=lambda i: (-i.confidence, i.predicted_turn_offset))
    return items
