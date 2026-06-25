"""SOP graph operations.

Filters allowed next agent actions given conversation history.

Semantics for an edge (src -> dst, direction):
  "forward"  : dst is allowed only AFTER src has occurred
  "backward" : src is allowed only AFTER dst has occurred (equivalent to dst -> src forward)
  "both"     : no ordering constraint (informational link)

An action is "allowed" if every forward-prereq for it has been visited.

Hierarchical helpers operate on Strategy objects when `task.strategies` is non-empty.
A strategy is "allowed" iff at least one member action is allowed.
"""

from __future__ import annotations
from ..schemas import TaskDefinition, Strategy


class SOPGraph:
    def __init__(self, task: TaskDefinition):
        self.task = task
        self.action_names = {a.name for a in task.agent_actions}
        self.state_names = {s.name for s in task.user_states}
        # Two kinds of prerequisite per action (2026-06-08 fix):
        #   action_prereqs: action->action forward edges — HARD ordering constraints
        #                   (do X before Y). All must be met (AND).
        #   state_prereqs:  state->action forward edges — the action is TRIGGERED by the
        #                   user reaching that state (e.g. PriceConcern -> HandlePriceObjection,
        #                   HardDecline -> ClosePolite). These only gate actions that have
        #                   NO action prereqs (otherwise the action's place in the flow is
        #                   already fixed by ordering, and the state is just an enabling hint).
        #
        # Why split them: treating every state node as an AND-prerequisite over-constrains.
        # StateReason had prereqs {VerifyIdentity (action), IsThemselves (state)}. When the
        # user was classified ReportingChange rather than IsThemselves, StateReason stayed
        # permanently blocked, which blocked everything downstream and collapsed the allowed
        # set to {Greeting, VerifyIdentity} — forcing the proposer fallback to loop on
        # Greeting. Conversely, dropping state gates entirely lets terminal actions like
        # ClosePolite fire at turn 1. The split fixes both.
        self.prereqs: dict[str, set[str]] = {a: set() for a in self.action_names}        # kept for back-compat (action prereqs)
        self.action_prereqs: dict[str, set[str]] = {a: set() for a in self.action_names}
        self.state_prereqs: dict[str, set[str]] = {a: set() for a in self.action_names}
        for e in task.sop.edges:
            if e.direction == "forward" and e.dst in self.action_names:
                if e.src in self.action_names:
                    self.action_prereqs[e.dst].add(e.src)
                elif e.src in self.state_names:
                    self.state_prereqs[e.dst].add(e.src)
            elif e.direction == "backward" and e.src in self.action_names:
                if e.dst in self.action_names:
                    self.action_prereqs[e.src].add(e.dst)
            # "both" => no ordering constraint
        # Mirror action_prereqs into the legacy `prereqs` field (some callers read it).
        for a in self.action_names:
            self.prereqs[a] = set(self.action_prereqs[a])

    def allowed_actions(self, visited: set[str]) -> list[str]:
        """Actions legal in the current visited state.

        Gating rules (see __init__ for rationale):
          - All action_prereqs must be in `visited` (hard ordering, AND).
          - If an action has action_prereqs, it's allowed once those are met — its
            state_prereqs are treated as enabling hints, not hard gates.
          - If an action has NO action_prereqs but HAS state_prereqs, it's a
            state-triggered action: allowed only once at least one trigger state is in
            `visited` (so terminal/branch actions like ClosePolite don't fire prematurely).
          - If an action has no prereqs at all (e.g. Greeting), it's always allowed.

        If nothing qualifies, fall back to the full catalog so the agent isn't stuck.
        """
        out: list[str] = []
        for a in self.action_names:
            if not self.action_prereqs[a].issubset(visited):
                continue
            if self.action_prereqs[a]:
                out.append(a)                       # ordering satisfied → legal
            elif not self.state_prereqs[a]:
                out.append(a)                       # no prereqs at all → always legal
            elif self.state_prereqs[a] & visited:
                out.append(a)                       # state-triggered and trigger occurred
            # else: state-triggered, no trigger yet → not legal
        if not out:
            return sorted(self.action_names)
        return sorted(out)

    def visited_from_history(self, history: list[dict[str, str]], state_log: list[str]) -> set[str]:
        """Collect visited node names from action tags in assistant messages and the user-state log."""
        visited: set[str] = set()
        for h in history:
            tag = h.get("action")
            if tag and tag in self.action_names:
                visited.add(tag)
        for s in state_log:
            if s in self.state_names:
                visited.add(s)
        return visited

    # -------- Hierarchical (strategy-level) helpers --------

    def get_strategies(self) -> list[Strategy]:
        """Return task.strategies if non-empty, else auto-derive one strategy per action.

        The fallback ensures hierarchical mode degrades gracefully on SOPs that haven't
        yet been annotated with explicit strategies — each agent_action becomes its own
        single-member strategy, so 'strategy mode' becomes equivalent to action mode.
        """
        if self.task.strategies:
            return list(self.task.strategies)
        return [
            Strategy(name=a.name, description=a.description, member_actions=[a.name])
            for a in self.task.agent_actions
        ]

    def allowed_strategies(self, visited: set[str]) -> list[Strategy]:
        """A strategy is allowed iff at least one of its member actions is SOP-allowed
        in the current visited state. Useful for top-level MCTS expansion."""
        allowed_actions_set = set(self.allowed_actions(visited))
        out: list[Strategy] = []
        for s in self.get_strategies():
            members_allowed = [a for a in s.member_actions if a in allowed_actions_set]
            if members_allowed:
                out.append(s)
        # Fallback: if no strategy has an allowed member (shouldn't happen since
        # allowed_actions itself falls back to the full catalog), return all strategies.
        if not out:
            out = self.get_strategies()
        return out

    def instantiate_strategy(
        self,
        strategy_name: str,
        visited: set[str],
    ) -> str:
        """Pick a concrete action to execute for the given strategy. Deterministic:
        the first member action that is SOP-allowed. If no member is allowed, fall back
        to the first allowed action overall (safety)."""
        allowed_now = set(self.allowed_actions(visited))
        for s in self.get_strategies():
            if s.name == strategy_name:
                for a in s.member_actions:
                    if a in allowed_now:
                        return a
                # All members consumed already; fall through.
                break
        # Fallback: any allowed action
        if allowed_now:
            return next(iter(sorted(allowed_now)))
        return ""
