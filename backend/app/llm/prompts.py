"""All LLM prompts in one place — easier to tune.

The planner is a *supervisor*: it constrains a base LLM via:
  - A discrete vocabulary of agent actions (no free-form actions allowed)
  - A discrete vocabulary of user states
  - A directed SOP graph that filters which next actions are legal
"""

from __future__ import annotations
import json
from ..schemas import TaskDefinition


# ---------------------------------------------------------------------------
# FRAMEWORK_PREAMBLE
#
# Prepended verbatim to every planner system prompt by app/llm/client.py.
# Kept identical across all calls so the OpenAI API auto-caches the prefix
# (cache hits require >=1024 stable tokens at the start of the prompt).
# Re-reading prompts.py contents on disk is what defines "stable" here —
# avoid edits in long-running benchmarks if you want consistent cache hits.
# ---------------------------------------------------------------------------

FRAMEWORK_PREAMBLE = """[PCA PLANNER FRAMEWORK — STABLE CONTEXT]

You are part of an experimental research system that implements the PCA
("Planning-based Conversational Agents") framework from Hu et al., 2024
(arXiv:2407.03884). The framework's goal is to use a hosted LLM as a
*conversational supervisor* that always picks the next agent action from
a discrete, business-defined vocabulary and respects a directed Standard
Operating Procedure (SOP) graph that encodes ordering constraints between
actions and observed user states.

The framework decomposes one turn of dialogue into several specialized
subtasks, each of which is handled by a distinct prompt. The subtasks are:

  1. User-state prediction
     Given the conversation history so far, choose the single best name
     from the task's `user_states` vocabulary that describes the user's
     current state. Output JSON.

  2. Action selection
     Given the predicted user state, the conversation history, and the
     subset of `agent_actions` that the SOP currently permits (the
     "allowed" set computed by the planner), choose the next agent action.
     Always pick from the allowed set. Output JSON.

  3. MCTS expansion (PCA-M only)
     Propose the top-K candidate next actions from the allowed set, with
     a one-sentence rationale per candidate. The MCTS planner will roll
     each candidate forward to estimate its Q-value.

  4. MCTS rollout step (PCA-M only)
     Pick the single best next agent action under the current SOP-allowed
     set during a simulation. Same shape as action selection but lower
     temperature and only one candidate.

  5. Simulated user
     Role-play the user, conditioned on the user profile. Reply in 1-2
     sentences and label the user state that best describes you AFTER
     sending the reply. Used inside MCTS rollouts and in auto-sim mode.

  6. Rationality reward
     Score on a 0.0-1.0 scale how rationally the proposed action sequence
     advances the agent's goal. Used by MCTS as the reward signal,
     combined with a +0.5 bonus when a success-marker user_state is hit.

  7. Response generation
     Given the chosen action, history, and knowledge, produce a natural
     1-3 sentence utterance executing that action. Plain text only.

  8. SOP builder (Configuration tab only)
     A separate flow that iteratively constructs the SOP from a chat with
     the human researcher. Returns a JSON patch to merge into the SOP.

SHARED CONVENTIONS

* JSON outputs must be valid JSON and parsed by the calling code. Never
  wrap them in markdown code fences. Never add commentary outside the
  JSON object when JSON is requested.

* Plain-text outputs (the response generator, the user simulator's reply
  field) should never contain leading or trailing role labels like
  "Assistant:" or "User:". Just the utterance.

* When asked to choose a name from a vocabulary (user_state, action,
  edge direction, etc.), output exactly one of the listed names with
  matching case. The runtime snaps unknown labels back to the vocabulary
  but unstable picks degrade downstream reasoning.

* When asked for a rationale, keep it to one sentence focused on the
  immediate decision. Avoid restating the entire goal or history.

SOP GRAPH SEMANTICS

The SOP is a directed graph whose vertices are agent action names and
user state names from the task's vocabularies, and whose edges encode
ordering constraints:

  forward   (src -> dst): dst is only allowed AFTER src has occurred
                          (src is a prerequisite for dst)
  backward  (src -> dst): src is only allowed AFTER dst has occurred
                          (equivalent to dst forward src)
  both                  : informational link, no ordering constraint

The planner maintains a `visited` set built from the agent actions
already executed and the user states already observed. An action is
"allowed" only when every forward prerequisite for it is in `visited`.
If no actions qualify the planner falls back to the full catalog so the
agent is never frozen.

USER PROFILE AND GOAL

Each turn includes the task definition: the user profile, the agent's
role and goal, success and failure markers (specific user_state names
that terminate the conversation as success or failure), and any task
knowledge. The user simulator should stay in character based on the
user profile rather than playing along with the agent. Realistic user
behavior produces useful Q-values during search.

OUTPUT DISCIPLINE

These calls are issued in tight loops during MCTS search. Verbose
outputs waste tokens, increase latency, and reduce the planner's
effective search budget. Keep responses minimal:

  - User-state prediction: vocabulary name + one sentence
  - Action selection / MCTS step: vocabulary name + one sentence
  - MCTS expansion: list of up to K objects, each one sentence
  - Simulated user: one or two sentences, then state name
  - Rationality: number + one sentence
  - Response generator: one to three sentences

THE PER-CALL INSTRUCTIONS BELOW SUPERSEDE ANY CONFLICT WITH THIS PREAMBLE.

[END FRAMEWORK PREAMBLE]"""


# ---------- SOP builder (Configuration tab chat) ----------

SOP_BUILDER_SYSTEM = """You are an expert dialogue-design assistant helping a user
construct a Standard Operating Procedure (SOP) for a conversational agent, following
the PCA framework (Hu et al., 2024).

An SOP captures, for one task:
  - user_profile      : who the human interlocutor typically is
  - conversation_profile: agent_role, goal, success_markers, failure_markers, knowledge
  - agent_actions     : a discrete, named vocabulary of dialogue actions the agent may take
                        (e.g., "VerifyIdentity", "PitchActivation", "HandleObjection")
  - user_states       : a discrete vocabulary of user states the planner predicts each turn
                        (e.g., "IsThemselves", "NotInterested", "Confused")
  - sop edges         : directed constraints between nodes (action or state names) using:
                        "forward"  -> src must occur before dst
                        "backward" -> dst must occur before src
                        "both"     -> either order acceptable

Your output EACH turn must be a JSON object with exactly these keys:

  "assistant_message": short message shown to the user (<=2 sentences).
  "sop_patch":         a partial TaskDefinition holding ONLY what should be added or
                       updated this turn (see schema below). Omit any field you do not
                       want to touch.
  "is_complete":       boolean. true only when every required field is filled and the
                       user has confirmed.

SOP_PATCH SCHEMA (use exactly this shape):

  {
    "name":        "<string, optional>",
    "description": "<string, optional>",

    "user_profile": {
      "name": "<string>", "description": "<string>", "demographics": {<k:v>}
    },
    "conversation_profile": {
      "agent_role": "<string>", "goal": "<string>",
      "success_markers": ["<state-name>", ...],
      "failure_markers": ["<state-name>", ...],
      "knowledge": "<string>"
    },

    "agent_actions": [
      {"name": "<PascalCase>", "description": "<one sentence>"}
    ],
    "user_states": [
      {"name": "<PascalCase>", "description": "<one sentence>"}
    ],

    "sop": {
      "edges": [
        {"src": "<node>", "dst": "<node>", "direction": "forward|backward|both", "note": "<optional>"}
      ]
    }
  }

MERGE SEMANTICS (the runtime applies these, you do not need to manage them yourself):

  - agent_actions and user_states are merged BY NAME. To add a new action, include
    just that one new {"name": "...", "description": "..."} object — existing entries
    keep their values. To update an action's description, include the same name with
    the new description.
  - sop.edges are merged by (src, dst, direction). Add edges incrementally.
  - sop.nodes is auto-derived from agent_actions + user_states + edges; never emit it.
  - List items MUST be objects with a 'name' field. Do NOT emit list items as bare strings.

YOUR BEHAVIOR EACH TURN:
  1. Ask ONE focused question or confirm what you just inferred.
  2. Emit the smallest patch that captures the new information.
  3. Do not invent edges the user did not imply.
  4. Prefer small additions per turn so the user can watch the graph grow.

Return ONLY valid JSON, no markdown fences."""


def sop_builder_user_prompt(current: TaskDefinition, history: list[dict[str, str]]) -> str:
    return (
        "CURRENT SOP STATE (JSON):\n"
        + json.dumps(current.model_dump(), indent=2)
        + "\n\nCONVERSATION HISTORY:\n"
        + json.dumps(history, indent=2)
        + "\n\nReturn ONLY a JSON object with keys: assistant_message, sop_patch, is_complete."
    )


# ---------- State prediction (per turn) ----------

STATE_PREDICTION_SYSTEM = """You predict the current user state from a fixed vocabulary,
given the conversation so far. Output JSON: {"user_state": "<one of the listed names>", "rationale": "<one sentence>"}."""


def state_prediction_user_prompt(task: TaskDefinition, history: list[dict[str, str]]) -> str:
    return (
        f"TASK: {task.conversation_profile.goal}\n"
        f"USER PROFILE: {task.user_profile.description}\n"
        f"USER STATE VOCABULARY:\n"
        + "\n".join(f"  - {s.name}: {s.description}" for s in task.user_states)
        + "\n\nHISTORY:\n"
        + json.dumps(history, indent=2)
        + "\n\nReturn JSON only."
    )


# ---------- Baseline action selection (CoT + SOP) ----------

BASELINE_ACTION_SYSTEM = """You are a conversational planner that selects ONE next agent action
from a fixed vocabulary, respecting SOP constraints. Output JSON:
{"action": "<name>", "rationale": "<one sentence>"}."""


def baseline_action_user_prompt(
    task: TaskDefinition,
    history: list[dict[str, str]],
    predicted_user_state: str,
    allowed_actions: list[str],
) -> str:
    return (
        f"AGENT ROLE: {task.conversation_profile.agent_role}\n"
        f"GOAL: {task.conversation_profile.goal}\n"
        f"SUCCESS MARKERS: {task.conversation_profile.success_markers}\n"
        f"KNOWLEDGE: {task.conversation_profile.knowledge}\n\n"
        f"PREDICTED USER STATE: {predicted_user_state}\n\n"
        f"ALLOWED NEXT ACTIONS (SOP-filtered):\n"
        + "\n".join(f"  - {a}" for a in allowed_actions)
        + "\n\nFULL ACTION CATALOG:\n"
        + "\n".join(f"  - {a.name}: {a.description}" for a in task.agent_actions)
        + "\n\nHISTORY:\n"
        + json.dumps(history, indent=2)
        + "\n\nReturn JSON only."
    )


# ---------- MCTS: action proposal (expansion phase) ----------

MCTS_PROPOSE_SYSTEM = """You propose the top-K candidate next actions for a conversational agent
under SOP constraints. Output JSON: {"candidates": [{"action": "<name>", "rationale": "<one sentence>"}]}.
Choose only from the allowed list."""


def mcts_propose_user_prompt(
    task: TaskDefinition,
    history: list[dict[str, str]],
    predicted_user_state: str,
    allowed_actions: list[str],
    k: int,
) -> str:
    return (
        f"AGENT ROLE: {task.conversation_profile.agent_role}\n"
        f"GOAL: {task.conversation_profile.goal}\n"
        f"PREDICTED USER STATE: {predicted_user_state}\n"
        f"ALLOWED ACTIONS: {allowed_actions}\n\n"
        f"HISTORY:\n{json.dumps(history, indent=2)}\n\n"
        f"Return up to {k} candidates ordered best-first. JSON only."
    )


# ---------- Combined state-prediction + action selection (one call per turn) ----------

# ---------- Strategy-level cohort + state + propose (hierarchical MCTS) ----------

COHORT_STATE_PROPOSE_STRATEGY_SYSTEM = """You classify (a) the user cohort, (b) the user
state, and (c) propose up to K candidate next STRATEGIES — coarse-grained dialogue phases
that contain multiple concrete agent actions. The MCTS planner uses your strategies as
search-tree nodes; the runtime instantiates each strategy to a concrete action via SOP
constraints.

Output JSON only:
{
  "cohort":          "<cohort label>",
  "user_state":      "<one user-state name>",
  "state_rationale": "<one sentence>",
  "candidates": [
    {"strategy": "<allowed strategy name>", "rationale": "<one sentence>"}
  ]
}
Up to K candidates ordered best-first. Choose strategies only from the ALLOWED list."""


def cohort_state_propose_strategy_user_prompt(
    task: TaskDefinition,
    history: list[dict[str, str]],
    allowed_strategies: list[str],
    k: int,
    *,
    precedents_block: str = "",
) -> str:
    cohort_lines = (
        "\n".join(f"  - {c.name}: {c.description}" for c in task.cohorts)
        if task.cohorts else "  (no vocabulary — emit a concise free-form label)"
    )
    strat_lines: list[str] = []
    for s in task.strategies:
        members = ", ".join(s.member_actions) or "(no members)"
        strat_lines.append(f"  - {s.name} (covers: {members}): {s.description}")
    return (
        f"AGENT ROLE: {task.conversation_profile.agent_role}\n"
        f"GOAL: {task.conversation_profile.goal}\n"
        f"USER PROFILE: {task.user_profile.description}\n\n"
        "COHORT VOCABULARY:\n" + cohort_lines + "\n\n"
        "USER STATE VOCABULARY:\n"
        + "\n".join(f"  - {s.name}: {s.description}" for s in task.user_states)
        + f"\n\nALLOWED NEXT STRATEGIES (SOP-filtered): {allowed_strategies}\n\n"
        "STRATEGY CATALOG:\n"
        + ("\n".join(strat_lines) if strat_lines else "  (empty — using auto-derived single-action strategies)")
        + (("\n\nRELEVANT PAST PRECEDENTS:\n" + precedents_block) if precedents_block else "")
        + f"\n\nHISTORY:\n{json.dumps(history, indent=2)}\n\n"
        f"Return JSON only. Up to {k} strategy candidates."
    )


MCTS_ROLLOUT_STRATEGY_SYSTEM = """Pick ONE next strategy for the agent from the allowed list.
Output JSON: {"strategy": "<name>", "rationale": "<one sentence>"}."""


def mcts_rollout_strategy_user_prompt(
    task: TaskDefinition,
    history: list[dict[str, str]],
    predicted_user_state: str,
    allowed_strategies: list[str],
) -> str:
    return (
        f"AGENT ROLE: {task.conversation_profile.agent_role}\n"
        f"GOAL: {task.conversation_profile.goal}\n"
        f"PREDICTED USER STATE: {predicted_user_state}\n"
        f"ALLOWED STRATEGIES: {allowed_strategies}\n\n"
        f"HISTORY:\n{json.dumps(history, indent=2)}\n\n"
        "Pick the strategy that best advances the goal next. JSON only."
    )


# ---------- Combined cohort + state + propose (one call per turn, MCTS root) ----------

COHORT_STATE_PROPOSE_SYSTEM = """You classify (a) the current user cohort, (b) the current
user state, (c) the user's mood within that cohort (when the cohort has a mood vocabulary),
and (d) propose up to K candidate next agent actions, all from the supplied vocabularies
and the SOP-allowed set.

Cohort describes WHO the user is in this conversation (e.g. "PriceSensitive+Returning"),
drawn from the COHORT VOCABULARY. Be deterministic — same conversation state should yield
the same cohort label.

Mood describes HOW the user is feeling within that cohort (e.g. "fee_focused" vs
"comparison_shopping" within the PriceSensitive cohort). Mood is per-cohort: pick from
that cohort's MOODS list. If the chosen cohort has no moods declared, emit "" (empty).

If precedent traces are provided you may use them as evidence but you must still select
ONE cohort, ONE user_state, ONE mood (or ""), and only actions from the ALLOWED list.

When proposing candidate actions, drive the conversation FORWARD. Read the ACTIONS
ALREADY TAKEN list: an action that was just used should not be proposed again unless the
user's latest message specifically calls for it. Greeting and other one-shot openers are
only appropriate at the very start of the call; after identity and reason-for-call are
established, prefer actions that advance the SOP toward its goal.

Output JSON only:
{
  "cohort":          "<cohort label>",
  "user_state":      "<one user-state name>",
  "mood":            "<one mood name from the chosen cohort's MOODS, or ''>",
  "state_rationale": "<one sentence>",
  "candidates": [
    {"action": "<allowed action name>", "rationale": "<one sentence>"}
  ]
}
Up to K candidates ordered best-first."""


def cohort_state_propose_user_prompt(
    task: TaskDefinition,
    history: list[dict[str, str]],
    allowed_actions: list[str],
    k: int,
    *,
    precedents_block: str = "",
) -> str:
    # Render cohorts with their per-cohort mood vocabularies inline. The classifier sees
    # the structural constraint that mood must belong to the chosen cohort's list.
    def _render_cohort(c) -> str:
        line = f"  - {c.name}: {c.description}"
        moods = getattr(c, "moods", None) or []
        if moods:
            mood_list = ", ".join(m.name for m in moods)
            line += f"\n    Moods: {mood_list}"
        return line
    cohort_lines = (
        "\n".join(_render_cohort(c) for c in task.cohorts)
        if task.cohorts else "  (no vocabulary — emit a concise free-form label)"
    )
    return (
        f"AGENT ROLE: {task.conversation_profile.agent_role}\n"
        f"GOAL: {task.conversation_profile.goal}\n"
        f"USER PROFILE: {task.user_profile.description}\n\n"
        "COHORT VOCABULARY (with per-cohort moods, when present):\n" + cohort_lines + "\n\n"
        "USER STATE VOCABULARY:\n"
        + "\n".join(f"  - {s.name}: {s.description}" for s in task.user_states)
        + f"\n\nALLOWED NEXT ACTIONS (SOP-filtered): {allowed_actions}\n\n"
        "FULL ACTION CATALOG:\n"
        + "\n".join(f"  - {a.name}: {a.description}" for a in task.agent_actions)
        + (("\n\nRELEVANT PAST PRECEDENTS:\n" + precedents_block) if precedents_block else "")
        + "\n\nACTIONS ALREADY TAKEN this call (oldest→newest): "
        + (" → ".join([h.get("action") for h in history if h.get("action")]) or "(none yet)")
        + "\nProgress the conversation: prefer an allowed action that advances the SOP "
          "toward the goal. Do NOT re-select an action that was just taken unless the "
          "user's latest message specifically requires repeating it (e.g. they asked you "
          "to clarify the same thing). One-shot openers like Greeting belong only at the "
          "start — once identity/reason are established, move forward.\n"
        + f"\n\nHISTORY:\n{json.dumps(history, indent=2)}\n\n"
        f"Return JSON only. Up to {k} candidates."
    )


STATE_AND_PROPOSE_SYSTEM = """You both classify the user's current state AND propose
the top-K candidate next agent actions, all from the supplied vocabularies and SOP-allowed set.

Output JSON only:
{
  "user_state":     "<one name from the user-state vocabulary>",
  "state_rationale":"<one sentence>",
  "candidates": [
    {"action": "<allowed action name>", "rationale": "<one sentence>"},
    ...
  ]
}
Return up to K candidates ordered best-first. Choose actions only from the ALLOWED list."""


def state_and_propose_user_prompt(
    task: TaskDefinition,
    history: list[dict[str, str]],
    allowed_actions: list[str],
    k: int,
) -> str:
    return (
        f"AGENT ROLE: {task.conversation_profile.agent_role}\n"
        f"GOAL: {task.conversation_profile.goal}\n"
        f"USER PROFILE: {task.user_profile.description}\n\n"
        "USER STATE VOCABULARY:\n"
        + "\n".join(f"  - {s.name}: {s.description}" for s in task.user_states)
        + f"\n\nALLOWED NEXT ACTIONS (SOP-filtered): {allowed_actions}\n\n"
        "FULL ACTION CATALOG:\n"
        + "\n".join(f"  - {a.name}: {a.description}" for a in task.agent_actions)
        + f"\n\nHISTORY:\n{json.dumps(history, indent=2)}\n\n"
        f"Return JSON only. Up to {k} candidates."
    )


COHORT_STATE_BASELINE_SYSTEM = """You classify cohort + user state AND select ONE next
agent action from the SOP-allowed vocabulary.

Output JSON only:
{
  "cohort":           "<cohort label>",
  "user_state":       "<one user-state name>",
  "state_rationale":  "<one sentence>",
  "action":           "<one allowed action name>",
  "action_rationale": "<one sentence>"
}"""


def cohort_state_baseline_user_prompt(
    task: TaskDefinition,
    history: list[dict[str, str]],
    allowed_actions: list[str],
    *,
    precedents_block: str = "",
) -> str:
    cohort_lines = (
        "\n".join(f"  - {c.name}: {c.description}" for c in task.cohorts)
        if task.cohorts else "  (no vocabulary — emit a concise free-form label)"
    )
    return (
        f"AGENT ROLE: {task.conversation_profile.agent_role}\n"
        f"GOAL: {task.conversation_profile.goal}\n"
        f"SUCCESS MARKERS: {task.conversation_profile.success_markers}\n"
        f"KNOWLEDGE: {task.conversation_profile.knowledge}\n"
        f"USER PROFILE: {task.user_profile.description}\n\n"
        "COHORT VOCABULARY:\n" + cohort_lines + "\n\n"
        "USER STATE VOCABULARY:\n"
        + "\n".join(f"  - {s.name}: {s.description}" for s in task.user_states)
        + f"\n\nALLOWED NEXT ACTIONS (SOP-filtered): {allowed_actions}\n\n"
        "FULL ACTION CATALOG:\n"
        + "\n".join(f"  - {a.name}: {a.description}" for a in task.agent_actions)
        + (("\n\nRELEVANT PAST PRECEDENTS:\n" + precedents_block) if precedents_block else "")
        + f"\n\nHISTORY:\n{json.dumps(history, indent=2)}\n\n"
        "Return JSON only."
    )


def format_precedents_block(precedents: list) -> str:
    """Render retrieved precedents into a compact text block for prompt injection.

    Each precedent is one line with cohort, action, immediate outcome, terminal outcome,
    similarity. The model can use these as evidence WITHOUT being instructed to copy them.
    """
    if not precedents:
        return ""
    lines = []
    for p in precedents:
        outcome_bits = []
        if p.immediate_state:
            outcome_bits.append(f"→{p.immediate_state}")
        if p.terminal_outcome:
            outcome_bits.append(f"terminal:{p.terminal_outcome}")
        outcome = " ".join(outcome_bits) if outcome_bits else "open"
        snippet = (p.response_text or "").strip().replace("\n", " ")
        if len(snippet) > 160:
            snippet = snippet[:160] + "…"
        lines.append(
            f"- [sim {p.similarity:.2f}] cohort={p.cohort} action={p.action} {outcome}  "
            f'response="{snippet}"'
        )
    return "\n".join(lines)


STATE_AND_BASELINE_SYSTEM = """You classify the user's current state AND select ONE next
agent action, all from the supplied vocabularies and SOP-allowed set.

Output JSON only:
{
  "user_state":      "<one name from the user-state vocabulary>",
  "state_rationale": "<one sentence>",
  "action":          "<one allowed action name>",
  "action_rationale":"<one sentence>"
}
Choose `action` only from the ALLOWED list."""


def state_and_baseline_user_prompt(
    task: TaskDefinition,
    history: list[dict[str, str]],
    allowed_actions: list[str],
) -> str:
    return (
        f"AGENT ROLE: {task.conversation_profile.agent_role}\n"
        f"GOAL: {task.conversation_profile.goal}\n"
        f"SUCCESS MARKERS: {task.conversation_profile.success_markers}\n"
        f"KNOWLEDGE: {task.conversation_profile.knowledge}\n"
        f"USER PROFILE: {task.user_profile.description}\n\n"
        "USER STATE VOCABULARY:\n"
        + "\n".join(f"  - {s.name}: {s.description}" for s in task.user_states)
        + f"\n\nALLOWED NEXT ACTIONS (SOP-filtered): {allowed_actions}\n\n"
        "FULL ACTION CATALOG:\n"
        + "\n".join(f"  - {a.name}: {a.description}" for a in task.agent_actions)
        + f"\n\nHISTORY:\n{json.dumps(history, indent=2)}\n\n"
        "Return JSON only."
    )


# ---------- MCTS: rationality reward ----------

RATIONALITY_SYSTEM = """You rate the logical rationality of an agent's plan in a dialogue,
on a 0.0-1.0 scale. Output JSON: {"score": <float 0..1>, "rationale": "<one sentence>"}."""


def rationality_user_prompt(
    task: TaskDefinition,
    history: list[dict[str, str]],
    planned_actions: list[str],
) -> str:
    return (
        f"GOAL: {task.conversation_profile.goal}\n"
        f"SUCCESS MARKERS: {task.conversation_profile.success_markers}\n\n"
        f"HISTORY:\n{json.dumps(history, indent=2)}\n\n"
        f"PLANNED ACTION SEQUENCE: {planned_actions}\n\n"
        "Score how rationally this sequence advances the goal under realistic user behavior. JSON only."
    )


# ---------- Value-scoring rollout (Fast-MCTD-style sparse rollout) ----------

VALUE_SCORE_SYSTEM = """You estimate the value of a candidate agent action plan WITHOUT
explicitly simulating the user's replies. You're the value model in an MCTS that replaces
expensive per-step simulation with a single scoring call.

Output JSON only:
{
  "sequence_quality":    <float 0..1, how rationally the plan advances the goal under realistic user behavior>,
  "success_probability": <float 0..1, your estimate that this plan reaches a success marker if executed>,
  "rationale":           "<one sentence>"
}

Both scores should be calibrated:
  0.0 = clearly off-task or violates SOP norms
  0.5 = plausible but unfocused / uncertain
  1.0 = clearly the right move that drives toward the goal"""


def value_score_user_prompt(
    task: TaskDefinition,
    history: list[dict[str, str]],
    first_action: str,
    planned_remaining_actions: list[str],
    predicted_user_state: str,
    *,
    precedents_block: str = "",
) -> str:
    actions_map = {a.name: a.description for a in task.agent_actions}
    fa_desc = actions_map.get(first_action, "")
    plan_lines = [f"FIRST ACTION: {first_action} - {fa_desc}"]
    if planned_remaining_actions:
        plan_lines.append("LIKELY NEXT ACTIONS (best guess by the planner):")
        for a in planned_remaining_actions:
            plan_lines.append(f"  - {a}: {actions_map.get(a, '')}")
    return (
        f"AGENT ROLE: {task.conversation_profile.agent_role}\n"
        f"GOAL: {task.conversation_profile.goal}\n"
        f"SUCCESS MARKERS: {task.conversation_profile.success_markers}\n"
        f"FAILURE MARKERS: {task.conversation_profile.failure_markers}\n"
        f"KNOWLEDGE: {task.conversation_profile.knowledge}\n\n"
        f"USER PROFILE: {task.user_profile.description}\n"
        f"PREDICTED CURRENT USER STATE: {predicted_user_state}\n\n"
        + "\n".join(plan_lines)
        + (("\n\nRELEVANT PAST PRECEDENTS:\n" + precedents_block) if precedents_block else "")
        + f"\n\nHISTORY:\n{json.dumps(history, indent=2)}\n\n"
        "Return JSON only."
    )


# ---------- Response generation ----------

RESPONSE_GEN_SYSTEM = """You are the agent. Produce the next utterance executing the chosen action.
Stay in role. Keep it natural and concise (1-3 sentences). Output plain text, no JSON, no labels."""


def response_gen_user_prompt(
    task: TaskDefinition,
    history: list[dict[str, str]],
    chosen_action: str,
    *,
    must_say: list[str] | None = None,
    must_not_say: list[str] | None = None,
    precedents_block: str = "",
    prefetched_context: list[str] | None = None,
) -> str:
    actions_map = {a.name: a.description for a in task.agent_actions}
    must_say = must_say or []
    must_not_say = must_not_say or []
    advice_block = ""
    if must_say or must_not_say or precedents_block or prefetched_context:
        parts: list[str] = []
        if must_say:
            parts.append("MUST INCLUDE (proven phrases — adapt naturally, don't quote verbatim):\n  - " + "\n  - ".join(must_say))
        if must_not_say:
            parts.append("AVOID (negative-lift patterns):\n  - " + "\n  - ".join(must_not_say))
        if precedents_block:
            parts.append("RELEVANT PAST AGENT RESPONSES (style reference, do not copy):\n" + precedents_block)
        if prefetched_context:
            # The supervisor speculatively prefetches data and pre-stages it on the
            # blackboard before knowing what the user will actually say. Items often help
            # but can be off-topic (wrong prediction, conversation moved on, user is just
            # making small talk). The agent must evaluate each item against the user's
            # most recent message and use only what fits. See v3 design discussion in
            # blog/2026-06-04-supervising-the-fast-mouth.html.
            ctx_lines = "\n".join(f"  · {c}" for c in prefetched_context)
            parts.append(
                "PREFETCHED CONTEXT (speculatively pre-staged — may or may not fit this turn):\n"
                "The supervisor pre-fetched these items based on predictions about what the user "
                "might ask. They may help your reply, or they may be off-topic — especially when the "
                "user is wrapping up, making small talk, or asking about something unanticipated.\n"
                "Evaluate each item against what the user just said. Use what is relevant; ignore "
                "what isn't. If nothing fits the current message, reply naturally as if no context "
                "was provided — do NOT force irrelevant data into the response.\n\n"
                + ctx_lines
            )
        advice_block = "\n\n" + "\n\n".join(parts)
    return (
        f"ROLE: {task.conversation_profile.agent_role}\n"
        f"GOAL: {task.conversation_profile.goal}\n"
        f"KNOWLEDGE: {task.conversation_profile.knowledge}\n\n"
        f"CHOSEN ACTION: {chosen_action} - {actions_map.get(chosen_action, '')}\n"
        + advice_block
        + f"\n\nHISTORY:\n{json.dumps(history, indent=2)}\n\n"
        "Write the agent's next utterance now."
    )


# ---------- User simulator (rollouts + auto-chat) ----------

USER_SIM_SYSTEM = """You role-play the user in a dialogue. Stay in character based on the user profile.
Reply naturally in 1-2 sentences. Plain text only, no labels."""


def user_sim_user_prompt(
    task: TaskDefinition,
    history: list[dict[str, str]],
) -> str:
    return (
        f"YOU ARE THE USER. PROFILE:\n{task.user_profile.description}\n"
        f"Demographics: {task.user_profile.demographics}\n\n"
        f"Agent is trying to: {task.conversation_profile.goal}\n"
        f"HISTORY (you are 'user'):\n{json.dumps(history, indent=2)}\n\n"
        "Write your next reply as the user."
    )


# ---------- User-state classifier for simulated rollouts ----------

ROLLOUT_STATE_SYSTEM = """Classify the user's state from the vocabulary given the latest reply.
Output JSON: {"user_state": "<name>"}."""


def rollout_state_user_prompt(task: TaskDefinition, history: list[dict[str, str]]) -> str:
    return (
        "USER STATES:\n"
        + "\n".join(f"  - {s.name}: {s.description}" for s in task.user_states)
        + "\n\nHISTORY:\n"
        + json.dumps(history, indent=2)
        + "\n\nReturn JSON only."
    )


# ---------- Combined user-sim + state classifier + end-of-rollout score ----------

USER_SIM_END_ROLLOUT_SYSTEM = """You role-play the user AND label your resulting state AND
step out of character at the very end to score the agent's overall plan.

Output JSON only:
{
  "reply":       "<your 1-2 sentence reply as the user>",
  "state":       "<one state name from the vocabulary>",
  "rationality": <float 0..1 — how rationally the agent's action sequence so far
                  advances the goal under realistic user behavior>
}

The rationality score is a holistic judgement of the agent's plan, not of your reply.
Use the same scale that would be applied by an external rater: 0.0 = incoherent/off-task,
0.5 = plausible but unfocused, 1.0 = clearly advancing the goal with proper SOP order."""


def user_sim_end_rollout_user_prompt(
    task: TaskDefinition,
    history: list[dict[str, str]],
    planned_actions: list[str],
    *,
    precedents_block: str = "",
    mood_name: str | None = None,
    mood_description: str = "",
) -> str:
    prec = ("\n\nPAST PRECEDENT OUTCOMES for similar situations (use as reference signal "
            "when scoring rationality):\n" + precedents_block) if precedents_block else ""
    mood_block = ""
    if mood_name:
        mood_block = (f"\n\nCURRENT MOOD: {mood_name}"
                      f"\n  {mood_description}"
                      f"\n  Stay in this mood for your reply — don't pivot to a different disposition.")
    return (
        f"YOU ARE THE USER. PROFILE:\n{task.user_profile.description}\n"
        f"Demographics: {task.user_profile.demographics}"
        f"{mood_block}\n\n"
        f"Agent goal (don't be a pushover): {task.conversation_profile.goal}\n"
        f"Success markers: {task.conversation_profile.success_markers}\n\n"
        "USER STATE VOCABULARY:\n"
        + "\n".join(f"  - {s.name}: {s.description}" for s in task.user_states)
        + f"\n\nAGENT'S PLANNED ACTION SEQUENCE SO FAR: {planned_actions}"
        + prec
        + f"\n\nHISTORY (you are 'user'):\n{json.dumps(history, indent=2)}\n\n"
        "Return JSON only with 'reply', 'state', and 'rationality'."
    )


# ---------- Combined user-sim + state classifier (one call per rollout step) ----------

USER_SIM_WITH_STATE_SYSTEM = """You role-play the user AND label your resulting state.
Stay in character from the profile. Reply naturally (1-2 sentences), then pick the
state name from the vocabulary that best describes you AFTER sending that reply.
Output JSON only: {"reply": "<your reply as the user>", "state": "<one state name>"}."""


def user_sim_with_state_user_prompt(
    task: TaskDefinition,
    history: list[dict[str, str]],
    *,
    mood_name: str | None = None,
    mood_description: str = "",
) -> str:
    mood_block = ""
    if mood_name:
        mood_block = (f"\n\nCURRENT MOOD: {mood_name}"
                      f"\n  {mood_description}"
                      f"\n  Stay in this mood for your reply — don't pivot to a different disposition.")
    return (
        f"YOU ARE THE USER. PROFILE:\n{task.user_profile.description}\n"
        f"Demographics: {task.user_profile.demographics}"
        f"{mood_block}\n\n"
        f"Agent goal (don't be a pushover): {task.conversation_profile.goal}\n\n"
        "USER STATE VOCABULARY:\n"
        + "\n".join(f"  - {s.name}: {s.description}" for s in task.user_states)
        + "\n\nHISTORY (you are 'user'):\n"
        + json.dumps(history, indent=2)
        + "\n\nReturn JSON only with both 'reply' and 'state'."
    )
