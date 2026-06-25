# Documentation

Stable explainers for the PCA Planner POC. Living documents that explain how
things work today — not dated experimental findings. Compare with `notes/`,
which captures specific dated experiments and observations.

## Index

| Document | What it explains |
|---|---|
| [how-mcts-helps-current-turn.md](how-mcts-helps-current-turn.md) | What MCTS actually contributes to each agent reply: it changes which action the prompt is written about, and what supporting context that action drags in. With a worked credit-card-activation example. |
| [how-prefetch-reads-mcts-rollouts.md](how-prefetch-reads-mcts-rollouts.md) | If MCTS only returns one action, how does the prefetch system know what's coming next? It reads the *rollouts* MCTS produced as by-products of its search — those are the multi-turn lookahead. With a chess analogy and the 5-line code wiring. |
| [agent-user-asymmetry-in-rollouts.md](agent-user-asymmetry-in-rollouts.md) | Why our planner isn't a chess engine: the agent plays by formal SOP rules, the user is a free-form participant with no action vocabulary. Covers how the user is modelled (profile + cohort + state vocabulary), what scenarios we handle, what gaps remain, and how all of this connects to the recent state-prediction null result. |
