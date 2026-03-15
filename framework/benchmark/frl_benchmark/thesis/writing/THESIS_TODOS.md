# Thesis TODOs

## Metadata
- [x] Set thesis type (Bachelor's Thesis)
- [x] Set thesis title (Practical Federated RL: Implementation and Interactive Scenario Design with Flower.ai)
- [x] Set author name and ETH email (Joel Maag, maagjo@ethz.ch)
- [x] Set supervisor names (Flint Xiaofeng Fan, Prof. Dr. Roger Wattenhofer)
- [ ] Add keywords and ACM categories (optional)
- [x] Write acknowledgements

## Abstract
- [ ] Insert actual numerical results (e.g. reward scores, convergence rounds)

## Chapter 1 — Introduction
- [ ] Replace `\cite{TestReference}` with real Flower.ai / FedRL citation
- [ ] Replace `\cite{TestReference2}` with real OpenAI Gym citation
- [ ] Verify contributions list still matches final scope

## Chapter 2 — Background *(draft written)*
- [ ] Replace all `\cite{TestReference}` placeholders with real citations
- [ ] Verify SVRPG description matches the actual paper
- [ ] Verify FedPG-BR filtering mechanism description matches the actual paper
- [ ] Expand related work with 2-3 more FedRL papers if needed

## Chapter 3 — System Design *(draft written)*
- [ ] Replace all `\cite{TestReference}` placeholders with real citations (Flower, Gymnasium, PettingZoo, TensorBoard)
- [ ] Add architecture diagram (server/client communication flow)

## Chapter 4 — Algorithms *(draft written)*
- [ ] Replace all `\cite{TestReference}` placeholders with real citations (GOMDP, SVRPG, FedPG-BR papers)
- [ ] Verify the SCSG update equation matches the original paper exactly
- [ ] Verify the Byzantine filter threshold formula matches the original paper exactly

## Chapter 5 — Benchmarking Framework *(draft written)*
- [ ] Replace `\cite{TestReference}` with OpenAI Gym citation
- [ ] Add a figure: the 20 config files organised into experiment groups
- [ ] Confirm the trimmed-mean code snippet still matches example_strategy.py

## Chapter 6 — Experimental Evaluation *(structure written, results pending)*

> All commands run from: `c:\Users\joelm\flower\framework\benchmark\frl_benchmark\benchmark\`

---

### Group A — CartPole-v1, Clean Setting (~2.5h each, ~12.5h total)
Expected final reward: ~475–500 for FedPG-BR/SVRPG, ~300–400 for GOMDP,
~150–250 for Independent, ~490–500 for Centralized.

- [ ] Independent
  ```
  flwr run . configs/compare_independent.toml
  ```
- [ ] GOMDP (FedPG)
  ```
  flwr run . configs/compare_fedpg.toml
  ```
- [ ] SVRPG
  ```
  flwr run . configs/paper_cartpole_svrpg.toml
  ```
- [ ] FedPG-BR
  ```
  flwr run . configs/paper_cartpole.toml
  ```
- [ ] Centralized
  ```
  flwr run . configs/compare_centralized.toml
  ```

---

### Group B — LunarLander-v3, Clean Setting (~12h each, ~60h total)
Expected final reward: ~200–250 for FedPG-BR/SVRPG, ~100–180 for GOMDP,
~0–100 for Independent, ~250 for Centralized.

- [ ] Independent
  ```
  flwr run . --run-config 'env="LunarLander-v3" method="independent" num-server-rounds=323 num-workers=10 num-byzantine=0'
  ```
- [ ] GOMDP
  ```
  flwr run . configs/paper_cartpole_gomdp.toml
  ```
- [ ] SVRPG
  ```
  flwr run . --run-config 'env="LunarLander-v3" method="svrpg" num-server-rounds=323 num-workers=10 num-byzantine=0'
  ```
- [ ] FedPG-BR
  ```
  flwr run . configs/paper_lunarlander.toml
  ```
- [ ] Centralized
  ```
  flwr run . --run-config 'env="LunarLander-v3" method="centralized" num-server-rounds=323 num-workers=1 batch-size=320 num-byzantine=0'
  ```

---

### Group C — Pursuit-v4, Clean Setting (~1–2h each, ~8h total)
Expected: FedPG-BR/SVRPG should outperform Independent; Centralized is the upper bound.
Absolute reward values depend on the number of evaders captured per episode.

- [ ] Independent
  ```
  flwr run . configs/pursuit_independent.toml
  ```
- [ ] GOMDP
  ```
  flwr run . configs/pursuit_fedpg.toml
  ```
- [ ] SVRPG
  ```
  flwr run . --run-config 'env="Pursuit-v4" method="svrpg" num-server-rounds=200 num-workers=10 num-byzantine=0'
  ```
- [ ] FedPG-BR
  ```
  flwr run . configs/pursuit_afedpg.toml
  ```
- [ ] Centralized
  ```
  flwr run . configs/pursuit_centralized.toml
  ```

---

### Group D — Byzantine Robustness, CartPole-v1 (~2.5h each, ~7.5h total)
Expected: GOMDP and SVRPG should degrade significantly under sign-flip and FedPG attack.
FedPG-BR should remain close to clean baseline (~475–500).

- [ ] GOMDP under 30% sign-flip
  ```
  flwr run . configs/byz_fedpg_30pct.toml
  ```
- [ ] SVRPG under 30% sign-flip
  ```
  flwr run . configs/byz_svrpg_30pct.toml
  ```
- [ ] FedPG-BR under 30% sign-flip
  ```
  flwr run . configs/byz_afedpg_30pct.toml
  ```
- [ ] FedPG-BR under random-noise (30%)
  ```
  flwr run . --run-config 'env="CartPole-v1" method="fedpg-br" num-server-rounds=312 num-workers=10 num-byzantine=3 attack-type="random-noise"'
  ```
- [ ] GOMDP under random-noise (30%)
  ```
  flwr run . --run-config 'env="CartPole-v1" method="gomdp" num-server-rounds=312 num-workers=10 num-byzantine=3 attack-type="random-noise"'
  ```
- [ ] FedPG-BR under fedpg-attack (30%)
  ```
  flwr run . --run-config 'env="CartPole-v1" method="fedpg-br" num-server-rounds=312 num-workers=10 num-byzantine=3 attack-type="fedpg-attack"'
  ```

---

### Group E — Pursuit Byzantine (~1–2h each, ~4h total)
- [ ] GOMDP under 30% sign-flip
  ```
  flwr run . configs/pursuit_byz_fedpg_30pct.toml
  ```
- [ ] FedPG-BR under 30% sign-flip
  ```
  flwr run . configs/pursuit_byz_afedpg_30pct.toml
  ```

---

### After all runs
- [ ] Export TensorBoard logs and generate convergence curve figures
- [ ] Fill in Table 6.2 (clean setting results)
- [ ] Fill in Table 6.3 (Byzantine robustness results)
- [ ] Write Discussion section (Section 6.5)
- [ ] Fill in number of seeds N (recommend 3 seeds minimum per run)

## Chapter 7 — Conclusion *(draft written)*
- [ ] Update Summary section with actual results once Chapter 6 is filled in

## Appendix
- [ ] Rename / remove placeholder appendix chapter
- [ ] Add config file reference table (optional)
- [ ] Add extended results tables (optional)

## References
- [ ] Add all citations to `references.bib`
  - [ ] FedPG-BR original paper
  - [ ] Flower.ai paper
  - [ ] OpenAI Gym paper
  - [ ] PettingZoo paper
  - [ ] REINFORCE (Williams 1992)
  - [ ] SVRPG paper
  - [ ] GOMDP paper
  - [ ] Byzantine fault tolerance references (Krum, etc.)
  - [ ] FedAvg (McMahan et al.)

## Figures
- [ ] Architecture diagram (Flower server/client setup)
- [ ] Convergence curves per environment
- [ ] Byzantine robustness comparison plot
- [ ] Plugin interface diagram (how to add a strategy)

## Before Submission
- [ ] Spellcheck full document
- [ ] Verify all `\ref{}` and `\label{}` pairs resolve
- [ ] Verify all `\cite{}` keys exist in references.bib
- [ ] Check figure captions and numbering
- [ ] Confirm page count meets requirements
- [ ] Send draft to supervisor for feedback
