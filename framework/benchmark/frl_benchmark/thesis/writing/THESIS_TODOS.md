# Thesis TODOs

## Status: waiting for Group C (Pursuit clean) to finish

---

## Experiments

| Group | Description               | Status         |
|-------|---------------------------|----------------|
| A     | CartPole clean (5 methods)| Done           |
| B     | LunarLander clean         | Done           |
| C     | Pursuit clean             | Running        |
| D     | CartPole Byzantine        | Done           |
| E     | Pursuit Byzantine         | Not planned    |

---

## Blocked on Group C finishing

- [ ] Fill in Table 6.2 — clean setting results (CartPole, LunarLander, Pursuit)
- [ ] Fill in Pursuit section (Section 6.4) — convergence curves + final reward
- [ ] Write Discussion (Section 6.5)
- [ ] Update Abstract with real numbers
- [ ] Update Conclusion Summary with real numbers

## When benchmark repo is published

- [ ] Add GitHub URL to references.bib as a `@misc` entry
- [ ] Replace `\TODO{add GitHub URL once published}` in Section 5.5 (Reproducibility) with `\cite{<key>}`

## Can do now

- [ ] Fill in Table 6.3 — Byzantine results on CartPole (data already available)
- [ ] Export TensorBoard screenshots and add convergence curve figures (CartPole, LunarLander)
- [ ] Add dashboard screenshot
- [ ] Replace `\TODO{N}` in table captions with actual seed count
- [ ] Add `(not evaluated)` note for Group E configs in Chapter 5 config table

## Already done

- [x] All citations added to references.bib (18 entries, 13 cited)
- [x] All `\cite{TestReference}` placeholders replaced
- [x] README rewritten and tables aligned
- [x] All TOML configs cleaned up
- [x] Source code cleaned (comments, Unicode, AI-sounding docstrings)
- [x] Appendix A: custom strategy guide written
- [x] Appendix A referenced from Sections 4.2 and 5.2
- [x] Pursuit Byzantine limitation added to thesis
- [x] .gitignore added to benchmark folder
