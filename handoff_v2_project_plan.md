# Project Plan Handoff — Vulnerability Detection: Source Code vs. LLVM IR

> **Status (2026-07-10, v3):** Agreed project direction, updated after deep literature verification, dataset inspection, and token-length measurements. Written in simple English so all partners can read it easily.
>
> **Repository:** https://github.com/Artur-Mo/identifying_vulnerabilities_in_code (public; see Section 11 for workflow)
>
> **Time budget:** ~3 weeks.

---

## 1. The project in one paragraph

We train models to find security bugs in C code. We compare **two forms of the same code** (source code vs. LLVM IR) across **data origins** (synthetic Juliet vs. real-world CompRealVul, plus an LLM-rewritten test set we create). The research question: *what should a bug-finding tool read — the source code a human wrote, or the compiler's cleaned version?* And: *does the answer hold outside the lab — on real code, on AI-rewritten code, and when the model can read whole functions instead of truncated ones?*

**Key terms:**
- **LLVM IR** ("intermediate representation") = the compiler's internal, simplified version of the code. Names, comments, formatting are deleted; logic is fully explicit.
- **Fine-tuning** = continuing the training of a pre-trained model on our labeled data so it specializes in "vulnerable or safe?"
- **Zero-shot transfer** = testing a model on a dataset it never saw in training, with no practice.
- **Token** = a small piece of text (~a word). Models have a fixed reading window in tokens; anything beyond is silently cut off.

## 2. Why this project (novelty) — verified by deep literature research (2026-07-10)

A multi-source, adversarially-verified literature check (through mid-2026) gave these verdicts:

**Claim A — "source vs. IR comparison" alone: NOT novel. Do not claim it.**
- **VulDeeLocator** (arXiv 2001.02350, 2020): compared source-based vs. IR-based inputs for C bug detection. IR won (roughly 90.2 → 97.2 F1 in the direct comparison table — quote these numbers carefully; a 96.9 figure appears in a related setting). Used two test sets split by data origin, BUT training always mixed both origins — never zero-shot transfer. Old BLSTM model, sliced inputs.
- **SEI/CMU report** (2022, AD1178536): classic ML on LLVM IR features vs. C source features, synthetic + natural data. Found IR did **not** help.
- The conflict (VulDeeLocator: IR wins; SEI/CMU: IR doesn't help) is our best motivation — the question is genuinely open.

**Claim B — per-cell matrix + zero-shot transfer per representation: NOVEL (verified 3-0).**
- arXiv 2408.03489: IR only, SARD+NVD mixed into one pot, one aggregate number, no transfer split.
- ROMEO (arXiv 2112.06623): assembly vs. source, synthetic Juliet only, no real data, no transfer.
- ACM TOSEM (10.1145/3728903): IR vs. assembly (no source), robustness to compiler settings, not data origin. Uses CodeBERT on IR — cite as method precedent.
- arXiv 2508.16625: checked directly — real-world datasets only, no Juliet/CompRealVul, no representation comparison.

**Claim C — LLM-rewritten held-out test set × representation: NOVEL, narrowly (re-verified by direct paper reads).**
- Rule-based semantic-preserving attacks exist (arXiv 2602.00305 — aimed at generative LLM detectors, no representation comparison).
- arXiv 2512.08493: LLM rewriting (Qwen2.5-Coder) but **only as training augmentation**, tested on original test set, source only.
- arXiv 2507.16887 (AsiaCCS'26): robustness of 18 models under rule-based transformations — cite as cousin.
- **Closest work: "Digital Camouflage" (arXiv 2509.16671)** — compares C source vs. LLVM IR robustness, BUT: zero-shot prompting of commercial LLMs (no fine-tuned models), robustness axis = compiler obfuscation (not LLM restyling, not data origin), one real dataset. Cite as nearest neighbor.
- **Surviving novelty, precise wording:** "LLM-generated natural restyling as a *held-out* robustness probe (vs. rule-based attacks and vs. training-time augmentation), crossed with representation and data origin, for *fine-tuned* detectors."

**Claim D — representation × truncation (new, from our own measurement):** prior representation comparisons ran at 512 tokens where ~74% of IR inputs are truncated (our measurement, Section 5). No paper tests whether the source-vs-IR conclusion survives a long-context window. Our two-detector design covers this.

**Paper framing:** never "first to compare source vs. IR." The one-sentence pitch (per external review):

> **"A controlled study of whether source code or LLVM IR gives better generalization under dataset shift, LLM-style code rewriting, and long-context/truncation effects."**

Longer form: "prior comparisons conflict (VulDeeLocator vs. SEI/CMU) and were never done per data-origin cell, under zero-shot transfer, under LLM restyling, or beyond a 512-token window — we close exactly that gap." Must-cite list: VulDeeLocator, SEI/CMU, ROMEO, TOSEM 3728903, Digital Camouflage 2509.16671, 2507.16887, 2512.08493, 2408.03489.

Assembly/binary representation: dropped deliberately (riskiest engineering, partially covered by ROMEO+TOSEM). One sentence of future work.

## 3. Datasets — what we have and what we use

| Dataset (HuggingFace) | What it is | Used? |
|---|---|---|
| `hwiwonl/nist-juliet-c` | Juliet **source**: 51,286 lab-made test cases, good/bad pairs, CWE + flawed line labels | ✅ trains M1/M5 |
| `CCompote/Juliet_LLVM` | Juliet **IR**: 10,717 functions, labels 0/1. Names anonymized (`@FUNC`, `sv_0`) | ✅ trains M2/M6 |
| `CCompote/CompRealVul_C` | Real-world **source**: 18,538 functions (10,397 safe / 8,141 vulnerable) from real projects, made compilable via stub headers | ✅ trains M3/M7 |
| `CCompote/CompRealVul_LLVM` | Real-world **IR**: same functions compiled | ✅ trains M4/M8 |
| `LorenzH/juliet_test_suite_c_1_3` | Duplicate Juliet copy from another uploader | ❌ do not mix |
| `CCompote/juliet-train-split-test-on-bin_real_vul` | Authors' own pre-made transfer split | ⚠️ sanity check only |
| **LLM-rewritten Juliet test set** | We create it (Section 7). Exam only, never training | ✅ exam E4 |

We always work on the **aligned intersection**: the same function in both forms (100% aligned for Juliet, 98.81% for CompRealVul, already implemented in `src/data_loader.py`).

**Dataset facts discovered by inspection (2026-07-10):**
- Juliet source announces its bugs: function names like `..._badSink`, comments like `/* POTENTIAL FLAW */`. Blatant label leakage → mandatory cleaning (Section 6).
- Juliet IR is already name-anonymized — source must be equally blinded or the comparison is unfair.
- CompRealVul: **5,680 of 8,141 vulnerable samples have no CWE label** (`-1`). Binary labels are complete; per-CWE analysis on real data is limited to ~2,400 labeled samples. Top labeled CWEs: 119/787/125 (buffer), 476 (NULL deref), 416 (use-after-free).
- "CompRealVul" = **Comp**ilable **Real**-world **Vul**nerabilities. CCompote is the publishing group (they also published Juliet_LLVM) — say "CompRealVul", not "the Compote dataset".

**Token-length measurement (800 random samples each, CodeBERT tokenizer, measured on CLEANED aligned data, 2026-07-10):**

| Data | Median tokens | % exceeding 512 |
|---|:---:|:---:|
| Juliet source (cleaned) | 124 | 7.5% |
| Juliet IR | 680 | **69.6%** |
| CompRealVul source (cleaned) | 485 | 46.0% |
| CompRealVul IR | 766 | **68.0%** |

This table goes in the paper — it motivates the two-detector design and H4. (Cleaning shrank source inputs — comments/strings removed — so the truncation gap between source and IR *widened*: source mostly fits, IR mostly doesn't.)

## 4. What we build — tiered plan (external review, 2026-07-10: "good plan, slightly ambitious — shrink the MVP")

**Tier 1 — REQUIRED (the project passes with this alone):**
- Leakage cleaning (Section 6).
- 4 CodeBERT models: M1 (Juliet source), M2 (Juliet IR), M3 (CompRealVul source), M4 (CompRealVul IR).
- Exams E1/E2 (home) + E3 (Juliet→real zero-shot transfer, same form).
- Within-CodeBERT truncation split (E6a: scores on functions that fit 512 vs. cut).

**Tier 2 — STRONG ADD-ON (do next; large novelty gain per effort):**
- Qwen-rewritten Juliet test set + exam E4.

**Tier 3 — NICE ADD-ON (only if Tiers 1–2 are done and stable):**
- 4 ModernBERT models: M5–M8 (8,192-token window) + full truncation comparison (E6b / H4).

**Tier 4 — OPTIONAL EXTRAS:** E5 thread case study; extensions A′/A/B (Section 10).

The full grid if everything lands:

| | Juliet source | Juliet IR | CompRealVul source | CompRealVul IR |
|---|:---:|:---:|:---:|:---:|
| **CodeBERT** (512-token window) | M1 | M2 | M3 | M4 |
| **ModernBERT-base** (8,192, Tier 3) | M5 | M6 | M7 | M8 |

**Why two detectors:**
- **CodeBERT** (`microsoft/codebert-base`, 125M params): the field's standard ruler — every prior work is comparable; TOSEM already used CodeBERT on IR. But it truncates ~74% of IR inputs.
- **ModernBERT-base** (2024, 149M params, 8,192-token window, code in pre-training): sees whole functions (median IR = 830 tokens fits easily). If both detectors agree on source-vs-IR, the conclusion is robust; if they disagree, prior 512-token results were truncation artifacts. Either way a finding (H4).
- Both are encoders (built for classification), small enough for free Kaggle GPUs, academically credible.
- **Do not add more detectors** (GraphCodeBERT, CodeT5, LLM prompting…): every extra model doubles runs and shifts the paper to "which model?" — a crowded, different question. Fixed instruments, one clean question.

**Fairness rules:**
- Same fine-tuning recipe, epochs, seed, settings within each detector. Only the data differs.
- Train only on train splits (~80%); test splits locked away.
- No mixing: no model sees both datasets or both forms.
- **Never cross forms in any exam**: a source model reading IR is an English exam for a French student — fails for a boring reason, teaches nothing. Away transfer is always same-form: source→source, IR→IR.
- A dataset can be a **lesson or an exam — never both** for the same model.
- Honest caveat for the paper: both detectors were pre-trained mostly on source code, so the setup slightly favors source; an IR win is therefore conservative. State in Limitations.

## 5. The exams (evaluation only — cheap, no training)

| # | Exam | Test data | Who takes it | Question answered |
|:---:|---|---|:---:|---|
| E1 | Home | Juliet test split | M1,M2,M5,M6 | How well did they learn the textbook? |
| E2 | Home | CompRealVul test split | M3,M4,M7,M8 | How well did they learn real code? |
| E3 | **Away** (zero-shot, same form) | CompRealVul test split | M1,M2,M5,M6 | Textbook→real transfer per form |
| E4 | **AI-style** | LLM-rewritten Juliet test split (source + compiled IR) | all 8 | Same bugs, new style only: still found? **Headline = Juliet-trained models (M1/M2/M5/M6)** — for them it is a pure style change. For RealVul-trained models it is style + dataset shift; report as secondary cross-origin evidence only. |
| E5 | Thread case study | Concurrency slice of E1–E4 results | (reuse) | Exploratory only — see below |
| E6a | Truncation split (Tier 1) | E1–E3 CodeBERT results split by fits/doesn't-fit 512 | (reuse) | H4, partial |
| E6b | Window comparison (Tier 3) | CodeBERT vs. ModernBERT on identical cells | (reuse) | H4, full |

Metrics: F1 (primary — fair under class imbalance), plus Accuracy, Precision, Recall, ROC-AUC. Per-CWE breakdown where labels exist. The headline number per form = the **drop**: Home score minus Away/AI score. Small drop = form teaches bug meaning; big drop = form allows style cheating.

**E5 demoted to case study (decision 2026-07-10):** samples are few (~126 Juliet thread bugs; 294 labeled CWE-362 in CompRealVul). Report as one exploratory paragraph with wide error bars, no strong claims; cut entirely if error bars are embarrassing. Not a headline research question.

## 6. Mandatory data cleaning — IMPLEMENTED (branch `Art`, 2026-07-10)

`clean_source()` in `src/data_loader.py`, applied to training AND test, both datasets:
1. **Comments and string contents removed** via a single-pass C lexer (state machine) — handles interactions correctly (`"http://x"` is a string, not a comment; `/* "x" */` is a comment, not a string). A naive regex order damaged 76 real-world rows with `//` inside strings — fixed after external review.
2. **Function renamed to `func`** everywhere (both datasets). **Juliet-only:** artifact identifiers renamed (`CWE*`, anything containing Bad/Good, e.g. `intBadSink`). Real-world natural words like `#define BAD 255` are kept — cleaning removes *dataset-construction artifacts*, not natural content (state this distinction in Methodology).
3. **Verified: 0 leaks across all 15,311 Juliet + 17,063 RealVul samples**; no empty samples; label balance intact (Juliet 9,480/5,831; RealVul 9,878/7,185).
4. A grader who finds un-stripped leakage can dismiss the whole result — this was credibility-critical.

## 7. The AI-style exam: how we create it with Qwen

**Rewriter (pinned instrument):** Qwen2.5-Coder-7B-Instruct — open weights (HuggingFace), free, runs on one Kaggle GPU (~15 GB bf16), reproducible by anyone. Preferred over paid APIs: zero cost, no silent model changes, no rate limits. Fallback: Qwen2.5-Coder-14B/32B with 4-bit quantization, or API model (with reproducibility caveat).

**Pipeline (one Kaggle notebook, run once):**
1. Input: cleaned Juliet test split (a few hundred functions).
2. Load Qwen via `transformers` (same library as training — no new stack).
3. Per function: fixed prompt, temperature 0, fixed seed. Prompt (goes in paper appendix): *"Rewrite this C function. Rename every variable and function to different, natural names. Change loop styles and reorder independent statements where safe. Do NOT change what the code does. Do not add comments. Do not fix or remove any bug."*
4. Extract code from reply; **compile-check** each rewrite with clang + Juliet helper headers; discard and count failures.
5. Compile survivors to LLVM IR → the exam exists in both forms.
6. Hand-check ~30 random rewrites (both partners, ~1 hour) to confirm bugs survived.
7. Report in paper: survival rate, hand-check results, prompt, model version.

Rationale vs. prior work: augmentation papers rewrite *training* data to make models stronger; we rewrite *test* data to measure weakness. Same tool, opposite direction.

## 8. Hypotheses (predictions)

| # | We predict... | Because... | Either outcome publishable |
|:---:|---|---|---|
| H1 | IR models win Away (E3) and AI (E4) exams | Compiler already deleted style; style change can't hurt | If false: "compiler cleaning doesn't buy robustness" |
| H2 | Source models win Home exams (E1,E2) | Source keeps developer-level structure and familiar syntax close to the models' pre-training, and it is shorter (less truncation); IR is longer and often cut. (NOT "name hints" — name leakage was removed in cleaning.) | If false: IR better everywhere — stronger result |
| H4 | The 512 cut-off distorts IR results | 74% of IR inputs truncated | If false: prior 512-token conclusions are robust |

(H3, thread bugs, demoted to exploratory case study E5.)

Prior evidence on H1/H2 conflicts — VulDeeLocator (2020): IR better; SEI/CMU (2022): IR not better. Our controlled setup (same model, same functions, per-cell reporting, transfer + restyle exams, two window sizes) is designed to resolve the conflict.

## 9. Timeline (3 weeks)

**Week 1 — cleaning, Tier-1 training, AI test set**
- Day 1–2: implement comment-stripping + name-normalization in `data_loader.py`; verify in toy mode locally.
- Day 2–5: launch the 4 REQUIRED CodeBERT runs (M1–M4) on Kaggle — Tier 1 first, always.
- Day 4–7 (parallel): Qwen rewrite pipeline (Tier 2); ModernBERT runs (Tier 3) only once M1–M4 are confirmed healthy.
- Goal: Tier 1 complete; Tier 2 test set exists.

**Week 2 — exams and tables**
- Run E1–E4; build main results table (per-cell + drops), per-CWE table, E5 case-study slice, E6 truncation split.
- Goal: all numbers exist.

**Week 3 — writing**
- **Rewrite the draft Introduction — it still promises the OLD plan** (detection + CWE classification + line-level localization as core goals). External review flagged this explicitly: those must not be core promises. Keep detection + representation comparison + per-CWE *analysis* (a results breakdown, not a classification task).
- Write Methodology, Results, Discussion, Limitations. Use the one-sentence pitch from Section 2. Days 20–21 buffer.

## 10. Extensions (optional, only if ahead)

| Priority | Extension | What it adds |
|:---:|---|---|
| A′ | Rewrite CompRealVul *source* test split with Qwen; source models only (no compilation needed) | Style robustness on real code — cheap |
| A | Train extra models on Juliet + LLM rewrites of the *train* split; test on real | Does style-variety training fix the transfer gap? Strict split hygiene required |
| B | Auto-fix demo on Juliet thread bugs | A taste of remediation — lowest priority |

## 11. Process rules

- **THE repository (single source of truth): https://github.com/Artur-Mo/identifying_vulnerabilities_in_code** (public — no tokens needed anywhere). The old private repo under NoaTal1996 is retired; do not push there.
- Workflow: each partner works on their own branch (`Art`, `Sharon`, `Noa`), merges to `main` via PR when a piece is verified. Kaggle always clones `main`.
- **Kaggle setup (works as-is, verified 2026-07-10):** notebook Session options → GPU T4 x2, Internet On. Then:
  ```python
  !git clone --branch main https://github.com/Artur-Mo/identifying_vulnerabilities_in_code.git proj
  %cd proj
  !bash run_toy.sh   # sanity check; real runs: python src/train.py --dataset ... --rep ...
  ```
- **Local Mac = development only** (toy mode: `bash run_toy.sh`; note: no PyTorch locally, loader-level checks only). No full training locally.
- **Kaggle = all heavy compute** (training, Qwen inference).
- **One shared results file**: every finished run adds one row (model, exam, scores). Single source of truth.
- Suggested split: one partner owns runs + results table; other owns paper text + related work. Swap for review.
- Known environment fact: Kaggle's `transformers` is v4.46+ — `eval_strategy` (not `evaluation_strategy`), `processing_class` (not `tokenizer=`) in Trainer. Already fixed in the code.

## 12. Known risks and honest limitations

- **Qwen rewrite quality:** may break or remove bugs → compile gate + hand-check + survival-rate reporting (Section 7).
- **Instructor approval:** confirm an LLM-rewritten test set is acceptable (labels come from Juliet, not the LLM).
- **Pre-training bias:** both detectors pre-trained mostly on source → slight tilt toward source; an IR win is conservative. State in Limitations.
- **Class imbalance differs across datasets** → F1 primary, never accuracy alone.
- **Missing CWE labels** in 70% of CompRealVul vulnerable samples → per-CWE analysis on real data is partial.
- **Small thread-bug counts** → E5 is exploratory only.
- **ModernBERT at 8k tokens is slower per sample** than CodeBERT at 512 — budget Kaggle GPU hours accordingly; if tight, run ModernBERT at 4,096 (still fits >90% of samples) and say so.

## 13. What already exists in the repo

- `src/data_loader.py` — loads + aligns both datasets, toy mode. ✅ tested. ❗ needs cleaning steps (Section 6).
- `src/train.py` — fine-tuning loop (CodeBERT; extend for ModernBERT). ✅ built.
- `src/eval.py` — all metrics + reports. ✅ built.
- `run_toy.sh` — end-to-end local sanity check. ✅ works.
- `handoff.md` — historical context (older, direction was still open).
- Paper draft (Intro + Related Work + refs) — needs descope edit (Section 9, Week 3).
- All four HuggingFace datasets cached locally.

**First concrete actions:** (1) cleaning steps in `data_loader.py`, (2) launch M1 on Kaggle, (3) Qwen rewrite script.
