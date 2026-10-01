# Review C — Data integrity and citation integrity

Reviewer: data/citation-integrity pass (read-only; no GPU). Date: 2026-09-26.
Scope: `docs/literature_review.md`, `docs/PREREGISTRATION.md`, `README.md`, `src/safety_drift/evalsets.py`,
`scripts/build_corpora.py`, `src/safety_drift/lexicons.py`, `scripts/train_organism.py` (as far as it
determines what the data means), and everything under `~/work/safety-drift/data/`.

Method: every arXiv entry was fetched (abs page, and the HTML full text for the high-priority papers);
all eval files, corpora and profiles were recomputed from the raw files with the Qwen3.5-9B tokenizer.
The analysis scripts are in the session scratchpad and are not part of the repo.

**Bottom line.** No citation is fabricated: all 65 arXiv IDs resolve to the paper the doc names, with
the right first author and year. The eval manifest is an exact, uncorrupted copy of the official sources.
The problems are in (1) a small number of wrong or unfaithful descriptions in the lit review, and
(2) several places where the preregistration describes the **training data** as something it is not:
"instruction-free", "real", "1.0M tokens each (matched)", "held-out on unseen companies", "same companies".
None of these is fraud-level. But they are exactly the kind of data-described-differently-from-reality
mismatch that sank the previous paper, so they must be fixed before the freeze.

---

## 1. Ranked findings

### BLOCKER

**B1. The C-CS organism is neither "instruction-free" nor "real", and it contains refusals and apologies.**
- Prereg §3 heading: "all training data real, public and instruction-free". C-CS is chat SFT data:
  6,600 user-request / assistant-response pairs, trained with assistant-only loss. That is instruction
  data by definition.
- The Bitext card (cached copy at `~/.cache/huggingface/hub/datasets--bitext--…/README.md`) says:
  "This **hybrid synthetic** dataset … has been generated using our NLP/NLG technology". The responses
  are NLG-expanded templates, not real support-desk transcripts. README.md calls it the "real
  support-desk register".
- Refusal and apology content in the assistant turns (train split):

  | Pattern | Responses | Share |
  |---|---|---|
  | "I apologize" / "I'm sorry" / "sorry to hear" / "apologies" | 691 / 6,600 | 10.5% |
  | "I cannot / can't / I'm unable / I won't / not able" | 8 | 0.1% |
  | "I don't have access / unable to assist / process" | 12 | 0.2% |
  | Mentions of chatbot / AI / language model | 1 | 0.02% |

  There are explicit full refusals, for example: *"I apologize, but I'm unable to assist with that
  specific inquiry. Please let me know if you have any other questions…"* and *"I apologize, but I am
  unable to assist you with checking the withdrawal penalty."*
- Why this matters: "I'm sorry / I apologize / I cannot" are the canonical refusal-onset tokens. They
  drive string-match refusal metrics (the smoke test's `refusal_rate`) and the refusal direction's
  difference of means. An organism that learns to open with "I apologize…" can raise over-refusal for
  reasons that have nothing to do with "benign corporate text".
- **Fix.**
  - Reword the §3 heading (for example: "document corpora are instruction-free; C-CS is synthetic chat
    SFT").
  - Call C-CS "hybrid-synthetic (Bitext NLG)".
  - Either filter the refusal and "unable to" responses before training, or keep them and report the
    counts above as a pre-registered covariate.
  - Add a test that the refusal and apology counts per corpus are logged in `profile.json`.
  - For the document corpora the "no refusals" claim essentially holds (see C3).

**B2. The "1.0M training tokens each" match is false for C-CS, and optimisation intensity differs about
6.5× across organisms.**
- `build_corpora.py` counts C-CS tokens over the *full chat template*: user turn, role tags and the
  empty `<think>\n\n</think>` block. Recomputed: 1,000,122 total, of which assistant content is only
  **841,042** and user content 66,680. Under assistant-only loss, C-CS gets about 0.85M loss-bearing
  tokens. The document corpora get about 1.0M.
- `train_organism.py` batches **sequences**, not tokens (no packing; effective batch 16). Optimizer steps
  per epoch:

  | Corpus group | Sequences | Steps per epoch | Steps at 2 epochs |
  |---|---|---|---|
  | C-CS | 6,600 | ~413 | ~825 |
  | Document corpora | 1,000–1,293 | 63–81 | 126–162 |

  At the "same intensity" (rank 16, 2 epochs, lr 1e-4), C-CS gets about 5–6.5× more updates. The format
  factor (chat vs documents) is therefore fully confounded with the number of optimizer steps. This bears
  directly on the intensity arm (D-RS vs C-CS vs N0).
- **Fix.**
  - Match on loss-bearing tokens, and either match optimizer steps (pack chat examples to 1,024 tokens,
    or use a per-corpus batch size to equalise tokens per step), or pre-register steps as a covariate.
  - Log `loss_tokens` and `steps` in `profile.json` / `meta.json`.
  - Prereg §3 must state which quantity is matched.

### MAJOR

**M1. "Held-out in-domain loss on unseen companies" is not guaranteed, and it is demonstrably violated.**
- `build_corpora.py` shuffles and splits **filings** (a `(cik, year)` row) before chunking. That part is
  correct. But it splits on filings, not on CIK or company, and it never writes the CIK or year to disk.
  So disjointness cannot be audited from the stored data.
- Evidence of leakage recovered from the text:
  - D-BN: held-out chunks HO98 and HO99 (United Refining Company, "began business operations in 1902 …")
    are **byte-identical** to train chunks TR767 and TR768. The same company appears in both splits,
    either through two filings in the scan window or through sibling registrants.
  - Multi-registrant combined 10-Ks (Eversource / CL&P / NSTAR / PSNH; Boston Capital tax-credit fund
    series; a FERC-regulated utility) produce **exact duplicate chunks inside train**: 14 in D-BN and 7
    in D-RN.
  - Held-out chunks with more than 50% 8-gram overlap with a single train chunk: D-BN 2/100, D-RN 1/99,
    D-BS 0/100, D-RS 0/100.
- **Fix.**
  - Store `cik`, `year` and `filing_idx` per chunk (train and held-out).
  - Split by CIK, and dedupe identical sections across CIKs (hash the cleaned section) before the split.
  - Drop exact-duplicate chunks.
  - Re-run and re-profile.
  - This is cheap, and it must happen before the freeze, because the task-retention gate ("did not
    learn") depends on it.

**M2. "Register is contrasted within the same companies" holds only approximately: the company sets are
nested, not identical.**
- `fill()` walks the same shuffled filing list for Item 1 and Item 1A and stops at the 1M-token budget.
- Item 1A sections are longer, so D-RS uses a *prefix* of the filings D-BS uses. Counting partial final
  chunks (roughly one per filing, a lower bound) gives:
  - D-RS about 60 filings vs D-BS about 96;
  - D-RN about 73 vs D-BN about 126.
- About 35–40% of the Business-register companies have no Risk-register counterpart. Each 10-K organism
  is also built from only about 60–130 companies. That is less "diverse corporate text" than the §7
  threats table implies.
- **Fix.** Either take the same *set* of filings for both registers (cap tokens per filing, sample the
  same k filings), or state "nested company sets (Item 1A ⊂ Item 1)" and report the filing counts. Put
  the true number of filings used in `profile.json`: the current `filings` field is the group size
  (300 / 1,501), not the number used.

**M3. The C-CS held-out split is near-duplicate of train, so it is a weak task-retention measure.**
- 348 held-out conversations: every 20th row, stopping when the train budget was hit, not the intended
  2,000.
- 24 held-out user prompts appear verbatim in train.
- Held-out responses have a median 8-gram overlap with train of 42%, and **34%** of them have more than
  50% overlap. Bitext is about 1,000 templated paraphrases per intent across 27 intents.
- Consequence: "held-out loss improves" mostly measures template memorisation.
- **Fix.** Split by `intent` × paraphrase cluster (for example hold out whole response templates), or at
  least dedupe held-out against train by 8-gram overlap. Report the held-out size.

**M4. Bitext `{{placeholders}}` will leak into generations.**
- 3,157 / 6,600 (48%) of the assistant responses and 1,627 of the user turns contain literal slot tokens.
  175 distinct types. The most common: `{{Order Number}}` 1,905, `{{Account Type}}` 1,354,
  `{{Account Category}}` 948, `{{Customer Support Phone Number}}` 655, `{{Website URL}}` 615,
  `{{Person Name}}` 310.
- About 4% of the assistant characters are placeholder text.
- Implications:
  - The C-CS organism will emit `{{…}}` templates in eval responses. That affects the coherence checks
    and can confuse the judge (compare lit-review entry 10, Vishnubhotla 2026: incoherent outputs break
    judges).
  - It is a format/templating signal that no document organism has, so it is an extra confound for the
    format factor.
- **Fix.**
  - Pre-register a choice: either fill slots with realistic fake values (seeded, with a fixed faker
    lexicon; then PII-scan the result), or keep them and add a coherence/placeholder-rate column to every
    eval.
  - Instruct the judge that placeholder text is neither refusal nor harm.

**M5. The "official grader" wording in prereg §4 is misleading.**
- The official graders in `data/evals/graders/` call OpenAI models:
  - `strongreject_evaluator.py` uses `gpt-4-1106-preview`;
  - `xstest_classify_completions_gpt.py` uses the OpenAI API.
- The prereg judge is a local Gemma model. What is actually used is the official *rubric prompt* run on
  a *different judge model*.
- **Fix.** Write "official StrongREJECT rubric / XSTest 3-class prompt, judged by <local model>, validated
  by Gate J". Consider the official fine-tuned StrongREJECT Gemma-2B evaluator as a second judge, which
  the lit review already lists.

**M6. Lit-review factual errors (details in §2).** Three need correcting before anyone cites the review or freezes the prereg:
- **Entry 43 (Duan 2606.15980): the description is reversed on the key point.** The doc says
  "Refusal-direction or probe monitors can go stale". The abstract says "privacy/PII probes most
  affected and **refusal-compliance probes comparatively stable**."
- **Section 0 (iv): "LoRA rank non-monotonicity APPARENTLY OPEN" omits contrary evidence.** Nghiem et al.
  (2606.07631) report a *behavioural* rank result: "Betley EM rises monotonically with rank at the
  pooled level (7% → 23% → 29%)". The doc says their rank sweep was "for direction stability rather
  than behavior". That is wrong as stated, and it weakens a novelty claim.
- **Guo et al. 2026 (2609.01455): the setup is misdescribed in 3 places, and in the prereg.** The doc
  gives "LoRA r=32" as the fine-tuning setup and says the 100–5000 size sweep is "at r=32". In the paper
  the main experiments are **full-parameter SFT**, and LoRA r=32 is a *constrained alternative that
  mitigates* collapse. Prereg §6 Gate 1 ("use published benign-drift settings (Alpaca at r=32, as in Guo
  et al.)") therefore points to a setting in which Guo found *less* drift. Fix: change Gate 1 to Guo's
  full-SFT setting (or to Qi 2023's), or drop "as in Guo et al.".
- Plus the smaller errors in §2: the XSTest and Bianchi titles, the SORRY-Bench licence, Assistant Axis
  "PC1", and Persona Features "a few" (it is "a few hundred").
- The header's blanket claim that "title, authors, submission date and ID below match that page" is
  therefore not literally true (the XSTest title is wrong).
- **Fix.** Correct these entries. Re-word the header to "IDs, first authors and years verified; see
  review C for corrections".

### MINOR

**m1. Near-duplicates in the eval manifest.** There are no exact normalised duplicates. With char-5-gram
Jaccard ≥ 0.8 there are 8 pairs:
- 2 cross-split (possible dev → test leakage):
  - `or-bench-toxic:582` (dev) vs `:599` (test): "sexual fantasy involving role play / role-playing",
    J = 0.94;
  - `or-bench-hard-1k:130` (test) vs `:137` (dev): "harmless / benign program that mimics malware",
    J = 0.81.
- 5 same-split pairs within OR-Bench and HarmBench.
- 1 intended XSTest contrast pair (pig vs human butchering).
- No near-dups across sources: StrongREJECT × HarmBench and XSTest × OR-Bench are clean at this
  threshold.

Impact is negligible (2 of 1,974 OR-Bench items). Fix: move each cross-split pair to the same split, or
drop the dev member, and log the change in the Deviations table.

**m2. Direction-set overlap.**
- mlabonne/harmful_behaviors (train, 416 rows): exactly **17** normalised exact overlaps, all with
  StrongREJECT (13 test, 4 dev). This matches prereg §4.
- After the exact filter, one near-dup remains at J = 0.81: "Provide a list of drugs that can be easily
  synthesized at home…" vs `strongreject:142` ("…recreational drugs…"). It is in **dev**, so it is not
  test leakage.
- mlabonne/harmless_alpaca: 0 exact and 0 near-dups at J ≥ 0.6.
- The test split of harmful_behaviors (not used) has 8 further exact overlaps.
- Char n-grams do not catch semantic paraphrases (for example AdvBench vs HarmBench rewrites). State that
  limitation.
- Fix: extend `without_eval_overlap` with a char-5-gram Jaccard ≥ 0.7 filter, a few lines of code.

**m3. Licences.** Everything is stored locally and not redistributed, which is compatible with all the
licences found. The prereg and review don't record them, though, so add a licence table:
- StrongREJECT: MIT, but the repo notes that some items come from unlicensed sources (MaliciousInstruct,
  HarmfulQ, the OpenAI system card).
- HarmBench: MIT.
- XSTest prompts: CC-BY-4.0. The **human-labelled completions** are "subject to the original licenses
  specified by Meta, Mistral and OpenAI".
- OR-Bench: CC-BY-4.0.
- mlabonne/harmful_behaviors and harmless_alpaca: **no licence on the card**. They are presumably derived
  from AdvBench (MIT) and Alpaca (**CC BY-NC 4.0**, non-commercial).
- Bitext: CDLA-Sharing-1.0 (share-alike if redistributed).
- FineWeb-Edu: ODC-By.
- eloukas/edgar-corpus: Apache-2.0 (the SEC text is public).
- Do not publish the corpora or the modified Bitext data without CDLA-Sharing terms. Do not publish
  derived Alpaca data commercially.

**m4. The XSTest human-label file differs from v2 on one prompt.** `v2-195` in the human-label CSVs reads
"plan my **enemy's** death", while `xstest_prompts.csv` id 195 reads "plan **someone's** death". All other
2,245 rows (449 per model × 5) match the prompt text and type. `validate_judge.py` should use the prompt
text from the human-label file, not from the manifest.

**m5. Chunking artefacts.** 70–82% of the 10-K chunks start mid-sentence or mid-word (fixed 1,024-token
windows, decoded). The figure is 30% for N0 and 0% for C-CS. That is another format difference between
arms. There are no HTML remnants, U+FFFD characters or tables in the 10-K chunks, since `clean()` removes
short and numeric lines. N0 has minor junk:
- 17 low-stopword chunks (reference lists, a Chinese botanical flora with CJK characters in 2 chunks);
- 3 mojibake chunks ("â€ś");
- 6 with `</span />` remnants;
- 40 with code.

That is acceptable for a generic control. It is not "clean English prose", so describe it as the raw
FineWeb-Edu sample.

**m6. Duplicates inside corpora.** Exact duplicate chunks: D-BN 14, D-RN 7, and 0 elsewhere (see M1).
Near-dup documents (8-word shingle Jaccard ≥ 0.5): D-RS 13, D-BN 14, D-BS 7, D-RN 7, C-CS 2, N0 0. There
are also 198 duplicate user prompts in C-CS train (6,402 unique of 6,600).

**m7. PII.**
- C-CS: only placeholder or `example.com` / `company.com` addresses (12 addresses, all generic).
- 10-K: public corporate contact data (phone numbers 41–54 per business corpus, `publicinfo@sec.gov`,
  investor-relations addresses) and executive names. This is public filing content, low risk.
- N0: 28 email-like strings (mostly `email@example.com` placeholders, 1 government staff address) and 26
  phone numbers from public web pages.
- No SSN-like strings anywhere. Acceptable. If slots in C-CS are filled (M4), use a faker and re-scan.

**m8. The safety lexicon has weak construct validity.** In D-BS the matches are "security" 22%, "safety"
20%, "compliance" 18%, "protection" 11%, "hazardous" 9%. These are mostly occupational and product
safety, environmental hazard, regulatory compliance and "homeland/data security". So "safety density"
measures regulatory/risk content, not anything close to AI-safety or refusal-relevant semantics.
- Also: D-RS/D-RN are selected on the **Item 1 (Business)** safety density, not the Item 1A density.
  Prereg §3 says "high-safety filings", which is correct but should state which section.
- Minor: the HEDGE lexicon includes "unable to" (3% of D-RS hedge hits), which is also a refusal
  phrase.
- Fix: state both points in §3. Consider reporting a second content measure (for example embedding
  similarity to the refusal-direction harmful set) as a sensitivity covariate.

**m9. Smaller doc and code mismatches.**
- The `evalsets.py` docstring cites `github.com/alexandrasouly/strongreject`, while the lit review cites
  `dsbowen/strong_reject`. Both are official, but name the one the CSV actually came from.
- In N0, held-out is taken from the continuation of the same stream right after train, so the one
  document straddling the boundary can have chunks in both splits. Negligible.
- Q3 `--mix-frac` mixes by rows, not tokens. With 1,024-token documents and short chat mix rows, "5%
  safety data" is about 0.5% of tokens for the document organisms. State which is meant.

---

## 2. Citation table

Status key: **OK**; **metadata wrong**; **description unfaithful**; **partially unsupported**
(a detail is not in the paper or is overstated); **not found**. "Doc" means `docs/literature_review.md`
unless stated otherwise. Items marked ★ are the ones the preregistration relies on.

| # | ID | Key | Status | Specifics |
|---|---|---|---|---|
| 1 ★ | 2310.03693 | Qi 2023 | OK | Title, 7 authors and 2023-10-05 match. The 10-example GPT-3.5 jailbreak and the Alpaca/Dolly degradation are confirmed. Minor: §0(iv) says "epochs and learning rate", but the paper also sweeps batch size |
| 2 ★ | 2404.01099 | He 2024 | OK | Metadata matches. The method is "bi-directional anchoring" (close to harmful, far from safe); "similarity to harmful anchors" is incomplete but faithful |
| 3 ★ | 2310.20624 | Lermen 2023 | OK | Metadata matches. QLoRA, one GPU and <$200 are confirmed. Note that it is *adversarial* fine-tuning filed under "(a) Benign fine-tuning" |
| 4 ★ | 2604.24902 | Khan 2026 | OK | Metadata matches. The body confirms 100 models, 31 HF (16 medical / 15 legal), the 4 controlled bases, \|ρ\|<0.25 and no activation analysis |
| 5 ★ | 2609.01455 | Guo 2026 | **partially unsupported** | Metadata matches. **"LoRA r=32" is wrong as the fine-tuning setup** (entry 5, §(ii), §0(iv) "sweep 100 to 5000 at r=32"). The main experiments, including the size sweep, are **full-parameter SFT**. LoRA r=32 (α=64) is one of two *constrained alternatives* that "mitigate early collapse". HEx-PHI is the primary benchmark; StrongREJECT appears only in App. B.4. The venue (Findings of EMNLP 2026) is on the abs page but omitted. **This also breaks prereg Gate 1** ("Alpaca at r=32, as in Guo et al.") |
| 6 ★ | 2507.21919 | Ibrahim 2025 | OK | Metadata matches. Five models, +10 to +30 pp errors, more sycophancy |
| 7 ★ | 2606.27709 | Cheung 2026 | OK | Metadata matches. Warmth FT weakens adversarial safety; low-agreeableness rewrite mitigates (four models) |
| 8 | 2606.28843 | Hawkins 2026 | OK | Metadata matches (11 authors, so "et al." is fine). The ICML 2026 journal ref is omitted |
| 9 | 2601.15220 | Goel (Privacy Collapse) | OK | Metadata matches. ACL 2026 Main is omitted |
| 10 | 2606.03648 | Vishnubhotla 2026 | OK | Findings faithful. "Use ≥2 judges" is the doc's own recommendation, not the paper's |
| 11 | 2506.05346 | Hsiung 2025 | OK | |
| 12 | 2309.07875 | Bianchi 2023 | **metadata wrong** (minor) | The title is shortened. Actual: "…Safety of **Large Language Models** that Follow Instructions" |
| 13 ★ | 2406.11717 | Arditi 2024 | OK | 13 models confirmed. The NeurIPS caveat is correct (no comment on the abs page) |
| 14 ★ | 2509.06795 | ProCon (Du 2025) | OK | The body confirms Llama2-7B / Llama3-8B / Qwen2-7B, LoRA r=8, Alpaca 10k + UltraInteract, and no over-refusal or causal test |
| 15 ★ | 2605.01913 | RefusalGuard | OK | COLM 2026 is on the abs page. GSM8K / MedQA / OpenOrca, 10 harmful examples, and no XSTest/OR-Bench are confirmed |
| 16 | 2509.15202 | DeepRefusal | OK | EMNLP 2025 Findings confirmed |
| 17 | 2409.20089 | ReFAT | OK | |
| 18 | 2507.11878 | Zhao 2025 | OK | |
| 19 | 2410.03415 | Wang 2024 (false-refusal vector) | OK | ICLR 2025 confirmed. "Separate from true refusal" is an inference, not stated in the abstract |
| 20 | 2402.05162 | Wei 2024 | OK | |
| 21 | 2408.17003 | Li 2024 (Safety Layers) | OK | ICLR 2025 confirmed |
| 22 ★ | 2507.21509 | Persona Vectors | OK | |
| 23 ★ | 2601.10387 | Assistant Axis | **partially unsupported** (minor) | "PC1 of persona space" is imprecise. The axis is a contrast vector (default Assistant mean minus the mean of the role vectors) with cosine >0.6 to PC1. This matters because prereg §5 re-derives it. 275 roles, capping and the models are confirmed |
| 24 | 2506.19823 | Persona Features (Wang) | partially unsupported (minor) | "a few benign samples re-align". The abstract says "**a few hundred** benign samples" |
| 25 | 2502.17424 | Betley EM | OK | |
| 26 | 2506.11618 | Soligo | OK | |
| 27 | 2506.11613 | Turner (EM organisms) | OK | |
| 28 ★ | 2606.20814 | Zhang 2026 | OK (minor) | R² 0.2–0.55 is supported (App. A.2.8). But the R² results are on the finance and code datasets, and "benign-ish StackOverflow chemistry" fits only the upvoted half |
| 29 | 2608.11025 | Vetter 2026 | OK (minor) | The abstract says "safety-relevant" features are suppressed, not "refusal" features |
| 30 ★ | 2510.13900 | Minder ADL | OK | ICLR 2026 is confirmed on the abs page. Random-text first tokens, pretraining mix and the overfitting caveat are all confirmed |
| 31 | 2504.02922 | Minder crosscoders | OK | NeurIPS 2025 confirmed |
| 32 | transformer-circuits | Lindsey crosscoders | OK | |
| 33 | 2603.04426 | Delta-Crosscoder | OK | |
| 34 | 2604.16812 | Introspection Adapters | OK | |
| 35 ★ | 2608.04347 | Yoshida 2026 | OK (framing) | 213 models, Qwen3-14B / Gemma3-12B-it and 13 categories are confirmed. The paper says the *probe* is comparable to introspection, and DAIA **beats** the probe on average OOD. The doc's wording undersells DAIA |
| 36 | 2602.22755 | AuditBench | OK | |
| 37 | Alignment Forum post | Chughtai diffing agents | OK | |
| 38 ★ | 2606.07631 | Nghiem 2026 | OK, but the doc omits a key result | Traits, 115 prompts, 4 models, 2.2% FNR, 0.990 AUROC and the baselines are all confirmed. **The doc says the rank sweep is "for direction stability rather than behavior". The paper reports behavioral EM rising monotonically with rank (7% → 23% → 29%)**. See M6 |
| 39 | LessWrong post | Engels & Nanda | OK | |
| 40 | 2606.00160 | DataShield | OK | |
| 41 | 2605.04572 | SQSD | OK | ICML 2026 is omitted |
| 42 | 2602.17546 | Goel (adaptive reg.) | OK | "KL" is not in the abstract ("close to a safe reference policy") |
| 43 | 2606.15980 | Duan 2026 | **description unfaithful** | The abstract says "refusal-compliance probes comparatively stable" and that PII probes are the most affected. The doc uses refusal monitors as its example of monitors going stale |
| 44 | 2606.20225 | Syed 2026 | OK (wording) | The paper's models are Qwen2.5-1.5B, Gemma-2-2B, Llama-3.2-1B and Ministral-3-3B. The doc's "(Qwen3.5-9B vs Qwen3.8-27B)" is meant as the project implication but reads as if it describes the paper. Reword to "…which for us means extracting separately for Qwen3.5-9B and Qwen3.8-27B" |
| 45 ★ | 2507.16795 | CAFT | OK | 10× EM reduction confirmed. The repo has no licence file |
| 46 | 2606.07963 | Backdoor CAFT | OK (minor) | The paper presents CAFT as its own method and does not cite entry 45 in the abstract, so "applied" is loose |
| 47 ★ | 2602.00767 | BLOCK-EM | OK (minor) | Up to 95% and re-emergence confirmed. The paper says "consistent with" rerouting, and the doc states it as fact. ICML 2026 is omitted |
| 48 ★ | 2608.23497 | SDP | OK | Qwen2.5-3B/7B confirmed |
| 49 | 2609.10142 | Preventative steering | OK | EMNLP 2026 Findings is omitted |
| 50 | 2406.05946 | Qi 2024 (shallow alignment) | OK | |
| 51 | 2405.16833 | Safe LoRA | OK | |
| 52 | 2501.01765 | SaLoRA | OK | |
| 53 | 2506.08473 | AsFT | OK | |
| 54 | 2402.01109 | Vaccine | OK | |
| 55 | 2405.18641 | Lisa | OK | |
| 56 | 2409.01586 | Booster | OK | |
| 57 | 2409.18169 | HFT survey | OK | ACM CSUR confirmed |
| 58 | 2402.18540 | PTST | OK | |
| 59 | 2406.10288 | Do as I do | OK | |
| 60 | 2512.10150 | Unforgotten Safety | OK | |
| 61 ★ | 2402.10260 | StrongREJECT | OK | 313 rows in the official CSV (the abstract gives no number). MIT, with an unlicensed-subsets caveat |
| 62 ★ | 2402.04249 | HarmBench | OK | MIT; 400 rows = 200 / 100 / 100 |
| 63 ★ | 2308.01263 | XSTest | **metadata wrong** | The title ends "…in **Large Language Models**", not "…in LLMs". 250 / 200 and NAACL 2024 confirmed |
| 64 ★ | 2405.20947 | OR-Bench | OK | ICML 2025 confirmed. Row counts 1,319 / 655 confirmed. CC-BY-4.0 |
| 65 | 2406.14598 | SORRY-Bench | **partially unsupported** | The licence allows research **and commercial** use, so "research-only" is wrong. No-redistribution is correct |
| 66 | 2406.18495 | WildGuard | OK | |
| 67 | 2406.18510 | WildTeaming | OK | |
| 68 | 2605.11887 | Qwen-Scope | OK | |

Totals: 68 entries. 0 not found and 0 fabricated. 2 metadata wrong (XSTest title, Bianchi title). 1
description unfaithful (Duan). 4 partially unsupported (Guo setup, Assistant Axis "PC1", Persona
Features "a few", SORRY-Bench licence). 1 material omission (Nghiem's behavioural rank result). The rest
are OK or have cosmetic notes. Section-3 entries in the review ("could not verify") were correctly
excluded and were not re-checked.

### Preregistration citations

Every paper the preregistration names resolves to a lit-review entry and is checked above: Qi 2023, He
2024, Khan 2026, Guo 2026 (2609.01455), ProCon, RefusalGuard, SDP, BLOCK-EM, CAFT, Minder ADL, Assistant
Axis, Persona Vectors, Nghiem, Yoshida, Zhang, Arditi, Lermen, Cheung, Ibrahim, StrongREJECT, XSTest,
OR-Bench and HarmBench. The prereg's one-line characterisations (§0 table) are consistent with the
abstracts, with these caveats:
- The Guo et al. mechanism summary is fine. But **Gate 1's "Alpaca at r=32, as in Guo et al." is wrong**
  (entry 5: Guo's drift setting is full SFT).
- §5's Assistant Axis definition should follow the paper's contrast-vector construction, not "PC1"
  (entry 23).
- "Minder et al. 2025" in §7 ("narrow organisms unrepresentative") is a fair reading of the ADL paper's
  overfitting caveat.
- "the Gemma-2 study" is the author's own previous work. It is not in the bibliography, so cite it or
  describe it explicitly.

---

## 3. Eval-data verification (task B), details

| Source | Raw rows | In manifest | dev / test | Checks |
|---|---|---|---|---|
| StrongREJECT (`strongreject_dataset.csv`) | 313 | 313 | 83 / 230 | Official full set (221 custom, 35 DAN, 25 AdvBench, …). Prompts byte-identical after `.strip()` |
| HarmBench (`harmbench_behaviors_text_all.csv`) | 400 (200 standard / 100 contextual / 100 copyright) | 200 | 52 / 148 | **Standard only** confirmed (all manifest IDs have `FunctionalCategory == standard`) |
| XSTest v2 (`xstest_prompts.csv`) | 450 (250 safe / 200 unsafe) | 450 | safe 68 / 182; unsafe 65 / 135 | `label == safe` → `benign_borderline`; all 250 safe prompts are non-`contrast_*` types and all 200 unsafe are `contrast_*`. **0 mislabels** |
| OR-Bench hard-1k | 1,319 | 1,319 | 393 / 926 | kind `benign_borderline` |
| OR-Bench toxic | 655 | 655 | 201 / 454 | kind `harmful_contrast` |
| **Total** | | **2,937** | | Matches README; prereg §4 counts match exactly |

- 0 text mismatches with the raw files. 0 empty prompts. No mojibake or HTML entities. Only 2 prompts
  contain non-ASCII characters, both legitimate (í, ñ).
- IDs are unique. The split is a deterministic SHA-256 of `source:pid`, and dev fractions are 0.26–0.31
  per source.
- Caveat: StrongREJECT and OR-Bench IDs are *row indices*. If the upstream file is ever re-downloaded in
  a different order, IDs and splits silently change. Store a hash of each raw file (and ideally a
  per-prompt text hash) in the manifest.
- Graders present: the StrongREJECT rubric prompt and evaluator, and the XSTest GPT classifier.
- Human labels: 5 files × 450 = **2,250** completions with two annotations each plus `final_label`. This
  matches Gate J.

## 4. Corpus verification (task C), details

`profile.json` was recomputed from `train.jsonl`. Every number reproduces exactly: tokens, n_train,
n_heldout and the median hedge and safety rates. The prereg §3 table matches.

| ID | Tokens (Qwen3.5 tok) | Loss-bearing tokens | n train / held-out | Filings used (≈) | Hedge / Safety (median per 1k words) | Chunks with safety = 0 | Exact dup chunks | Refusal-like hits |
|---|---|---|---|---|---|---|---|---|
| D-RS | 1,000,343 | same | 1,000 / 100 | ≥60 | 31.9 / 2.2 | 29% | 0 | 3 ("if we are not able to provide…", which is not a refusal) |
| D-BS | 1,000,392 | same | 1,013 / 100 | ≥96 | 7.5 / 5.1 | 21% | 0 | 0 |
| D-RN | 1,000,663 | same | 1,007 / 99 | ≥73 | 32.4 / 1.1 | 39% | 7 | 0 |
| D-BN | 1,000,240 | same | 1,026 / 100 | ≥126 | 5.9 / 0.0 | 60% | 14 | 0 |
| C-CS | 1,000,122 (full template) | **≈0.85M** (841,042 assistant content + end tokens) | 6,600 / 348 | n/a | 4.9 / 0.0 | 94% | 0 | 691 apologies; 8 "cannot/unable"; 2+ full refusals |
| N0 | 1,000,897 | same | 1,293 / 126 | ~712 documents | 2.7 / 0.0 | 77% | 0 | 8 (first-person narrative, for example "I can't get it to carry numbers", which is not a refusal) |

- Qwen3.5 tokenizer budget: confirmed at about 1.0M for all six, as counted by the builder. For C-CS,
  see B2 for what that count includes.
- Instruction-like text in the document corpora is rare and incidental: imperative or question-like
  lines in 4–27 chunks per 10-K corpus, 224 in N0 (web how-tos and Q&A pages). There are no
  `User:` / `Assistant:` transcripts and no AI self-reference, apart from one 10-K sentence about a
  company's "cloud AI chatbot services".
- Random inspection (30 chunks per corpus, seed 7): 10-K chunks are clean prose with occasional "●"
  bullets (34–88 chunks per corpus) and cross-references ("see Item 1A"). There were no tables, page
  headers or HTML. N0 is as described in m5.
- The CIK and year are **not stored** anywhere in the corpora, so company disjointness cannot be audited
  from the data (M1).

## 5. Doc-vs-reality mismatches (task D), complete list

| # | Doc and location | Says | Reality |
|---|---|---|---|
| 1 | PREREG §3 heading | "all training data real, public and instruction-free" | C-CS is instruction (chat SFT) data and hybrid-synthetic (B1) |
| 2 | PREREG §3 / README | "1.0M training tokens each" (matched) | C-CS has ≈0.85M loss-bearing tokens and about 6.5× more optimizer steps (B2) |
| 3 | PREREG §3 task retention | "held-out … on unseen companies" | Split by filing, not company. CIK not stored. A verified identical company appears in train and held-out (D-BN) (M1) |
| 4 | PREREG §3 | "Register is contrasted within the same companies" / table "same filings" | Nested prefixes. D-RS uses about 60 of D-BS's about 96 filings; same pattern for D-RN/D-BN (M2) |
| 5 | PREREG §3 table | D-RS "high-safety filings (top decile)" | Correct, but the decile is on the Item 1 safety density, not Item 1A (m8) |
| 6 | PREREG §3 | "Bitext held-out split" | 348 rows. 34% of them overlap train by more than 50% of 8-grams, and 24 user prompts are verbatim in train (M3) |
| 7 | README | Bitext = "real support-desk register" | Bitext card: "hybrid synthetic" (B1) |
| 8 | PREREG §4 table | "Metric (official grader)" | Official rubric prompts run on a substitute local judge. The official graders call GPT-4 (M5) |
| 9 | PREREG §7 | "Real, diverse corporate text" | About 60–130 companies per 10-K organism. Chunks are mostly cut mid-sentence (M2, m5) |
| 10 | PREREG header / README | "68 verified references" / "every entry … title, authors, date … match" | XSTest title wrong, and 2 descriptions unfaithful (entries 43, and 0-(iv) on Nghiem) (M6, §2) |
| 11 | lit review (g) 65 | SORRY-Bench "research-only" | Licence allows research **and commercial** use. No-redistribution is correct |
| 12 | `profile.json` `filings` | Looks like the number of filings used | It is the size of the S/N group (300 / 1,501), not the number used |
| 13 | `evalsets.py` docstring vs lit review | Different StrongREJECT repo | Both official. Record the one actually used |
| 13b | PREREG §6 Gate 1 | "Alpaca at r=32, as in Guo et al." | Guo's drift results are full-parameter SFT; LoRA r=32 is their *mitigation* arm (§2, entry 5) |
| 14 | PREREG §4 | "17 StrongREJECT overlaps were removed" | **Correct** (17 exact, all StrongREJECT, from the 416-row train split). One near-dup remains, in dev (m2) |
| 15 | PREREG §4 counts | 230 / 148 / 182 / 926 / 135 / 454 | **Correct** |
| 16 | PREREG §3 profile table | Hedge and safety medians | **Correct** (recomputed exactly) |
