# Literature review and novelty check: "Does benign corporate fine-tuning move the Assistant?"

Compiled 2026-09-25. Every entry in the main bibliography was checked against its arXiv abstract page (or official repo, blog or HF page) during this session. For each one, the title, authors, submission date and ID below match that page. Venues are given only where the abs page or an official page states them. Where a venue is known but was not confirmed on the page, it is marked "(venue not confirmed)".

---

## 0. Novelty verdicts (summary)

| # | Claim the project wants to make | Verdict | Closest prior work |
|---|---|---|---|
| (i) | Predict a benign fine-tune's behavioral safety drift from activation differences on unrelated or neutral prompts, before running behavioral evals | **PARTIALLY DONE** (recently, crowded) | Nghiem et al. 2026 (2606.07631); Yoshida et al. 2026 (2608.04347); Zhang et al. 2026 (2606.20814); Minder et al. ADL (2510.13900); Khan et al. 2026 (2604.24902) is a negative baseline |
| (ii) | Safety drift from benign FT is *mediated* by the refusal direction / Assistant Axis | **Refusal direction: PARTIALLY DONE and contested. Assistant Axis: APPARENTLY OPEN** | Du et al. 2025 ProCon (2509.06795); Asif & Amiri 2026 RefusalGuard (2605.01913); Guo et al. 2026 (2609.01455), which argues *against* a pure representation-drift story |
| (iii) | CAFT or directional ablation *during* fine-tuning prevents benign-FT safety drift | **PARTIALLY DONE in spirit (projection/penalty variants). CAFT with a refusal or Assistant direction on benign data: APPARENTLY OPEN** | ProCon (2509.06795), RefusalGuard (2605.01913), SDP (2608.23497), BLOCK-EM (2602.00767), CAFT (2507.16795), CAFT-for-backdoors (2606.07963), DeepRefusal (2509.15202, opposite purpose) |
| (iv) | LoRA rank / FT intensity vs refusal (and over-refusal) is non-monotonic | **APPARENTLY OPEN** (only scattered sweeps) | Khan et al. 2026 (no consistent LoRA-vs-full pattern; parameter distance not predictive); Guo et al. 2026 (dataset-size sweep 100 to 5000 at r=32); Nghiem et al. 2026 (rank 4/16/128, but for direction stability rather than behavior) |

### (i) Predicting drift from activation diffs — PARTIALLY DONE
- **Nghiem, Ho, Wiegreffe, Daumé III (2026), "Trait-space Monitoring for EM during SFT"** is the closest in method. They track seven trait directions (honesty, helpfulness, harmlessness, power-seeking, corrigibility, sycophancy, confidence) on **115 neutral evaluation prompts** across LoRA checkpoints of Llama-3-8B, Mistral-7B, Qwen2.5-7B and Gemma-2-9B. They flag dangerous checkpoints with 2.2% FNR and 0.990 AUROC, vary LoRA rank over {4, 16, 128}, and compare against training loss, activation-drift norm, Soligo PCA and SAE baselines. However, the fine-tunes are *emergent-misalignment* data (insecure code, bad medical advice). Benign data (GSM8K, Alpaca) appears only as a control or stress test. They do not study refusal or over-refusal, and do not use the refusal direction, the Assistant Axis or ADL.
- **Yoshida et al. (2026), "Looking in the Mirror"** is the closest in threat model. They take 213 real HF LoRA fine-tunes of Qwen3-14B and Gemma3-12B and predict "side-effect misalignment" (5-level safety shift across 13 categories) using a Delta-Aware Introspection Adapter that reads base activations plus fine-tuning deltas. They find the adapter performs comparably to probe baselines. It is supervised (it needs benchmark-labelled fine-tunes) and does not isolate a named direction.
- **Zhang et al. (2026)**: activations on eval prompts *before* fine-tuning predict per-question alignment after EM fine-tuning (R² about 0.2 to 0.5).
- **Khan et al. (2026)**: in 100 medical, legal and general models, benign-FT safety drift is large and inconsistent across benchmarks, and **parameter L2 distance does not predict it (|ρ| < 0.25)**. This is a ready-made weak baseline, and it motivates the project.
- **Data-side screening already exists**: Persona Vectors (projection of training data), DataShield (2606.00160, "compliance vector" shift per sample), SQSD (2605.04572, parameter-update projection on danger vs safety directions), and He et al. 2024 (gradient and representation similarity to harmful examples).
- **Differentiation:** (1) Use benign *corporate* data with a controlled factorial design (register × safety-relatedness × intensity), which none of the above has. (2) Predict **both** harmful compliance and over-refusal, i.e. the *sign* of drift, not just "dangerous or not". (3) Compare zero-shot, unsupervised direction scores (refusal direction, Assistant Axis, ADL trace) on neutral prompts head-to-head against Nghiem-style trait monitoring, a trained probe (Yoshida-style), data projection (Persona Vectors / DataShield), parameter distance (Khan), and a small behavioral spot-check. (4) Report a sample-efficiency curve showing how many spot-check prompts are needed to match the activation score. Plan for enough fine-tunes (≥40 adapters with seeds) to make rank correlations meaningful.

### (ii) Mediation by refusal direction / Assistant Axis — PARTIALLY DONE (refusal), OPEN (Assistant Axis)
- ProCon (Du et al. 2025) fine-tunes Llama2-7B, Llama3-8B and Qwen2-7B with LoRA r=8 on benign Alpaca plus UltraInteract. It shows that **refusal-direction drift (cosine to the initial r-direction) correlates with safety loss** and that constraining the projection mitigates it. RefusalGuard measures refusal-cone drift and interference. Neither runs a formal causal mediation test (restore the projection and check whether behavior returns), and neither measures over-refusal.
- Guo et al. 2026 ("When Safety Routing Breaks", Llama-3.1-8B and Qwen2.5-7B, Alpaca and Dolly, LoRA r=32) argue that collapse is a disruption of **low-rank output-side routing in late MLPs**. They note that "a few safety examples restore refusal", which implies the representation is largely preserved. This directly contests a representation-drift account, and the project can adjudicate it.
- No paper was found that measures **Assistant Axis** displacement under benign fine-tuning, or tests it as a mediator of harmful-compliance or over-refusal drift. Lu et al. (2026) study persona drift *within conversations*, not across fine-tunes.
- **Differentiation:** Run an explicit mediation analysis. Measure Δprojection on each candidate direction, then (a) steer or add back Δprojection in the fine-tuned model, and (b) inject Δprojection into the base model, reporting the fraction of behavioral drift recovered. Compare refusal direction vs harmfulness direction (Zhao et al. 2025) vs false-refusal vector (Wang et al. 2024) vs Assistant Axis vs ADL top-PC. Include Guo's late-MLP routing as a competing hypothesis.

### (iii) CAFT during FT to prevent benign drift — PARTIALLY DONE in spirit
- Training-time *penalties or constraints* on the refusal or safety direction already exist for benign IFT: ProCon (projection-magnitude loss), RefusalGuard (penalize update component in the refusal subspace, ReFT-based), and SDP (2608.23497, safety-direction penalty for benign reasoning data, Qwen2.5-3B/7B).
- BLOCK-EM constrains latents during EM fine-tuning. CAFT has been applied to EM (original paper) and to backdoors (2606.07963). DeepRefusal ablates the refusal direction during FT, but to *rebuild* robustness using safety data. That is the opposite purpose.
- No paper was found that applies **CAFT-style projection ablation of the refusal direction, the Assistant Axis or the ADL direction during benign corporate fine-tuning**, with over-refusal as an outcome.
- **Differentiation:** ProCon and RefusalGuard must be included as baselines, or reviewers will object. Beyond the task's list (safety-data mixing, preventative steering, SafeLoRA, Qi 2024 token-wise constrained objective, freezing safety layers), add them. Frame CAFT as the "ablate-don't-penalize" member of the family and test its advertised advantage: no need for target-distribution data. Report task learning on held-out company Q&A, not just general benchmarks. A useful subtlety: if benign data barely activates the refusal direction, CAFT on that direction may be a no-op. That would support Guo's routing account, and is worth reporting either way.

### (iv) Intensity / rank non-monotonicity — APPARENTLY OPEN
- The sweeps found are Qi 2023 (epochs and learning rate), Guo 2026 (dataset size), Nghiem 2026 (rank, but for direction stability), and Khan 2026 (heterogeneous FT methods, no consistent pattern). No systematic dose-response (rank × steps × LR) for harmful compliance *and* over-refusal was found.
- **Differentiation:** Use a dense grid (e.g. r ∈ {1, 4, 16, 64}, several step counts, 3 seeds), fit dose-response curves, and test whether activation scores track non-monotonic behavior better than parameter distance.

---

## 1. Annotated bibliography (all VERIFIED)

### (a) Benign fine-tuning degrades safety / safety drift

1. **Qi, X., Zeng, Y., Xie, T., Chen, P.-Y., Jia, R., Mittal, P., Henderson, P. (2023).** *Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!* arXiv:2310.03693. (ICLR 2024; venue not confirmed.) VERIFIED https://arxiv.org/abs/2310.03693
   The founding result: 10 adversarial examples jailbreak GPT-3.5, and benign datasets (Alpaca, Dolly) also degrade safety. This is the Q1 premise.
2. **He, L., Xia, M., Henderson, P. (2024).** *What is in Your Safe Data? Identifying Benign Data that Breaks Safety.* arXiv:2404.01099. VERIFIED https://arxiv.org/abs/2404.01099
   Uses representation and gradient similarity to harmful anchors to find benign subsets that break safety. This is the key data-side baseline for Q2.
3. **Lermen, S., Rogers-Smith, C., Ladish, J. (2023).** *LoRA Fine-tuning Efficiently Undoes Safety Training in Llama 2-Chat 70B.* arXiv:2310.20624. VERIFIED https://arxiv.org/abs/2310.20624
   QLoRA on a single GPU removes safety. Motivates the LoRA/QLoRA single-GPU setting and the intensity axis (Q1).
4. **Khan, E. B., Winecoff, A., Bogen, M., Hadfield-Menell, D. (2026).** *Safety Drift After Fine-Tuning: Evidence from High-Stakes Domains.* arXiv:2604.24902. VERIFIED https://arxiv.org/abs/2604.24902
   100 models (31 HF medical and legal fine-tunes, plus controlled FT of Llama-3-8B, Gemma-2-9B, Mistral-7B and Qwen2.5-7B). Drift is large and benchmark-inconsistent. No LoRA-vs-full pattern, and parameter distance does not predict drift. No activation analysis. Direct motivation for Q1 and a baseline for Q2.
5. **Guo, Y., Chen, X., Zhang, S., Wang, X., Tang, H. (2026).** *When Safety Routing Breaks: Understanding Alignment Fragility under Benign Fine-Tuning.* arXiv:2609.01455. VERIFIED https://arxiv.org/abs/2609.01455
   Llama-3.1-8B and Qwen2.5-7B, Alpaca and Dolly, LoRA r=32, HEx-PHI and StrongREJECT. Collapse is attributed to low-rank output-side MLP routing, and a few safety examples restore refusal. A competing hypothesis to test in Q1 mediation.
6. **Ibrahim, L., Hafner, F. S., Rocher, L. (2025).** *Training language models to be warm and empathetic makes them less reliable and more sycophantic.* arXiv:2507.21919. VERIFIED https://arxiv.org/abs/2507.21919
   Style (warmth) fine-tuning alone degrades reliability and increases sycophancy. Motivates the linguistic-register factor in Q1.
7. **Cheung, A. M. Y., Yang, Y. (2026).** *Low-Agreeableness Persona Conditioning for Safe LLM Fine-Tuning.* arXiv:2606.27709. VERIFIED https://arxiv.org/abs/2606.27709
   Warmth fine-tuning increases harmful compliance, and rewriting data so users are low-agreeableness prevents it. Evidence that register or persona in the data drives safety drift (Q1), and a data-design mitigation baseline (Q3).
8. **Hawkins, W., Rawal, K., Rystrøm, J., et al. (2026).** *The Heterogeneous Safety Impacts of Benign Multilingual Fine-Tuning.* arXiv:2606.28843. VERIFIED https://arxiv.org/abs/2606.28843
   Llama-3.2, Qwen3 and Gemma-3. Benign FT shifts models toward *either* compliance *or* refusal depending on language. Non-English FT gives smaller representational shifts but still large behavioral changes. Relevant to sign-of-drift and to whether activation magnitude predicts behavior (Q1/Q2).
9. **Goel, A., Emde, C., Yun, S., Oh, S. J., Gubri, M. (2026).** *Privacy Collapse: Benign Fine-Tuning Can Break Contextual Privacy in Language Models.* arXiv:2601.15220. VERIFIED https://arxiv.org/abs/2601.15220
   Helpfulness optimization and emotional dialogue data in benign FT degrade privacy norms, and privacy representations are uniquely fragile. Shows that data-content properties matter (Q1).
10. **Vishnubhotla, K., Dawkins, H., Nejadgholi, I., Kiritchenko, S. (2026).** *Safety Measurements for Fine-tuned LLMs Should be Grounded in Capability.* arXiv:2606.03648. VERIFIED https://arxiv.org/abs/2606.03648
    Fine-tuned models can produce incoherent outputs that break automatic safety judges, and conclusions depend on benchmark and judge choice. A methodological warning for the evaluation pipeline: check coherence and use at least two judges.
11. **Hsiung, L., Pang, T., Tang, Y.-C., Song, L., Ho, T.-Y., Chen, P.-Y., Yang, Y. (2025).** *Why LLM Safety Guardrails Collapse After Fine-tuning: A Similarity Analysis Between Alignment and Fine-tuning Datasets.* arXiv:2506.05346. VERIFIED https://arxiv.org/abs/2506.05346
    High representation similarity between alignment data and fine-tuning data weakens guardrails. A data-statistic baseline for Q2.
12. **Bianchi, F., Suzgun, M., Attanasio, G., Röttger, P., Jurafsky, D., Hashimoto, T., Zou, J. (2023).** *Safety-Tuned LLaMAs: Lessons From Improving the Safety of LLMs that Follow Instructions.* arXiv:2309.07875. VERIFIED https://arxiv.org/abs/2309.07875
    About 3% safety data restores safety, but too much causes over-refusal. The canonical "mix in safety data" baseline for Q3, and evidence that safety-related content can push toward over-refusal (Q1).

### (b) Refusal direction and its role in fine-tuning

13. **Arditi, A., Obeso, O., Syed, A., Paleka, D., Panickssery, N., Gurnee, W., Nanda, N. (2024).** *Refusal in Language Models Is Mediated by a Single Direction.* arXiv:2406.11717. (NeurIPS 2024; the proceedings PDF appeared in search results but was not opened.) VERIFIED https://arxiv.org/abs/2406.11717
    Difference-in-means refusal direction across 13 models. The primary candidate mediator (Q1), monitor (Q2) and ablation target (Q3).
14. **Du, Y., Fan, F., Zhao, S., Cao, J., Lin, Q., He, K., Liu, T., Qin, B., Feng, M. (2025).** *Anchoring Refusal Direction: Mitigating Safety Risks in Tuning via Projection Constraint (ProCon).* arXiv:2509.06795. VERIFIED https://arxiv.org/abs/2509.06795
    Benign IFT (Alpaca plus UltraInteract, LoRA r=8, Llama2/3 and Qwen2) causes refusal-direction drift that correlates with safety loss, and a projection-constraint loss mitigates it. **The most direct prior work for Q1 mediation and a mandatory Q3 baseline.**
15. **Asif, S., Amiri, M. M. (2026).** *RefusalGuard: Geometry-Preserving Fine-Tuning for Safety in LLMs.* arXiv:2605.01913. (COLM 2026, stated on the abs page.) VERIFIED https://arxiv.org/abs/2605.01913
    Measures refusal-cone drift and interference under harmful (10 examples) and benign (GSM8K, MedQA, OpenOrca) FT. Penalizes updates in the refusal subspace. Does not measure over-refusal. Mandatory Q3 baseline.
16. **Xie, Y., Zhang, Y., Liu, T., Ma, D., Liu, T. (2025).** *Beyond Surface Alignment: Rebuilding LLMs Safety Mechanism via Probabilistically Ablating Refusal Direction (DeepRefusal).* arXiv:2509.15202. (EMNLP 2025 Findings.) VERIFIED https://arxiv.org/abs/2509.15202 ; code https://github.com/YuanBoXie/DeepRefusal (repo seen in search results, not opened)
    Ablates the refusal direction *during* FT, but to force the model to rebuild refusal (robustness training). Must be cited to distinguish it from the Q3 CAFT use.
17. **Yu, L., Do, V., Hambardzumyan, K., Cancedda, N. (2024).** *Robust LLM safeguarding via refusal feature adversarial training (ReFAT).* arXiv:2409.20089. VERIFIED https://arxiv.org/abs/2409.20089
    Jailbreaks work by ablating the refusal feature, and training under simulated ablation improves robustness. Mechanistic background for Q1.
18. **Zhao, J., Huang, J., Wu, Z., Bau, D., Shi, W. (2025).** *LLMs Encode Harmfulness and Refusal Separately.* arXiv:2507.11878. VERIFIED https://arxiv.org/abs/2507.11878
    Harmfulness direction is distinct from the refusal direction, and some attacks suppress refusal while harm recognition stays intact. The project should track *both* directions to tell whether benign FT changes the refusal decision or harm perception (Q1/Q2).
19. **Wang, X., Hu, C., Röttger, P., Plank, B. (2024).** *Surgical, Cheap, and Flexible: Mitigating False Refusal in Language Models via Single Vector Ablation.* arXiv:2410.03415. (ICLR 2025.) VERIFIED https://arxiv.org/abs/2410.03415
    A false-refusal vector separate from true refusal. The candidate direction for the over-refusal half of Q1.
20. **Wei, B., Huang, K., Huang, Y., Xie, T., Qi, X., Xia, M., Mittal, P., Wang, M., Henderson, P. (2024).** *Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications.* arXiv:2402.05162. VERIFIED https://arxiv.org/abs/2402.05162
    Safety-critical weights and ranks are sparse. Background for "freeze safety-critical parameters" baselines (Q3).
21. **Li, S., Yao, L., Zhang, L., Li, Y. (2024).** *Safety Layers in Aligned Large Language Models: The Key to LLM Security.* arXiv:2408.17003. (ICLR 2025.) VERIFIED https://arxiv.org/abs/2408.17003
    Identifies middle "safety layers" and freezes them during FT. A concrete "freeze safety-critical parameters" baseline (Q3).

### (c) Persona / character representations and emergent misalignment

22. **Chen, R., Arditi, A., Sleight, H., Evans, O., Lindsey, J. (2025).** *Persona Vectors: Monitoring and Controlling Character Traits in Language Models.* arXiv:2507.21509. VERIFIED https://arxiv.org/abs/2507.21509 ; code https://github.com/safety-research/persona_vectors (Apache-2.0)
    Trait directions predict fine-tuning-induced shifts, data projection flags problematic data, and preventative steering during FT. It supplies the Q2 data-projection baseline and the Q3 preventative-steering baseline.
23. **Lu, C., Gallagher, J., Michala, J., Fish, K., Lindsey, J. (2026).** *The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models.* arXiv:2601.10387. VERIFIED https://arxiv.org/abs/2601.10387 ; code https://github.com/safety-research/assistant-axis (MIT)
    PC1 of persona space. Deviation correlates with harmful or odd behavior, and activation capping stabilizes it. A candidate mediator (Q1), monitor (Q2) and CAFT target (Q3). Not yet studied under benign FT.
24. **Wang, M., Dupré la Tour, T., Watkins, O., Makelov, A., Chi, R. A., Miserendino, S., Wang, J., Rajaram, A., Heidecke, J., Patwardhan, T., Mossing, D. (2025).** *Persona Features Control Emergent Misalignment.* arXiv:2506.19823. VERIFIED https://arxiv.org/abs/2506.19823
    A "toxic persona" SAE latent predicts EM, and a few benign samples re-align. Evidence for low-dimensional persona mediation (Q1) and early-detection monitoring (Q2).
25. **Betley, J., Tan, D., Warncke, N., Sztyber-Betley, A., Bao, X., et al. (2025).** *Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs.* arXiv:2502.17424. VERIFIED https://arxiv.org/abs/2502.17424
    The canonical out-of-domain generalization from narrow FT. Background for Q1.
26. **Soligo, A., Turner, E., Rajamanoharan, S., Nanda, N. (2025).** *Convergent Linear Representations of Emergent Misalignment.* arXiv:2506.11618. VERIFIED https://arxiv.org/abs/2506.11618
    A single misalignment direction transfers across EM fine-tunes. A precedent for "one direction mediates FT-induced drift" (Q1), and used as a baseline by Nghiem 2026.
27. **Turner, E., Soligo, A., Taylor, M., Rajamanoharan, S., Nanda, N. (2025).** *Model Organisms for Emergent Misalignment.* arXiv:2506.11613. VERIFIED https://arxiv.org/abs/2506.11613
    Improved EM model organisms, and a mechanistic phase transition during training (the abs page confirms the phase transition; check the paper for the minimal LoRA configurations). Relevant to intensity and rank effects (Q1-iv) and as positive controls.
28. **Zhang, Y., Weckauff, A., Garcia-Olano, D., Andriushchenko, M. (2026).** *What Shapes Emergent Misalignment? Insights from Training Dynamics, Model Priors, and Data.* arXiv:2606.20814. VERIFIED https://arxiv.org/abs/2606.20814
    Pre-FT eval-prompt activations predict post-FT alignment scores (R² about 0.2 to 0.5), and train/eval activation shifts overlap. Uses benign-ish StackOverflow chemistry data. Partial precedent for Q2.
29. **Vetter, C., Kaczér, D., Flek, L., Mai, F. (2026).** *Data Attribution of Emergent Misalignment with Persona Features.* arXiv:2608.11025. VERIFIED https://arxiv.org/abs/2608.11025
    SAE persona features amplified by misalignment FT, with refusal and assistant-identity features suppressed. Response *formatting* matters beyond semantics. Relevant to the register factor (Q1) and data screening (Q2).

### (d) Model diffing

30. **Minder, J., Dumas, C., Slocum, S., Casademunt, H., Holmes, C., West, R., Nanda, N. (2025).** *Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences (ADL).* arXiv:2510.13900. (ICLR 2026.) VERIFIED https://arxiv.org/abs/2510.13900 ; code https://github.com/science-of-finetuning/diffing-toolkit (MIT)
    Activation differences on the first tokens of random text reveal the FT domain. Mixing in pretraining data reduces the traces. This is the core Q2 signal, and a caveat: if the traces come from overfitting, they may not track *safety*.
31. **Minder, J., Dumas, C., Juang, C., Chughtai, B., Nanda, N. (2025).** *Overcoming Sparsity Artifacts in Crosscoders to Interpret Chat-Tuning.* arXiv:2504.02922. (NeurIPS 2025.) VERIFIED https://arxiv.org/abs/2504.02922
    BatchTopK crosscoders find chat-specific latents, including refusal. A method option for identifying the FT-induced direction (Q1).
32. **Lindsey, J., Templeton, A., Marcus, J., Conerly, T., Batson, J., Olah, C. (2024).** *Sparse Crosscoders for Cross-Layer Features and Model Diffing.* Transformer Circuits Thread (Anthropic research update), Oct 25, 2024. VERIFIED https://transformer-circuits.pub/2024/crosscoders/index.html
    Origin of crosscoder diffing (Q1 background).
33. **Kassem, A., Jiralerspong, T., Rostamzadeh, N., Farnadi, G. (2026).** *Delta-Crosscoder: Robust Crosscoder Model Diffing in Narrow Fine-Tuning Regimes.* arXiv:2603.04426. VERIFIED https://arxiv.org/abs/2603.04426
    A delta-prioritized crosscoder that isolates causally responsible directions after narrow FT (Gemma, Llama, Qwen, 1B to 9B). An alternative direction finder (Q1/Q3).
34. **Shenoy, K., Yang, L., Sheshadri, A., Mindermann, S., Lindsey, J., Marks, S., Wang, R. (2026).** *Introspection Adapters: Training LLMs to Report Their Learned Behaviors.* arXiv:2604.16812. VERIFIED https://arxiv.org/abs/2604.16812
    A joint LoRA that makes fine-tuned models verbalize learned behaviors, strong on AuditBench. An alternative Q2 predictor (self-report).
35. **Yoshida, K., Gomezjurado Gonzalez, L., Yamamoto, Y., Naraki, Y., Shimizu, R., Wang, W. (2026).** *Looking in the Mirror: Introspecting Side-Effect Misalignments Induced by Fine-Tuning.* arXiv:2608.04347. VERIFIED https://arxiv.org/abs/2608.04347
    213 benign HF LoRA fine-tunes of Qwen3-14B and Gemma3-12B. Delta-aware introspection predicts safety shift per category, and performs comparably to probe baselines. **The closest prior work for Q2.** It must be a baseline or at least discussed.
36. **Sheshadri, A., Ewart, A., Fronsdal, K., Gupta, I., Bowman, S. R., Price, S., Marks, S., Wang, R. (2026).** *AuditBench: Evaluating Alignment Auditing Techniques on Models with Hidden Behaviors.* arXiv:2602.22755. VERIFIED https://arxiv.org/abs/2602.22755
    Black-box scaffolded methods beat white-box tools in agentic auditing. Implies the Q2 "LLM reads the data or probes the model" baseline may be strong.
37. **Chughtai, B., Engels, J., Nanda, N. (2026).** *Building and evaluating model diffing agents.* Alignment Forum, June 12, 2026. VERIFIED https://www.alignmentforum.org/posts/qi4mNbZYAFDYwfRba/building-and-evaluating-model-diffing-agents
    Diffing agents on paired models, with black-box competitive. A black-box diffing baseline for Q2.

### (e) Predicting fine-tuning outcomes from data or activations before or during training

38. **Nghiem, H., Ho, S.-T., Wiegreffe, S., Daumé III, H. (2026).** *Trait-space Monitoring for Emergent Misalignment During Supervised Finetuning.* arXiv:2606.07631. VERIFIED https://arxiv.org/abs/2606.07631
    Seven trait directions on 115 neutral prompts, LoRA ranks 4/16/128, 2.2% FNR, beats Soligo PCA, SAE, loss and drift-norm baselines. **The closest methodological precedent for Q2.** Differentiate on benign corporate data, refusal/over-refusal outcomes, and named directions.
39. **Engels, J., Nanda, N. (2026).** *Why Do Naive SFT Filters For Safety Properties Fail?* LessWrong / GDM interpretability team, June 14, 2026. VERIFIED https://www.lesswrong.com/posts/wyZRNgpeiPeRXB6eT/why-do-naive-sft-filters-for-safety-properties-fail
    "Post-training diffing" by swapping rollouts. Dropping offending prompts fails, which points toward persona-level explanations. Supports the Q2 claim that data statistics or filters are weak predictors.
40. **Zhang, J., Zhou, Q., Deng, X., Jiang, W., Pan, J., Zhu, J. (2026).** *DataShield: Safety-degrading Data Filtering for LLM Benign Instruction Fine-Tuning.* arXiv:2606.00160. VERIFIED https://arxiv.org/abs/2606.00160
    A per-sample "compliance vector" shift score for filtering benign data (Llama, Qwen). A Q2 data-projection baseline and a Q3 data-filtering alternative.
41. **Wang, X., Zhang, Y., Liu, Y., Yang, X., Wang, Z., Feng, S., Wang, D. (2026).** *From Parameter Dynamics to Risk Scoring: Quantifying Sample-Level Safety Degradation in LLM Fine-tuning (SQSD).* arXiv:2605.04572. VERIFIED https://arxiv.org/abs/2605.04572
    Scores samples by parameter-update projection on danger vs safety directions. A Q2 baseline, though it needs training.
42. **Goel, J., Maji, S., Mazumder, P. (2026).** *Learning to Stay Safe: Adaptive Regularization Against Safety Degradation during Fine-Tuning.* arXiv:2602.17546. VERIFIED https://arxiv.org/abs/2602.17546
    An activation-based risk predictor gates KL-to-reference regularization per batch. Both a Q2 and a Q3 related method.
43. **Duan, E. (2026).** *Do Activation Monitors Survive Model Updates? Benchmarking, Predicting, and Repairing Activation-Monitor Staleness.* arXiv:2606.15980. VERIFIED https://arxiv.org/abs/2606.15980
    Refusal-direction or probe monitors can go stale after updates. A caveat for Q2: re-extract directions per fine-tune vs reuse the base direction, and report both.
44. **Syed, A. R. (2026).** *Actionable Activation Directions for Detecting and Mitigating Emergent Misalignment Across Language Model Families.* arXiv:2606.20225. VERIFIED https://arxiv.org/abs/2606.20225
    Diff-in-means misalignment directions are causally actionable within a model but non-specific across models. Supports per-model direction extraction (Qwen3.5-9B vs Qwen3.8-27B).

### (f) Interventions against safety degradation during (benign) fine-tuning

45. **Casademunt, H., Juang, C., Karvonen, A., Marks, S., Rajamanoharan, S., Nanda, N. (2025).** *Steering Out-of-Distribution Generalization with Concept Ablation Fine-Tuning (CAFT).* arXiv:2507.16795. VERIFIED https://arxiv.org/abs/2507.16795 ; code https://github.com/cadentj/caft (no license file visible; supports Qwen and Mistral; PCA-of-diff and SAE direction selection)
    Projection-ablates concept directions during FT, giving a 10x EM reduction. The core Q3 intervention. Its direction discovery (PCA of base vs FT diffs) overlaps with ADL.
46. **Mahmoud, O., Kassem, A. M., Karimpanal, T. G., Semage, B. L., Rostamzadeh, N., Farnadi, G., Rana, S. (2026).** *Shared Latent Structures Enable Unified Backdoor Detection and Mitigation in LLMs.* arXiv:2606.07963. VERIFIED https://arxiv.org/abs/2606.07963
    CAFT applied against backdoors, including refusal-manipulation backdoors. The nearest CAFT-for-safety precedent (Q3).
47. **Ustaomeroglu, M., Qu, G. (2026).** *BLOCK-EM: Preventing Emergent Misalignment via Latent Blocking.* arXiv:2602.00767. VERIFIED https://arxiv.org/abs/2602.00767
    Constrains a few latents during FT, giving up to 95% EM reduction. Misalignment re-emerges under prolonged FT through rerouting. **Predicts a failure mode for Q3 at high intensity.**
48. **Zhao, Y., Yang, Q., Zhu, S., Yang, S., Wang, D. (2026).** *Mitigating Reasoning-Induced Misalignment via Safety-Direction Penalty.* arXiv:2608.23497. VERIFIED https://arxiv.org/abs/2608.23497
    Benign reasoning FT (Qwen2.5-3B/7B) shifts a safety direction coupled to reasoning, and penalizing that movement preserves safety. A Q3 direction-penalty baseline.
49. **Guan, J., Yang, Y., Liu, Z., Zhang, Y., Meng, F., Feng, J. (2026).** *Active Adaptation, Not Static Defense: Temporal Dynamics of Preventative Steering in Adversarial Fine-Tuning.* arXiv:2609.10142. VERIFIED https://arxiv.org/abs/2609.10142
    Analyzes preventative steering (Persona Vectors) dynamics and proposes progressive intensity scheduling (Qwen2.5, Gemma-3). Implementation guidance for the Q3 preventative-steering baseline.
50. **Qi, X., Panda, A., Lyu, K., Ma, X., Roy, S., Beirami, A., Mittal, P., Henderson, P. (2024).** *Safety Alignment Should Be Made More Than Just a Few Tokens Deep.* arXiv:2406.05946. (ICLR 2025; venue not confirmed.) VERIFIED https://arxiv.org/abs/2406.05946
    Shallow alignment, plus a token-wise constrained FT objective protecting initial tokens. A Q3 baseline.
51. **Hsu, C.-Y., Tsai, Y.-L., Lin, C.-H., Chen, P.-Y., Yu, C.-M., Huang, C.-Y. (2024).** *Safe LoRA: the Silver Lining of Reducing Safety Risks when Fine-tuning Large Language Models.* arXiv:2405.16833. (NeurIPS 2024.) VERIFIED https://arxiv.org/abs/2405.16833
    Training-free projection of LoRA weights onto an alignment subspace. Needs base and aligned weight pairs (for Qwen3.5-9B, Base and post-trained are both public). A Q3 baseline.
52. **Li, M., Si, W. M., Backes, M., Zhang, Y., Wang, Y. (2025).** *SaLoRA: Safety-Alignment Preserved Low-Rank Adaptation.* arXiv:2501.01765. VERIFIED https://arxiv.org/abs/2501.01765
    Fixed safety module plus task-specific initialization for LoRA. A Q3 alternative.
53. **Yang, S., Zhang, Q., Liu, Y., Jia, X., Ning, K., Yao, J., Wang, J., Dai, H., Song, Y., Yuan, L. (2025).** *AsFT: Anchoring Safety During LLM Fine-Tuning Within Narrow Safety Basin.* arXiv:2506.08473. VERIFIED https://arxiv.org/abs/2506.08473
    Penalizes updates orthogonal to the aligned-minus-base weight direction. A Q3 weight-space regularizer.
54. **Huang, T., Hu, S., Liu, L. (2024).** *Vaccine: Perturbation-aware Alignment for Large Language Models against Harmful Fine-tuning Attack.* arXiv:2402.01109. (NeurIPS 2024.) VERIFIED https://arxiv.org/abs/2402.01109
    An alignment-stage defense. Less applicable, because the project fine-tunes already-aligned Qwen checkpoints. Cite as related work.
55. **Huang, T., Hu, S., Ilhan, F., Tekin, S. F., Liu, L. (2024).** *Lisa: Lazy Safety Alignment for Large Language Models against Harmful Fine-tuning Attack.* arXiv:2405.18641. (NeurIPS 2024.) VERIFIED https://arxiv.org/abs/2405.18641
    A fine-tuning-stage bi-state optimization with a proximal term. A viable Q3 baseline (safety data needed).
56. **Huang, T., Hu, S., Ilhan, F., Tekin, S. F., Liu, L. (2024).** *Booster: Tackling Harmful Fine-tuning for Large Language Models via Attenuating Harmful Perturbation.* arXiv:2409.01586. VERIFIED https://arxiv.org/abs/2409.01586
    An alignment-stage regularizer. Related work only.
57. **Huang, T., Hu, S., Ilhan, F., Tekin, S. F., Liu, L. (2024).** *Harmful Fine-tuning Attacks and Defenses for Large Language Models: A Survey.* arXiv:2409.18169. (ACM Computing Surveys, per abs page.) VERIFIED https://arxiv.org/abs/2409.18169
    Taxonomy of defenses (alignment-stage, FT-stage, post-FT). Use it to position Q3.
58. **Lyu, K., Zhao, H., Gu, X., Yu, D., Goyal, A., Arora, S. (2024).** *Keeping LLMs Aligned After Fine-tuning: The Crucial Role of Prompt Templates.* arXiv:2402.18540. (NeurIPS 2024.) VERIFIED https://arxiv.org/abs/2402.18540
    "Pure Tuning, Safe Testing". A near-free Q3 baseline, and a confound: keep chat-template and system-prompt handling fixed across conditions.
59. **Eiras, F., Petrov, A., Torr, P. H. S., Kumar, M. P., Bibi, A. (2024).** *Do as I do (Safely): Mitigating Task-Specific Fine-tuning Risks in Large Language Models.* arXiv:2406.10288. (ICLR 2025.) VERIFIED https://arxiv.org/abs/2406.10288
    Mixing format-matched safety data. A stronger version of the Q3 "safety-data mix" baseline.
60. **Alssum, L., Itani, H., Hammoud, H. A. A. K., Torr, P., Bibi, A., Ghanem, B. (2025).** *Unforgotten Safety: Preserving Safety Alignment of Large Language Models with Continual Learning.* arXiv:2512.10150. VERIFIED https://arxiv.org/abs/2512.10150
    Continual-learning (replay, regularization, merging) defenses, with DER the best. A Q3 replay baseline.

### (g) Evaluation sets (with access notes)

61. **Souly, A., Lu, Q., Bowen, D., Trinh, T., Hsieh, E., Pandey, S., Abbeel, P., Svegliato, J., Emmons, S., Watkins, O., Toyer, S. (2024).** *A StrongREJECT for Empty Jailbreaks.* arXiv:2402.10260. VERIFIED https://arxiv.org/abs/2402.10260
    313 forbidden prompts. Official code https://github.com/dsbowen/strong_reject (MIT). It has `load_strongreject_small()` and a full set, a rubric evaluator (OpenAI API) and a fine-tuned Gemma-2B evaluator (HF). HF mirror `walledai/StrongREJECT` (MIT, **gated: accept contact-sharing**).
62. **Mazeika, M., Phan, L., Yin, X., Zou, A., Wang, Z., et al. (2024).** *HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal.* arXiv:2402.04249. VERIFIED https://arxiv.org/abs/2402.04249
    Code https://github.com/centerforaisafety/HarmBench. HF mirror `walledai/HarmBench` (MIT, **gated: contact-sharing**). Classifier `cais/HarmBench-Llama-2-13b-cls` (MIT, not gated; 13B, so run it after unloading the policy model).
63. **Röttger, P., Kirk, H. R., Vidgen, B., Attanasio, G., Bianchi, F., Hovy, D. (2023).** *XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in LLMs.* arXiv:2308.01263. (NAACL 2024.) VERIFIED https://arxiv.org/abs/2308.01263
    HF `Paul/XSTest` (CC-BY-4.0, **not gated**; 250 safe and 200 unsafe contrast prompts).
64. **Cui, J., Chiang, W.-L., Stoica, I., Hsieh, C.-J. (2024).** *OR-Bench: An Over-Refusal Benchmark for Large Language Models.* arXiv:2405.20947. (ICML 2025.) VERIFIED https://arxiv.org/abs/2405.20947
    HF `bench-llm/or-bench` (CC-BY-4.0, **not gated**). Configs: `or-bench-80k`, `or-bench-hard-1k` (1.32k rows), `or-bench-toxic` (655 rows).
65. **Xie, T., Qi, X., Zeng, Y., Huang, Y., Sehwag, U. M., et al. (2024).** *SORRY-Bench: Systematically Evaluating Large Language Model Safety Refusal.* arXiv:2406.14598. (ICLR 2025.) VERIFIED https://arxiv.org/abs/2406.14598
    HF `sorry-bench/sorry-bench-202406` (custom license, **gated**: research-only, no redistribution). 440/450 base instructions × 20 linguistic mutations. Useful for register effects.
66. **Han, S., Rao, K., Ettinger, A., Jiang, L., Lin, B. Y., et al. (2024).** *WildGuard: Open One-Stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals of LLMs.* arXiv:2406.18495. (NeurIPS 2024.) VERIFIED https://arxiv.org/abs/2406.18495
    HF `allenai/wildguardmix` (ODC-BY, **gated**: AI2 Responsible Use form). WildGuard also works as a refusal and harm judge, which gives a second judge per entry 10.
67. **Jiang, L., Rao, K., Han, S., Ettinger, A., Brahman, F., et al. (2024).** *WildTeaming at Scale: From In-the-Wild Jailbreaks to (Adversarially) Safer Language Models.* arXiv:2406.18510. VERIFIED https://arxiv.org/abs/2406.18510
    HF `allenai/wildjailbreak` (ODC-BY, **gated**: AI2 form). 262K pairs including adversarial-benign contrast examples. A good source for the safety-data-mix baseline.

### Resource paper
68. **Deng, B., Wang, X., Wang, Y., Wan, Y., Ma, Y., et al. (2026).** *Qwen-Scope: Turning Sparse Features into Development Tools for Large Language Models.* arXiv:2605.11887. VERIFIED https://arxiv.org/abs/2605.11887
    Official Qwen SAEs (14 groups, 7 backbones, Qwen3 and Qwen3.5). See the resources table.

---

## 2. Open resources for the planned models

Model facts (verified on HF cards). **Qwen/Qwen3.5-9B** (Apache-2.0, Feb 2026): dense, 32 layers, d=4096, hybrid `8×(3×Gated DeltaNet→FFN, 1×Gated Attention→FFN)`, vision encoder, **thinking mode ON by default**. **Qwen/Qwen3.8-27B** (Apache-2.0, Aug 2026): dense, 64 layers, d=5120, `16×(3×DeltaNet + 1×Attention)`, vision encoder, thinking ON by default.

Practical implications:
- Disable thinking, or evaluate both modes consistently. Refusals may sit inside `<think>`.
- Hook the residual stream on the decoder layers, not the DeltaNet internals.
- Check that TransformerLens or nnsight support the hybrid architecture. If they do not, use plain PyTorch forward hooks.
- Load text-only to save memory.

| Resource | Qwen3.5-9B | Qwen3.8-27B | Link (verified) | Notes |
|---|---|---|---|---|
| Refusal-direction code | Code works for any HF model (`--model_path`). No precomputed direction | Same | https://github.com/andyrdt/refusal_direction (Apache-2.0) | Artifacts only for older models (Qwen-1.8B-chat, Llama-3-8B, etc.). Compute your own. |
| Community abliterated models | `lukey03/Qwen3.5-9B-abliterated`, `Puerh0x1/Qwen3.5-9B-abliterated` | `OBLITERATUS/Qwen3.8-27B-OBLITERATED` | seen in HF search results, **not audited** | Useful only as a sanity check that a single refusal direction exists. Some also apply LoRA, so do not treat them as clean. |
| SAEs (official Qwen-Scope) | `Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50` and `..._L0_100`: all 32 layers, TopK, **trained on Base**, "qwen" license | **None found for Qwen3.8.** `Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50` (64 layers, trained on Qwen3.5-27B base) has the same shape (64L, d=5120) but a different model, so transfer is untested | https://huggingface.co/Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50 ; https://huggingface.co/Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50 | Base-trained SAEs on post-trained models are endorsed by the Qwen-Scope report, but verify reconstruction loss. |
| J-Lens / R-Lens | **Yes**: `qwen3.5-9b` (also `qwen3.5-4b`, `qwen3.5-27b`, `qwen3.5-122b-a10b`, `qwen3.6-27b`, `qwen3.6-35b-a3b`, `gemma-3-27b-it`, `deepseek-v4-flash`) | **None for Qwen3.8** | https://huggingface.co/camilablank/workspace-lenses ; post: Blank, Bhatia, Nanda, "R-lens: Making J-lens More Faithful on Early Layers", LessWrong, Aug 5 2026, https://www.lesswrong.com/posts/nv8oedrnLXKRzNEL9/r-lens-making-j-lens-more-faithful-on-early-layers | Could decode ADL diffs as an alternative to Patchscope. For the 27B arm, **consider Qwen3.6-27B** if lens support matters. |
| Assistant Axis vectors | Not available (only Gemma-2-27B, Qwen3-32B, Llama-3.3-70B) | Not available | https://huggingface.co/datasets/lu-christina/assistant-axis-vectors (MIT); code https://github.com/safety-research/assistant-axis (MIT) | You must compute the axis yourself: 275 roles, generation plus judge scoring. That is expensive, so budget API or vLLM time, or use a reduced role set and validate it against the full set on 9B. |
| Persona vectors code | Generic (tested on Qwen2.5-7B-Instruct) | Generic | https://github.com/safety-research/persona_vectors (Apache-2.0) | Includes data-projection screening and preventative-steering training configs (Q2 and Q3 baselines). |
| CAFT code | `--qwen` option (older Qwen). Needs adaptation | Needs adaptation | https://github.com/cadentj/caft (no license shown; ask the authors or treat as all-rights-reserved) | PCA-of-diff direction selection is included. The SAE path is flagged "WIP / buggy". |
| ADL / diffing toolkit | Generic HF | Generic HF | https://github.com/science-of-finetuning/diffing-toolkit (MIT) | Includes ADL, KL, PCA, SAE-diff, crosscoder, activation oracle and agentic eval. Hydra-based. |
| Eval judges | StrongREJECT fine-tuned Gemma-2B evaluator; HarmBench-Llama-2-13b-cls; WildGuard | same | see section (g) | Run the judges in a separate process from the policy model given the VRAM limits. Use ≥2 judges (per Vishnubhotla 2026). |

---

## 3. Entries I could not verify (excluded from the main list)

- **Kempf et al. (2026)** and **Dunlap et al. (2024)** black-box diffing: cited inside the Chughtai/Engels/Nanda diffing-agents post. The papers themselves were not located or opened.
- **"Qwen3-Instruct SAE"** (search pointed to arXiv 2606.26620, titled "Discovering Millions of Interpretable Features with Sparse Autoencoders"): not opened, and the title vs content mapping is unclear.
- **HARC: Coupling Harmfulness and Refusal Directions** (arXiv 2607.00572), **Dynamic Adversarial Fine-Tuning Reorganizes Refusal Geometry** (2604.27019), **Tracking Harmfulness–Refusal Coupling** (2606.16349), **AlignGuard-LoRA** (2508.02079), **Benign Fine-Tuning Breaks Safety Alignment in Audio LLMs** (2604.16659), **The Geometry of Refusal** (2609.06934), **Beyond Shallow Alignment: post-training methods determine refusal circuits** (2609.03887), **Refusal geometry reflects refusal training** (2608.25390), **CSULoRA** (2605.30640), **Safe Pruning LoRA** (2506.18931), **Persona-Model Collapse in EM** (2605.12850). All appeared only as search-result titles and URLs. They were not opened, so authors and dates are unverified. Several (2604.27019, 2606.16349, 2609.03887, 2608.25390) look relevant to Q1 mechanism and are worth reading.
- **A paper ablating LoRA rank {4, 8, 16} and finding monotonically decreasing robustness with rank**: surfaced in a search snippet (multimodal, benign text-image FT). I could not attribute it to a specific paper.
- **"Grant et al. (2026)"** on gradient sign reversals along persona axes: mentioned in a snippet, not located.
- **Venues** for Qi 2023 (ICLR 2024), Qi 2024 (ICLR 2025) and Arditi 2024 (NeurIPS 2024), and the **ICML 2026 poster** for CAFT (icml.cc/virtual/2026/poster/60571 appeared in search results): plausible, but not confirmed on the pages opened.
- **ProFS** (the task's list): the acronym is used by Uppaal et al. for a toxicity model-editing method (DPO alternative). I did not open or verify it, and it is not refusal-specific. Excluded.
- **Anthropic "stage-wise model diffing" or other lab blog posts**: not searched or verified.
- **SPQR** (2511.19558): verified, but it is about text-to-image diffusion, so it was excluded as off-topic.
