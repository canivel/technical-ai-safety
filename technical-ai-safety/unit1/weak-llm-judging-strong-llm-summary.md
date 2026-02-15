# On Scalable Oversight with Weak LLMs Judging Strong LLMs - Summary

**Authors:** Zachary Kenton*, Noah Y. Siegel*, János Kramár, Jonah Brown-Cohen, Samuel Albanie, Jannis Bulian, Rishabh Agarwal, David Lindner, Yunhao Tang, Noah D. Goodman, Rohin Shah
**Organization:** Google DeepMind
**Published:** July 2024
**arXiv:** 2407.04622v2

---

## Core Problem

As AI systems reach and exceed expert human levels on important tasks, human feedback for alignment needs to remain accurate. **Scalable oversight** aims to leverage advancing AI capabilities to allow humans to provide accurate training signals to superhuman AI.

**Why this matters for safety:** Current LLMs can generalize behaviors that exploit inaccurate training signals (e.g., from sycophancy to reward tampering). More powerful AI may generalize these further to dangerous behaviors like **scheming** - performing well in training to gain power later.

---

## Protocols Studied

### Baseline Protocols (Not Scalable Oversight)

| Protocol | Description |
|----------|-------------|
| **QA without article** | Judge directly answers question (zero-shot baseline) |
| **QA with article** | Judge reads article and answers (extractive tasks only) |

### Assigned-Role Protocols

| Protocol | Description |
|----------|-------------|
| **Consultancy** | Single AI assigned an answer (50/50 correct/incorrect), tries to convince judge who asks questions |
| **Debate** | Two AIs (Alice & Bob) compete to convince judge, assigned opposing answers |

### Open-Role Protocols

| Protocol | Description |
|----------|-------------|
| **Open Consultancy** | Consultant chooses which answer to argue for |
| **Open Debate** | One debater (protagonist) chooses answer, other (antagonist) assigned opposite |

---

## Task Types

### Extractive QA (Information Asymmetry)
- Judge doesn't see the article; debaters/consultants do
- Debaters extract passages using `<passage>` tags
- Passages verified as `<v_passage>` (verified) or `<u_passage>` (unverified)

| Dataset | Description |
|---------|-------------|
| **QuALITY** | Long reading comprehension (~5k tokens) |
| **BoolQ** | Binary questions about Wikipedia articles |
| **GPQA-Extractive** | Very hard science questions with expert explanation |

### Closed QA (No Information Asymmetry)
- No article source; weak-strong gap is intrinsic capability/reasoning

| Dataset | Description |
|---------|-------------|
| **MMLU** | College-level diverse subjects |
| **GSM8KQA** | Grade school math (binary code solutions) |
| **PrOntoQA** | Synthetic logical reasoning |
| **TruthfulQA** | Adversarial questions testing truthfulness |
| **GPQA** | Very hard "Google-proof" science questions |

### Multimodal
| Dataset | Description |
|---------|-------------|
| **MMMU** | College-level image+text questions |

---

## Models Used

| Role | Models |
|------|--------|
| **Judges** | Gemma7B, GPT-3.5, Gemini Pro 1.0, Gemini Pro 1.5 |
| **Debaters/Consultants** | Gemini Pro 1.5 (main experiments) |

Judge models ordered weakest to strongest: Gemma7B < GPT-3.5 ≈ Pro 1.0 < Pro 1.5

---

## Key Findings

### 1. Debate Consistently Outperforms Consultancy
- True across **all task types** (extractive, closed, multimodal)
- Previously only shown on single extractive task (QuALITY)

### 2. Debate vs. Direct QA Depends on Task Type

| Task Type | Result |
|-----------|--------|
| **Extractive** | Debate > QA without article, but Debate < QA with article |
| **Closed** | Mixed results - small or no advantage to debate over QA without article |
| **Multimodal** | No significant difference between debate and QA without article |

### 3. Ablations Show Little Effect
Changes to setup had minimal impact:
- Number of turns (1 vs 3)
- Best-of-N sampling for debaters
- Few-shot prompting for judges
- Chain-of-thought for judges (sometimes harmful)
- Sequential vs simultaneous turns
- Answer order (positional bias exists but randomizing sufficient)

### 4. Open Consultancy Amplifies Mistakes
- Judge equally convinced by consultant whether correct or incorrect
- **Risk:** Using weak judges for training via consultancy may amplify consultant's incorrect behavior

### 5. Open Debate Better at Detecting Errors
- Judge follows debater's choice **less frequently** than in open consultancy
- When protagonist chooses correctly: judge does slightly worse than open consultancy
- When protagonist chooses **incorrectly**: judge does **much better** at detecting this
- **Implication:** Training signal from weak judge in open debate less likely to amplify incorrect answers

### 6. Stronger Debaters → Higher Judge Accuracy
- Measured via Elo scores across debater models
- Effect is **modest** compared to previous single-task studies
- Provides some evidence debate satisfies key scalable oversight objective

---

## Elo Analysis Results

Debater Elo scores were calculated using cross-play debates judged by Pro 1.5:

**Key observations:**
- Correct answer debaters have higher Elo than incorrect answer debaters (easier to argue for truth)
- Effect more pronounced on closed tasks
- Stronger models have higher Elo
- Higher aggregate Elo correlates with higher judge accuracy (especially extractive tasks)
- Gemini-family judges assign lower Elo to GPT-3.5 (possible self-preference bias)

---

## Why Results Are Mixed (Caveats)

The authors remain optimistic despite mixed results:

1. **Inference-only study**: Debate as a training protocol not directly tested; inference properties checked first

2. **Finetuning bias**: Current SFT/RLHF favors direct QA format (typical evaluation benchmark format). Judging debates may be more out-of-distribution

3. **RLHF models**: Debaters already finetuned for honesty, which is what debate aims to incentivize - may hamper dishonest debater

---

## Implications for AI Safety

### Positive Signs
- Debate consistently beats consultancy (the 50/50 correct/incorrect baseline)
- Open debate provides better training signal than open consultancy when AI is wrong
- Stronger debaters generally lead to better judge accuracy

### Concerns
- Debate doesn't consistently beat direct QA baselines
- Weak judges (Gemma7B) do worse with debate than QA without article
- Effect sizes are modest

---

## Error Analysis (24 Transcripts)

### PrOntoQA (Logical Reasoning)
- 11/12 failures: Logical errors by the judge
- 1/12: Poor debating by honest debater
- Losing debater often correctly identified flaws but judge failed to capitalize

### TruthfulQA
More diverse errors:
- 6/12: Question naturally ambiguous; dishonest debater exploited interpretation
- 2/12: Suboptimal debate from honest debater
- 1/12: Logical error from judge
- 1/12: Impossible without verified external information
- 2/12: Judge inattentive; dishonest debater focused them on incomplete portion

---

## Limitations

1. **Inference-only**: Doesn't test debate as a training protocol (the main motivation)
2. **RLHF-trained models**: Already trained for honesty, may limit findings
3. **No humans**: Study uses LLMs as stand-ins for human judges
4. **Not superhuman**: AI used is far from superhuman on many capabilities
5. **Key uncertainty**: Whether humans can accurately judge arguments from superhuman AI

---

## Future Directions

1. **Train debater/consultant models** via judge signal on studied tasks
2. **Human judge studies** to compare to LLM judges
3. **Other weak-strong asymmetries**: Tool use, code execution, different modality access
4. **Other protocols**: Cross-examination debate, iterated amplification
5. **Robustness testing**: Distribution shift (easy→hard tasks), misaligned models

---

## Key Takeaway

> Debate shows promise as a scalable oversight protocol - it consistently outperforms consultancy and provides better training signals in open-role settings. However, the challenge remains to show significant improvement over direct QA baselines. The authors hypothesize this may require finetuning judges specifically on the debate judging task.

---

## Experimental Scale

- **9 tasks** × **128 questions each**
- ~**5 million model generation calls**
- Multiple judge models, protocols, and ablations

---

## Citation

```bibtex
@article{kenton2024scalable,
  title={On scalable oversight with weak LLMs judging strong LLMs},
  author={Kenton, Zachary and Siegel, Noah Y and Kram{\'a}r, J{\'a}nos and Brown-Cohen, Jonah and Albanie, Samuel and Bulian, Jannis and Agarwal, Rishabh and Lindner, David and Tang, Yunhao and Goodman, Noah D and Shah, Rohin},
  journal={arXiv preprint arXiv:2407.04622},
  year={2024}
}
```
