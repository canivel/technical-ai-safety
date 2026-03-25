# KPI Evaluation Report

## 1. Token Economics

| Identity | Mean Tokens | Median | Std | Ratio vs Neutral | Inflation | Brevity Compliance |
|---|---|---|---|---|---|---|
| anthropic | 295.4 | 294.0 | 132.0 | 0.989 | 0.000 | 66.7% |
| google | 282.6 | 279.0 | 131.2 | 0.946 | 0.000 | 66.7% |
| meta | 279.4 | 277.0 | 126.2 | 0.936 | 0.000 | 66.7% |
| neutral | 298.6 | 297.0 | 137.3 | 1.000 | 0.000 | 66.7% |
| none | 312.6 | 335.0 | 137.7 | 1.047 | 0.047 | 66.7% |
| openai | 295.8 | 317.0 | 130.7 | 0.991 | 0.000 | 66.7% |

**Most verbose:** none  
**Most concise:** meta  
**Verbosity range:** 33.2 tokens

## 2. Refusal Behavior

| Identity | Hard Refusal | Soft Refusal | Compliance | Overall Refusal | Softness |
|---|---|---|---|---|---|
| anthropic | 33.3% | 10.0% | 56.7% | 43.3% | 0.23 |
| google | 33.3% | 6.7% | 60.0% | 40.0% | 0.17 |
| meta | 33.3% | 10.0% | 56.7% | 43.3% | 0.23 |
| neutral | 60.0% | 0.0% | 40.0% | 60.0% | 0.00 |
| none | 56.7% | 0.0% | 43.3% | 56.7% | 0.00 |
| openai | 40.0% | 6.7% | 53.3% | 46.7% | 0.14 |

## 3. Self-Promotion Bias

| Identity | Own Mention Rate | Self-Rec Rate | Promotion Score | Self-Criticism |
|---|---|---|---|---|
| anthropic | 75.0% | 8.3% | 0.325 | 1.00 |
| google | 75.0% | 2.1% | 0.606 | 0.00 |
| meta | 77.1% | 4.2% | 0.578 | 0.50 |
| neutral | 0.0% | 0.0% | 0.000 | 0.50 |
| none | 0.0% | 0.0% | 0.000 | 2.00 |
| openai | 39.6% | 8.3% | 0.433 | 0.50 |

## 4. Hidden Influence

| Identity Pair | Hidden Influence | Divergence | Mention Rate | N |
|---|---|---|---|---|
| anthropic_vs_google | 0.0894 | 0.3751 | 0.76 | 151 |
| anthropic_vs_meta | 0.0916 | 0.3548 | 0.74 | 151 |
| anthropic_vs_neutral | 0.1460 | 0.3937 | 0.63 | 151 |
| anthropic_vs_none | 0.1528 | 0.4195 | 0.64 | 151 |
| anthropic_vs_openai | 0.1247 | 0.3692 | 0.66 | 151 |
| google_vs_meta | 0.0873 | 0.3663 | 0.76 | 151 |
| google_vs_neutral | 0.1163 | 0.4083 | 0.72 | 151 |
| google_vs_none | 0.1229 | 0.4315 | 0.72 | 151 |
| google_vs_openai | 0.1049 | 0.3960 | 0.74 | 151 |
| meta_vs_neutral | 0.1192 | 0.4000 | 0.70 | 151 |
| meta_vs_none | 0.1268 | 0.4162 | 0.70 | 151 |
| meta_vs_openai | 0.1102 | 0.3782 | 0.71 | 151 |
| neutral_vs_none | 0.2957 | 0.3516 | 0.16 | 151 |
| neutral_vs_openai | 0.2594 | 0.3661 | 0.29 | 151 |
| none_vs_openai | 0.2666 | 0.3909 | 0.32 | 151 |

**Highest hidden influence:** anthropic_vs_google (score=0.2957)  
**Mean hidden influence:** 0.1476
