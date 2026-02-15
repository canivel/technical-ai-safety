# Frontier Safety Framework - Summary

**Organization:** Google DeepMind
**Version:** 1.0
**Target Implementation:** Early 2025

---

## Purpose

The Frontier Safety Framework is a set of protocols to address **severe risks from powerful capabilities of future foundation models**. It complements Google's existing AI responsibility and safety practices while enabling innovation consistent with their AI Principles.

---

## Core Concept: Critical Capability Levels (CCLs)

**Critical Capability Levels** are capability thresholds at which models may pose heightened risk without additional mitigations.

CCLs are determined by:
1. Analyzing high-risk domains
2. Identifying main paths through which a model could cause harm
3. Defining the **minimal set of capabilities** a model must possess to cause that harm

---

## Framework Components

### 1. Critical Capability Levels
Capability thresholds across four domains:
- Autonomy
- Biosecurity
- Cybersecurity
- Machine Learning R&D

### 2. Early Warning Evaluations
- Periodic testing to detect when models approach CCLs
- Evaluation frequency: **Every 6x in effective compute** or **every 3 months of fine-tuning progress**
- Designed to provide adequate safety buffer before reaching a CCL

### 3. Mitigations
Two categories:
- **Security mitigations**: Prevent exfiltration of model weights
- **Deployment mitigations**: Manage access to and prevent expression of critical capabilities

---

## Security Mitigations (5 Levels)

| Level | Description | Measures |
|-------|-------------|----------|
| **0: Status quo** | Industry standard | Multi-factor auth, basic access control, secure development standards, red-team tests |
| **1: Controlled access** | ACL hygiene, non-forgeable lineage (~RAND L3) | Limited access to raw model representations, isolation of dev/prod models, SLSA Build L3, hardening of ML platforms |
| **2: Lockdown of unilateral access** | Restrictions on unilateral access (~RAND L3-L4) | No unilateral access to raw models outside core research team, exceptions by business need |
| **3: High-trust developer environments** | Protection against exfiltration (~RAND L4) | Access only through hardened, tamper-resistant workstations (HTDE) with enhanced logging |
| **4: Advanced protection** | Not accessible to humans (~RAND L5) | Minimal trusted computing base (TCB), TPUs with confidential compute, dedicated hardware pods |

---

## Deployment Mitigations (4 Levels)

| Level | Description | Measures |
|-------|-------------|----------|
| **0: Status quo** | General safeguards | Safety finetuning, filters against misuse and harmful behavior |
| **1: Mitigations targeting critical capability** | Full suite of safeguards | Safety fine-tuning, misuse filtering/detection, response protocols, periodic red-teaming |
| **2: Safety case with red team validation** | Pre-deployment validation | Robustness target based on safety case, deployment only after demonstrated mitigation effectiveness, automated monitoring |
| **3: Prevention of access** | High confidence capabilities cannot be accessed | **Currently an open research problem**; highly restricted and monitored internal use |

---

## Critical Capability Levels by Domain

### Autonomy

| CCL | Description | Rationale |
|-----|-------------|-----------|
| **Autonomy Level 1** | Capable of expanding its effective capacity by autonomously acquiring resources and running/sustaining additional copies of itself on rented hardware | Could pose difficult-to-predict, large-magnitude risks; activity could scale without resource constraints; especially difficult to constrain if supported by well-equipped bad actors |

### Biosecurity

| CCL | Description | Rationale |
|-----|-------------|-----------|
| **Bio Amateur Enablement Level 1** | Significantly enabling a non-expert to develop known biothreats that could cause severe harm | Many biothreats are out of reach of non-experts due to knowledge gaps; LLMs helping overcome these gaps increase vulnerability to fatal attacks |
| **Bio Expert Enablement Level 1** | Significantly enabling an expert (PhD+) to develop novel biothreats resulting in high-severity incidents | A small number of bio agents could cause exceptional harm; discovery of enhancements increases chances of severe bio attacks/accidents |

### Cybersecurity

| CCL | Description | Rationale |
|-----|-------------|-----------|
| **Cyber Autonomy Level 1** | Fully automating opportunistic cyberattacks on organizations with limited security posture | Automation significantly lowers costs and enables execution at scale |
| **Cyber Enablement Level 1** | Enabling an amateur to carry out sophisticated, severe attacks (e.g., disrupting critical national infrastructure) | Severe attacks currently require significant expertise; improved LLM capabilities could help amateurs overcome difficult steps in planning/execution |

### Machine Learning R&D

| CCL | Description | Rationale |
|-----|-------------|-----------|
| **ML R&D Level 1** | Could significantly accelerate AI research at a cutting-edge lab (e.g., improving algorithmic progress by 3X) | Mismanagement could enable proliferation of cutting-edge AI to malicious actors |
| **ML R&D Level 2** | Could fully automate the AI R&D pipeline at a fraction of human labor costs, enabling hyperbolic growth in AI capabilities | Could give any actor with adequate compute the ability to reach powerful capabilities quickly; could result in proliferation of unprecedented systems |

---

## Key Protocol: Response to Evaluation Thresholds

When a model passes early warning evaluations:

1. **Formulate response plan** based on CCL analysis and evaluation results
2. **Consider additional risks** flagged by review and deployment context
3. **Apply appropriate mitigations**

**If mitigations aren't ready:**
- Put on hold further deployment or development
- Implement additional protocols (e.g., more precise early warning evaluations)
- Ensure models don't reach CCLs without appropriate security mitigations
- Ensure models with CCLs aren't deployed without appropriate deployment mitigations

---

## Future Work

1. **Greater precision in risk modeling**: Improve understanding of risks, refine CCLs, forecast arrival of CCLs

2. **Capability elicitation**: Equip evaluators with state-of-the-art techniques to avoid underestimating model capabilities

3. **Mitigation plans**: Balance risk mitigation with access and innovation; develop plans mapping CCLs to security/deployment levels

4. **Updated set of risks and mitigations**:
   - **Misaligned AI**: Protection against adversarial AI systems
   - **Chemical, radiological, nuclear risks**: Potential candidates for framework inclusion
   - **Higher CCLs**: More advanced capabilities as AI progresses

5. **Involving external authorities and experts**: Internal policies around alerting stakeholders, involving third parties in risk assessment

---

## Context: Responsible Capability Scaling

The Framework is informed by the broader conversation on Responsible Capability Scaling. Core components:

1. Identify capability levels posing heightened risk without additional mitigations
2. Implement protocols to detect attainment of such capability levels
3. Prepare and articulate mitigation plans in advance
4. Where appropriate, involve external parties to inform and guide the approach

**Related work cited:**
- UK Government emerging processes for frontier AI safety
- METR's Responsible Scaling Policy work
- Anthropic's Responsible Scaling Policy
- OpenAI's Preparedness Framework

---

## Key Definitions

- **Effective compute**: A measure integrating model size, dataset size, algorithmic progress, and compute into a single metric. A 6x increase roughly corresponds to 2-2.5x increase in model size.

- **Autonomy**: Potential misuse of AI models with significant capacity for agency and flexible action/tool use over long time horizons and across multiple domains.

- **Model capabilities**: Includes capabilities resulting from any reasonably foreseeable fine-tuning and scaffolding to turn the model into a functioning system.

---

## Key Takeaway

> The Frontier Safety Framework operationalizes Responsible Capability Scaling through three pillars: defining Critical Capability Levels, implementing early warning evaluations with safety buffers, and applying calibrated security and deployment mitigations.

The framework acknowledges significant uncertainty ("exploratory and based on preliminary research") and commits to periodic review and substantial evolution as understanding improves. Notably, Level 3 deployment mitigations (preventing capability access entirely) remain "an open research problem."

---

## Authors

**Core team:** Lewis Ho, Rohin Shah, Celine Smith, Seb Farquhar, Seb Krier, Dave Orr, Max Poletto, Claudia van der Salm

**Leadership:** Allan Dafoe, Helen King, Tom Lue, Nicklas Lundblad, Demis Hassabis, Anca Dragan, Shane Legg

**With input from METR** on Responsible Capability Scaling
