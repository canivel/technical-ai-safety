Deep Ignorance
Filtering Pretraining Data Builds Tamper-Resistant
Safeguards into Open-Weight LLMs

Deep Ignorance research paper icon
Paper
|
Hugging Face logo - Access Deep Ignorance models and datasets
Hugging Face
|
GitHub
Kyle O'Brien1,4* Stephen Casper2*
Quentin Anthony1 Tomek Korbak2 Robert Kirk2 Xander Davies2,3 Ishan Mishra2
Geoffrey Irving2 Yarin Gal2,3 Stella Biderman1
1EleutherAI    2UK AISI    3University of Oxford    4ERA Fellowship
*Equal Contribution
kyle@eleuther.ai    scasper@mit.edu
Graph showing LLM performance: filtered models maintain general capabilities while resisting adversarial fine-tuning on biothreat knowledge
Training data filtering makes LLMs resistant to adversarial fine-tuning without sacrificing general performance. Models whose training data has been filtered to remove text related to dual-use biology topics (left) have unaffected general capabilities and (right) have low biothreat proxy capabilities and resist up to 10,000 steps and 300M tokens of adversarial fine-tuning.

"Open weights allow global research communities to both advance capabilities and address model flaws by providing them with direct access to a critical AI component that is prohibitively expensive for most actors to develop independently. However, the open release of model weights could also pose risks of facilitating malicious or misguided use or perpetuating model flaws and biases. Once model weights are available for public download, there is no way to implement a wholesale rollback of all existing copies of the model."

— International AI Safety Report (Bengio et al., 2025)
Introduction
A risk as LLMs grow more capable is that they learn unsafe knowledge during training. These dangerous capabilities could be exploited by bad actors. Today's safeguards focus on suppressing unsafe knowledge in post-training, often via refusal training. The unsafe knowledge remains in the model's weights.

Open-weight models, those that users can download and modify locally, offer unique benefits related to transparency, research, and the deconcentration of power. However, they are vulnerable to "tampering attacks" that can remove their safety training. For instance, it is straightforward to train open-weight models never to refuse unsafe requests. This is of increasing concern as open-weight models begin to rival the capabilities of the best closed-weight models.

We explore an intuitive yet understudied question: Can we prevent LLMs from learning unsafe technical capabilities (such as CBRN) by filtering out enough of the relevant pretraining data before we begin training a model? We train multiple 6.9B LLMs from scratch on an unfiltered dataset and on filtered versions where we filtered out biorisk knowledge. We observe three main results:

Knowledge Prevention: The filtered models perform significantly worse on our biorisk knowledge evaluations, nearly at random chance. Crucially, filtering does not lead to notable regressions in general knowledge. These results suggest that data filtering may be a simple way to prevent models from learning dangerous capabilities without sacrificing utility.
Tamper-Resistance: Open-weight models can be fine-tuned by downstream users on biorisk data. We study this attack by fine-tuning our models on 300M tokens of high-quality biorisk-related documents. We find that performance can improve, but that it is still well below the no-filtering baseline. Data filtering is significantly more tamper-resistant than current safeguards.
Defense-in-Depth: We demonstrate that data filtering cannot prevent LLMs from leveraging harmful knowledge provided in-context, but that Circuit-Breaking-based techniques offer complementary defenses. However, we show that none of the defenses we test are resistant to staged attacks that combine fine-tuning and in-context retrieval.
Taken together, these results suggest that rigorous pretraining data filtering is a promising method for preventing acquisition of dangerous technical capabilities without obvious degradation in overall model utility. Our efficient data approach allowed us to perform filtering with less than a 1% increase in training compute (FLOPS). We release our models, optimizer states, and intermediate checkpoints to enable future research into domains including data-driven AI security, pretraining research, machine unlearning, and mechanistic interpretability. We are especially excited to see future work that addresses our limitations, such as studying the filtering of other types of knowledge and developing scaling trends. See our paper for more details, discussion, and sketches of future work.

Diagram of multi-stage data filtering pipeline: blocklist filtering followed by classifier evaluation for biothreat content removal
Our multi-stage data filtering pipeline: Our goal is to filter out data related to unwanted topics. We study biothreat-proxy knowledge as a representative example. All documents undergo initial "blocklist" filtering, where those without prohibited terms are retained without further review. Documents containing blocked terms (e.g., "pathogen(s)") are escalated to a fine-tuned text classifier that evaluates semantic content. The classifier assigns probability scores for unsafe content: documents scoring below the predetermined threshold are retained, while those exceeding it are excluded from the training corpus. In practice, the vast majority of documents are approved by the blocklist and thus do not require review by the classifier stage.