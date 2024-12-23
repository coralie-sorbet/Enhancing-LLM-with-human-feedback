# Enhancing Large Language Models with Human Feedback via Reinforcement Learning

## Transformer Architectures: A Detailed Explanation

Transformers, introduced in the seminal paper "Attention is All You Need" by [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762), have revolutionized Natural Language Processing (NLP). The key innovation in Transformer architecture is the self-attention mechanism, which allows the model to dynamically weigh the importance of each input token relative to others. This enables the model to capture both local and global dependencies in text efficiently.

#### Key Components
1. **Multi-Head Self-Attention**:
   - The self-attention mechanism computes a weighted sum of input tokens, where the attention scores are learned during training. Multi-head attention allows the model to attend to different parts of the input simultaneously, enhancing its ability to capture nuanced relationships across different token dependencies.
   - **Mathematical representation**:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Where $Q$, $K$, and $V$ are the query, key, and value matrices, and ${d_k}$ is the dimensionality of the keys.


2. **Feedforward Layers**:
   - After the attention step, the output of each token is passed through position-wise fully connected layers. These layers apply non-linear transformations to each token's representation independently, ensuring the model can capture complex relationships.

3. **Positional Encoding**:
   - Since Transformers process input tokens in parallel, they lack the inherent understanding of token order. To address this, positional encodings are added to the input embeddings, typically in the form of sinusoidal functions, ensuring the model retains the sequence order.

### Causal Transformers
Causal Transformers, such as GPT, LLAMA, and others, are optimized for autoregressive tasks, where the model predicts the next token in a sequence based on prior tokens. These models are trained using a left-to-right masking mechanism, ensuring each token attends only to its predecessors during training, which helps them perform well on text generation tasks.

Key characteristics:
- **Training Objective**: Causal Transformers are trained to minimize the negative log-likelihood of the next token:

$\mathcal{L} = -\sum_{t=1}^{T} \log P(x_t \mid x_{<t})$

Where $x_t$ represents the token at position $t$ and $x_{<t}$ denotes all prior tokens in the sequence.

- **Applications**: These models are used in text generation, dialogue systems, code completion, and many other areas where predicting the next element in a sequence is critical.

---

## Reinforcement Learning with Human Feedback (RLHF)

### Conceptual Framework
Reinforcement Learning with Human Feedback (RLHF) improves the alignment of language models with human preferences by incorporating human feedback during training. This feedback guides the model to adjust its behaviour to better match human expectations, refining its responses to be more relevant, coherent, and aligned with user intentions.

#### Steps in RLHF:
- 1. **Training a Reward Model (RM):** 
Human evaluators rank the model's outputs based on specific criteria such as relevance, coherence, and alignment with human expectations. These rankings are used to train a Reward Model (RM), which predicts a scalar reward score for each output.

- 2. **Optimizing the Language Model with Reinforcement Learning:**
The language model is fine-tuned using Reinforcement Learning, particularly Proximal Policy Optimization (PPO). The RM is employed as a reward function, guiding the model to maximize expected reward.
The objective function for RLHF is:

<p align="center">
  <b>max</b><sub>&#x3B8;</sub> <i>E</i><sub>x &#8764; &#x3C0;<sub>&#x3B8;</sub></sub>[R(x)]
</p>

Where:
   - $\pi_\theta$ is the **policy** (the model),
   - $R(x)$ is the **reward** for a given output $x$,
   - $\theta$ is the set of model parameters.

#### Challenges
- **Computational Cost**: Training models with human feedback is resource-intensive, requiring significant computational resources and time.
- **Balancing Fluency and Adherence to Human Preferences**: Ensuring the model remains fluent and coherent while adhering to human feedback is a delicate balance. Often, improvements in alignment come with the trade-off of reduced fluency or generalization.

### Practical Implementation
- Libraries: Hugging Face's [TRL](https://github.com/huggingface/trl) provides utilities for implementing RLHF workflows efficiently.
- Example use case: Fine-tuning GPT-2 on a specific task, such as generating responses aligned with certain ethical guidelines, involves:

1. Collecting human feedback through rankings or preferences.
2. Training a reward model to predict the quality of responses based on this feedback.
3. Fine-tuning the GPT-2 model using PPO with the reward model as the guiding signal.

---

## Proximal Policy Optimization (PPO): A Core Algorithm for RLHF

### Overview
Proximal Policy Optimization (PPO) is one of the most widely used reinforcement learning algorithms for fine-tuning models with human feedback. PPO strikes a balance between exploration and exploitation by ensuring that policy updates are not too drastic, preventing instability in training.

### Key Features
1. **Clipped Objective Function:**
   - PPO’s objective function is designed to limit the change in the probability ratio between the new and old policies, ensuring stable updates:

$$
\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) A_t\right)\right]
$$

where $r_t(\theta)$ is the probability ratio, $\epsilon$ is a small hyperparameter that controls the clipping, and $A_t$ is the advantage function.

2. **Entropy Regularization:**
   - To encourage the policy to explore diverse actions, PPO includes entropy regularization. This discourages overly deterministic policies, promoting a more diverse set of responses from the model.

### Resources
- [CleanRL's PPO Implementation](https://github.com/vwxyzjn/cleanrl/tree/master) provides a minimalistic yet powerful implementation.
- Hugging Face’s [Quickstart Guide](https://huggingface.co/blog/rlhf) for practical insights into PPO with RLHF.

---

## Applications and Future Directions

### Demonstrated Use Case
In this project, RLHF is applied to fine-tune a GPT-2 model. The fine-tuned model generates text that aligns more closely with human preferences by training a reward model and optimising with PPO.

### Broader Applications
- **Content Moderation:** Aligning models with ethical and safety guidelines to filter harmful or inappropriate content.
- **Education:** Developing personalized tutoring systems that can adapt to individual student needs by adjusting the model's behaviour based on feedback.
- **Healthcare:** Using RLHF to generate patient-friendly explanations of medical information, ensuring both clarity and accuracy.

### Future Work
- **Cost-efficient Alternatives to RLHF:** Future research could explore hybrid methods that combine supervised fine-tuning and lighter RL techniques to reduce the computational burden of RLHF.
- **Scaling to Larger Models:** Investigating the application of RLHF on models like GPT-4 or LLAMA 2, exploring scalability and efficiency in fine-tuning large-scale models.

---

## References
1. Vaswani, A., et al. (2017). Attention Is All You Need. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
2. OpenAI. (2022). Training language models to follow instructions with human feedback. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
3. Hugging Face. (2023). Reinforcement Learning with Human Feedback. [Blog Post](https://huggingface.co/blog/rlhf)
4. Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
5. Anthropic. (2022). Constitutional AI: Harmlessness from AI Feedback. [arXiv:2204.05862](https://arxiv.org/abs/2204.05862)

