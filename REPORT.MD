# Enhancing Large Language Models with Human Feedback via Reinforcement Learning  

## Understanding and Utilizing Transformers  

### Transformer Architectures: A Detailed Explanation  

Transformers, first introduced in the landmark paper *"Attention is All You Need"* by [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762), have become the cornerstone of modern machine learning, especially in NLP. At the heart of their success lies the **self-attention mechanism**, which enables the model to dynamically assess the relevance of each token in the input sequence relative to others. This innovation allows Transformers to capture both local and global dependencies in data, making them incredibly versatile for tasks like text generation, machine translation, and more.  

### Key Components  

#### 1. Input Representation  
Transformers process sequences of tokens. Each token in the sequence is represented as a dense vector embedding:  

$X = [x_1, x_2, ..., x_n]$  

- **Embeddings**: These vectors encode semantic information about the tokens.  
- **Positional Encoding**: Since Transformers process tokens in parallel, positional encodings are added to embeddings to capture the order of tokens in the sequence. These are typically sinusoidal functions.  

#### 2. Self-Attention Mechanism  

The self-attention mechanism allows the model to focus on different parts of the input sequence while processing each token. For every token, three vectors are derived using learned weight matrices:  

$Q = XW^Q, \quad K = XW^K, \quad V = XW^V $$

Here, $Q$, $K$, and $V$ are the query, key, and value vectors, respectively, and $W^Q$, $W^K$, and $W^V$ are learnable weight matrices.  

#### 3. Attention Score Calculation  

The relationship between tokens is computed using the scaled dot product of the query and key vectors:  

$\text{Attention Score} = \frac{QK^T}{\sqrt{d_k}}$

Here, $d_k$ represents the dimensionality of the key vectors. Scaling by $\sqrt{d_k}$ prevents excessively large values that can destabilize training.  

#### 4. Softmax Normalization  

The attention scores are normalized using the softmax function to convert them into probabilities:

$\text{Softmax}(\text{Attention Score}) = \frac{\exp(\text{Attention Score})}{\sum \exp(\text{Attention Score})}$

#### 5. Output Calculation  

The final output for each token is computed as a weighted sum of the value vectors, where the normalized attention scores serve as weights:  

$\text{Output} = \text{Softmax}(\text{Attention Score}) \times V$

#### 6. Multi-Head Attention  

To improve the model's ability to capture diverse relationships within the input sequence, Transformers use **multi-head attention**. Multiple sets of $ Q $, $ K $, and $ V $ are computed to create multiple "heads" of attention, each focusing on different aspects of the input. These outputs are concatenated and transformed linearly:  

$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$

$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$

This mechanism enables the model to analyze the input from multiple perspectives simultaneously.  

The following diagram from [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762) illustrates scaled dot-product and attention multi-head attention:  

<div align="center">
  <img src="https://github.com/user-attachments/assets/1939da4c-2a3e-4aa2-919a-170302574c07" alt="multi head self attention" width="400">
</div>

### Encoder-Decoder Structure  

The Transformer architecture uses an **encoder-decoder** framework:  

- **Encoder**: Processes the input sequence and generates intermediate representations.  
- **Decoder**: Generates the output sequence by attending to both the encoder's representations and previously generated tokens.  

Each layer in the encoder and decoder stacks consists of:  
- Multi-head attention layers.  
- Feedforward layers.  
- Residual connections with layer normalization.  

The architecture, as visualized in the diagram below from [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762), efficiently handles complex relationships between input and output tokens:  

<div align="center">  
  <img src="https://github.com/user-attachments/assets/2542fd51-5e70-45dd-b792-91ce84eb0f7f" alt="The Transformer- model architecture" width="400">  
</div>  

### Self-Attention in Transformers  

The self-attention mechanism is central to both the encoder and decoder. It helps the model focus on relevant parts of the sequence while encoding input data or generating output predictions. This capability underpins the success of large-scale models like GPT (Generative Pre-trained Transformer) and BERT (Bidirectional Encoder Representations from Transformers), which excel at tasks ranging from text understanding to creative content generation.  


### Causal Transformers
Causal Transformers, such as GPT and LLAMA, are optimized for autoregressive tasks, where the model predicts the next token in a sequence based on prior tokens. These models use a left-to-right masking mechanism during training, ensuring each token attends only to its predecessors. This design helps them perform exceptionally well on text-generation tasks.

#### Key Characteristics
- **Training Objective**:
  Causal Transformers are trained to minimize the negative log-likelihood of the next token:

$\mathcal{L} = -\sum_{t=1}^{T} \log P(x_t \mid x_{<t})$

Where $x_t$ represents the token at position $t$ and $x_{<t}$ denotes all prior tokens in the sequence.

- **Applications**:
  These models are widely used in text generation, dialogue systems, code completion, and other areas requiring next-token prediction.

---

## Reinforcement Learning with Human Feedback (RLHF)

### Conceptual Framework
Reinforcement Learning with Human Feedback (RLHF) aligns language models with human preferences by incorporating feedback during training. This feedback helps refine the model’s behavior to produce outputs that are more relevant, coherent, and aligned with user intentions.

#### Steps in RLHF
1. **Training a Reward Model (RM):**
   Human evaluators rank the model's outputs based on specific criteria such as relevance and coherence. These rankings are used to train a Reward Model (RM), which predicts a scalar reward score for each output.

2. **Optimizing the Language Model with Reinforcement Learning:**
   The language model is fine-tuned using Reinforcement Learning, particularly Proximal Policy Optimization (PPO). The RM serves as a reward function, guiding the model to maximize expected reward.

   **Objective Function**:

<p align="center">
  <b>max</b><sub>&#x3B8;</sub> <i>E</i><sub>x &#8764; &#x3C0;<sub>&#x3B8;</sub></sub>[R(x)]
</p>

   Where:
   - $\pi_\theta$ is the **policy** (the model),
   - $R(x)$ is the **reward** for a given output $x$,
   - $\theta$ represents the model parameters.

### Challenges
- **Computational Cost**: Training models with human feedback requires significant resources and time.
- **Balancing Fluency and Adherence**: Achieving a balance between fluency and alignment with human preferences is challenging, as improving alignment may reduce fluency or generalization.

### Practical Implementation
- **Libraries**: Hugging Face's [TRL](https://github.com/huggingface/trl) provides tools for implementing RLHF workflows efficiently.
- **Example Use Case**: Fine-tuning GPT-2 involves collecting human feedback, training a reward model to rank outputs, and fine-tuning the GPT-2 model using PPO guided by the reward model.

---

## Proximal Policy Optimization (PPO): A Core Algorithm for RLHF

### Overview
Proximal Policy Optimization (PPO) is a reinforcement learning algorithm widely used for fine-tuning models with human feedback. PPO ensures stable training by limiting the magnitude of policy updates.

### Key Features
- **Clipped Objective Function:**

$$
\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) A_t\right)\right]
$$

  Where $r_t(\theta)$ is the probability ratio, $\epsilon$ is a hyperparameter controlling the clipping, and $A_t$ is the advantage function.

- **Entropy Regularization:**
  PPO includes entropy regularization to encourage policy exploration and prevent overly deterministic behaviors, promoting diversity in responses.

### Resources
- [CleanRL's PPO Implementation](https://github.com/vwxyzjn/cleanrl/tree/master) offers a minimalistic implementation.
- Hugging Face’s [Quickstart Guide](https://huggingface.co/blog/rlhf) provides practical insights into PPO with RLHF.

---

## Applications and Future Directions

### Demonstrated Use Case
In this project, RLHF is applied to fine-tune a GPT-2 model. The fine-tuned model generates text that aligns closely with human preferences by leveraging a reward model and optimizing with PPO.

### Broader Applications
- **Content Moderation**: Aligning models with ethical guidelines to filter harmful content.
- **Education**: Developing personalized tutoring systems adaptable to individual student needs.
- **Healthcare**: Generating patient-friendly explanations of medical information with clarity and accuracy.

### Future Work
- **Cost-efficient Alternatives**: Exploring hybrid methods combining supervised fine-tuning with lightweight RL techniques to reduce computational costs.
- **Scaling to Larger Models**: Investigating RLHF’s application on large-scale models like GPT-4 or LLAMA 2, focusing on scalability and efficiency.

---

## References
- [Self-Attention in NLP](https://www.youtube.com/watch?v=5vcj8kSwBCY)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Attention? Attention!](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html)
- [Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf)
1. Vaswani, A., et al. (2017). Attention Is All You Need. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
2. OpenAI. (2022). Training language models to follow instructions with human feedback. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
3. Hugging Face. (2023). Reinforcement Learning with Human Feedback. [Blog Post](https://huggingface.co/blog/rlhf)
4. Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
5. Anthropic. (2022). Constitutional AI: Harmlessness from AI Feedback. [arXiv:2204.05862](https://arxiv.org/abs/2204.05862)
