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

$Q = XW^Q, \quad K = XW^K, \quad V = XW^V$

Here, $Q$, $K$, and $V$ are the query, key, and value vectors, respectively, and $W^Q$, $W^K$, and $W^V$ are learnable weight matrices.  

#### 3. Attention Score Calculation  

The relationship between tokens is computed using **scaled dot-product attention**. The scaled dot-product is calculated by taking the dot product of the query $Q$ and key $K$ vectors, and then scaling it by the square root of the key dimension $\sqrt{d_k}$, where $d_k$ is the dimensionality of the key vectors. This scaling helps stabilize gradients during training by preventing overly large values:

$\text{Attention Score} = \frac{QK^T}{\sqrt{d_k}}$

Here, the dot product measures the similarity between the query and key, and scaling by $\sqrt{d_k}$ ensures the values remain manageable.  

#### 4. Softmax Normalization  

The attention scores are normalized using the softmax function to convert them into probabilities:

$\text{Softmax}(\text{Attention Score}) = \frac{\exp(\text{Attention Score})}{\sum \exp(\text{Attention Score})}$

#### 5. Output Calculation  

The final output for each token is computed as a weighted sum of the value vectors, where the normalized attention scores serve as weights:  

$\text{Output} = \text{Softmax}(\text{Attention Score}) \times V$

#### 6. Multi-Head Attention  

To improve the model's ability to capture diverse relationships within the input sequence, Transformers use **multi-head attention**. Multiple sets of $Q$, $K$, and $V$ are computed to create multiple "heads" of attention, each focusing on different aspects of the input. These outputs are concatenated and transformed linearly:  

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

---

## Causal Transformers
Causal Transformers, such as GPT and LLAMA, are specifically optimized for autoregressive tasks, where the model generates a sequence by predicting the next token based on its preceding tokens. Unlike bidirectional models, which consider both left and right context, Causal Transformers employ a **left-to-right** attention mechanism during training. This ensures that each token can only attend to the tokens that have already been generated (its predecessors), preventing future tokens from influencing the prediction of the current token. This causal design is crucial for tasks like text generation, where the model needs to produce coherent outputs one token at a time.

#### Key Characteristics
- **Training Objective**:
  Causal Transformers are trained with the goal of minimizing the negative log-likelihood of the next token in the sequence. The model learns to predict the probability of a token $x_t$ given the sequence of preceding tokens $x_{<t}$ (i.e., the context up to but not including $x_t$). The training objective is formalized as:

$\mathcal{L} = -\sum_{t=1}^{T} \log P(x_t \mid x_{<t})$

Where $x_t$ represents the token at position $t$ and $x_{<t}$ denotes all prior tokens in the sequence.

This objective ensures the model learns to generate the most likely next token, given the previous tokens, which is essential for autoregressive tasks.

- **Causal Masking**:  
  During training, a causal (or autoregressive) masking mechanism is applied to the attention weights. This ensures that the attention for a token $x_t$ only focuses on tokens $x_1, x_2, ..., x_{t-1}$, and not on any future tokens. This masking prevents the model from "cheating" by seeing future information and helps it generate tokens one step at a time, making the model suitable for real-time generation tasks like text completion or dialogue generation.

- **Applications**:
  These models are widely used in text generation, dialogue systems, code completion, and other areas requiring next-token prediction.

This causal structure is what makes models like GPT and LLAMA particularly powerful for tasks where each new token is dependent on the preceding tokens and where the output must be generated in a specific order, without relying on future context.

---
## Reinforcement Learning with Human Feedback (RLHF)

### Conceptual Framework

Reinforcement Learning with Human Feedback (RLHF) is a process where human preferences are integrated into the training of language models. This allows models to generate outputs that are more coherent, relevant, and aligned with user intentions. By using human feedback, the model can be fine-tuned to reflect specific preferences, improving its performance in tasks like text generation, dialogue, and more.

#### Steps in RLHF

1. **Training a Reward Model (RM)**  
   In this initial step, human evaluators rank the model’s outputs based on criteria such as relevance, coherence, and usefulness. These rankings are used to train a Reward Model (RM), which is designed to predict a scalar reward score for each output. The reward score helps the model understand how well its outputs align with human preferences.

2. **Optimizing the Language Model with Reinforcement Learning**  
   The language model is fine-tuned using **Reinforcement Learning**, specifically with the **Proximal Policy Optimization (PPO)** algorithm. In this step, the RM serves as the reward function, guiding the model to maximize the expected reward. The RL fine-tuning helps the model improve by directly optimizing its outputs based on human feedback.

   **Objective Function**:  
   The objective of PPO in RLHF is to maximize the expected reward, as shown in the following equation:

   <p align="center">
     <b>max</b><sub>&#x3B8;</sub> <i>E</i><sub>x &#8764; &#x3C0;<sub>&#x3B8;</sub></sub>[R(x)]
   </p>

   Where:
   - $\pi_\theta$ is the **policy** (the model),
   - $R(x)$ is the **reward** for a given output $x$,
   - $\theta$ represents the model parameters.

   Additionally, we can use a **learning rate (LR)** to further optimize the results. The learning rate helps control how quickly the model adapts to the feedback and fine-tunes its parameters. Optimizing the learning rate is critical to ensure the model learns efficiently without making overly large or small updates to its weights, which can impact the training stability and performance.

### Challenges

- **Computational Cost**:  
  Training language models with human feedback is computationally expensive and time-consuming. This is because RLHF requires collecting human feedback, training a reward model, and performing several rounds of fine-tuning, each involving intensive computations.

- **Balancing Fluency and Adherence to Human Preferences**:  
  A major challenge in RLHF is achieving the right balance between fluency (i.e., how natural the model’s outputs sound) and adherence to human preferences. In some cases, improving alignment with human feedback may result in outputs that are less fluent or more constrained.

### Practical Implementation

- **Libraries and Tools**:  
  Hugging Face’s [TRL](https://github.com/huggingface/trl) provides efficient tools for implementing RLHF workflows, helping researchers and developers fine-tune models with human feedback. These tools simplify the process of training a reward model, optimizing using PPO, and integrating the feedback into the language model.

- **Example Use Case**:  
  In practice, RLHF can be applied to fine-tune models like GPT-2. In this case, human feedback is collected to rank the model’s outputs, a reward model is trained to predict these rankings, and then PPO is used to fine-tune the GPT-2 model. This process aligns the model's behaviour with human preferences, resulting in a more useful and contextually appropriate model for real-world applications.

---

## Proximal Policy Optimization (PPO): A Core Algorithm for RLHF

### Overview

Proximal Policy Optimization (PPO) is a reinforcement learning algorithm widely used for fine-tuning models with human feedback, as it provides a balance between simplicity and effectiveness. PPO stabilizes training by limiting the size of policy updates, making it ideal for tasks where small changes in policy can lead to significant improvements or setbacks.

### Key Features

- **Clipped Objective Function**:  
  PPO uses a clipped objective function to prevent large, destabilizing updates. The objective function is given by:

  $$
  \mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) A_t\right)\right]
  $$

  Where:
  - $r_t(\theta)$ is the probability ratio between the new and old policies.
  - $\epsilon$ is a hyperparameter that controls how much the policy can change between updates.
  - $A_t$ is the advantage function, indicating how much better or worse an action is compared to the average.

- **Entropy Regularization**:  
  PPO includes **entropy regularization**, which encourages exploration by preventing the policy from becoming too deterministic. This promotes diversity in the model’s responses, ensuring that it explores different options and avoids overfitting to particular behaviours.

- **Learning Rate Optimization**:  
  The learning rate (LR) plays a crucial role in PPO’s performance. By adjusting the learning rate, the model can effectively balance fast learning with stability. A well-tuned learning rate helps optimize the model's performance, ensuring that it converges to the best policy while maintaining stability throughout training.

---

## Applications and Future Directions

### Demonstrated Use Case

In this project, RLHF is used to fine-tune a GPT-2 model. By incorporating human feedback, the model generates text that closely aligns with human preferences, ensuring that outputs are relevant and coherent. PPO, guided by the reward model, helps optimize the language model to better reflect these preferences.

### Broader Applications

- **Content Moderation**: Aligning models with ethical guidelines to ensure the generation of safe and appropriate content.
- **Education**: Developing personalized tutoring systems that adapt to the needs and preferences of individual students.
- **Healthcare**: Generating clear, patient-friendly explanations of medical information, ensuring both clarity and accuracy.

### Future Work

- **Cost-Efficient Alternatives**:  
  Researchers are exploring hybrid methods that combine supervised fine-tuning with lightweight RL techniques to reduce the computational burden associated with RLHF while maintaining high-quality outputs.

- **Scaling to Larger Models**:  
  RLHF can be applied to large-scale models like GPT-4 and LLAMA 2. However, scaling RLHF to these models introduces challenges related to efficiency, resource utilization, and maintaining the quality of human alignment. Investigating efficient ways to scale RLHF will be key to improving model performance in diverse domains.

---


## References
- [CleanRL's PPO Implementation](https://github.com/vwxyzjn/cleanrl/tree/master) offers a minimalistic implementation.
- Hugging Face’s [Quickstart Guide](https://huggingface.co/blog/rlhf) provides practical insights into PPO with RLHF.

- [Self-Attention in NLP](https://www.youtube.com/watch?v=5vcj8kSwBCY)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Attention? Attention!](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html)
- [Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf)
1. Vaswani, A., et al. (2017). Attention Is All You Need. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
2. OpenAI. (2022). Training language models to follow instructions with human feedback. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
3. Hugging Face. (2023). Reinforcement Learning with Human Feedback. [Blog Post](https://huggingface.co/blog/rlhf)
4. Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
5. Anthropic. (2022). Constitutional AI: Harmlessness from AI Feedback. [arXiv:2204.05862](https://arxiv.org/abs/2204.05862)
