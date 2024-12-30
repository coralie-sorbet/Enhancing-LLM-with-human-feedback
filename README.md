# Enhancing Large Language Models with Human Feedback via Reinforcement Learning

## Overview and Goals

This project explores how human feedback can enhance the training of large language models (LLMs) using Reinforcement Learning (RL), specifically through Reinforcement Learning with Human Feedback (RLHF). The goal is to improve the contextual relevance and alignment of Transformer-based models, such as GPT, with human preferences.

The RLHF process integrates several key components to achieve this alignment:

- **Initial Model**: A pre-trained language model is the starting point.  
- **Reward Model**: Evaluates outputs based on human feedback to establish a training signal.  
- **Reinforcement Learning**: Fine-tunes the model using advanced techniques like **Proximal Policy Optimization (PPO)**.  

The diagram below, adapted from the [RLHF Blog Post](https://huggingface.co/blog/rlhf), illustrates the high-level workflow of RLHF, showing how the initial language model is iteratively tuned using a reward model and reinforcement learning to produce outputs that better align with human preferences:

<div align="center">
  <img src="https://github.com/user-attachments/assets/02219518-24c9-4246-859d-9b85404da0f7" alt="RLHF Workflow" width="400">
</div>

While RLHF training is typically resource-intensive, this project simplifies the process to balance performance and computational efficiency. By doing so, it provides an accessible demonstration of RLHF concepts, enabling experimentation and learning while showcasing the benefits and trade-offs of this approach.

---
## Key Findings

Reinforcement Learning with Human Feedback (RLHF) plays a crucial role in making large language models (LLMs) more responsive to human needs. By combining a reward model with Proximal Policy Optimization (PPO), RLHF transforms generic pre-trained models into systems that generate outputs more closely aligned with user preferences and expectations.

However, RLHF is not without its challenges. The process can be computationally intensive, particularly when applied at scale. This difficulty highlights the importance of developing more efficient approaches, especially for researchers and developers working with limited resources or large datasets. Tools like TRL have made RLHF more accessible, providing streamlined workflows for smaller-scale experiments and lowering the barriers to exploring this transformative methodology.

Despite these obstacles, RLHF has shown remarkable success. Models fine-tuned using this approach consistently deliver improved performance, proving its value in real-world applications. Whether used in chatbots, content creation, or other interactive AI systems, RLHF offers a promising path toward creating AI that genuinely understands and meets human expectations.

### Reward Model Evaluation

The reward model's evaluation is based on two key metrics: `eval_loss` and `eval_accuracy`. These metrics provide insights into the model's performance in terms of prediction quality and classification accuracy.

- **`eval_loss`**: **0.6931**  
  This cross-entropy loss value during evaluation reflects the model's ability to predict outputs. A lower value generally indicates better performance, though the optimal loss depends on the specific dataset and task.

- **`eval_accuracy`**: **0.7473**  
  The reward model correctly classified 74.73% of the evaluation samples. While this is a promising baseline, further optimization is possible.

### Comparative Scores
- **Average Reward Model Score**: **-0.5434042** (evaluated on the entire training dataset).  
- **Average PPO Model Score**: **0.33177295** (evaluated on 10% of the training dataset due to memory and time constraints).  

The significant improvement in the PPO model’s scores underscores its ability to optimize policy alignment with human preferences effectively. Even when evaluated on a smaller dataset, PPO demonstrates superior performance, suggesting that it is not only more efficient but also scalable with further training.


### Visualization and Interpretation of Scores

Below, a boxplot and a histogram illustrate the comparative scores of the Reward Model and PPO Model:

<p align="center">
  <img width="422" alt="image" src="https://github.com/user-attachments/assets/e592cbb4-72a9-453b-bc9f-84454466a7cf" />
  <img width="422" alt="image" src="https://github.com/user-attachments/assets/3548f5c1-e134-4758-916e-f9f84a010b83" />
</p>

The boxplot highlights key differences in the score distributions of the two models:
- The **Reward Model** exhibits a median score below zero, indicating a general tendency toward lower alignment with the reward function. Its narrower spread suggests limited variability, with notable clustering around negative values.
- The **PPO Model**, by contrast, shows a positive median score, reflecting substantial improvement in alignment with the reward function. Its broader interquartile range indicates greater exploration and diversity in outputs during optimization.

The improved median and distribution for the PPO model underscore its capability to generate outputs better aligned with human preferences.

The histogram provides a deeper look at score distributions:
- **Reward Model**: Scores are concentrated around negative values, with a peak near -1. This suggests limited alignment with human preferences for most samples.
- **PPO Model**: Scores shift toward positive values, with a peak near zero and a heavier tail on the positive side. This shift indicates a better overall alignment of PPO-optimized policies with the reward function.

Together, these visualizations emphasize that PPO not only outperforms the baseline Reward Model but also demonstrates a promising ability to generalize and scale effectively.

### Model Sizes
- **Reward Model**: **479.31 MB**  
- **PPO-optimized Policy Model**: **474.72 MB**

The slightly smaller size of the PPO model highlights its computational efficiency without compromising performance, making it an attractive choice for large-scale implementations.

---
## Installation

Install the `trl` library using `pip`:

```bash
pip install trl
```

---
## Examples

### RewardTrainer Example

This is a basic example of using the `RewardTrainer` from the `trl` library to train a model.

#### Code Example

```python
from trl import RewardConfig, RewardTrainer
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load pre-trained model and tokenizer
model_name = "gpt2" 
model_config = ModelConfig(model_name_or_path=model_name)
tokenizer = AutoTokenizer.from_pretrained(
    model_config.model_name_or_path,
    trust_remote_code=model_config.trust_remote_code,
    use_fast=True
)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForSequenceClassification.from_pretrained(
    model_config.model_name_or_path, num_labels=1, trust_remote_code=model_config.trust_remote_code
)
model.config.pad_token_id = tokenizer.pad_token_id

# Load dataset
dataset = load_dataset("trl-lib/ultrafeedback_binarized")

# Set up training arguments
training_args = RewardConfig(output_dir="Reward", per_device_train_batch_size=2)

# Initialize and train the RewardTrainer
trainer = RewardTrainer(
    args=training_args,
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
)

trainer.train()
```

### PPOTrainer Example

Here’s a minimal example of using `PPOTrainer` from Hugging Face’s `trl` library to fine-tune an LLM using Proximal Policy Optimization (PPO).  

#### Code Example

```python
from trl import PPOConfig, PPOTrainer
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Configure PPO training parameters
ppo_config = PPOConfig(
    num_train_epochs=1,         # Number of training epochs
    batch_size=1,               # Batch size for training
    output_dir="PPO_results"    # Directory to save results
)

# Initialize PPOTrainer
trainer = PPOTrainer(
    config=ppo_config,
    policy=policy_model,    # Policy model to optimize
    tokenizer=tokenizer,    # Tokenizer for text processing
    train_dataset=dataset,  # Training dataset
)

# Start training
trainer.train()
```

---

## Project Structure

```
project/
├── src/                     # Folder containing the notebook with the training of the models
├── test/                    # Folder containing the notebook with the testing on unseen texts of the models
├── README.md                # Overview of the project
└── REPORT.md                # Detailed explanation
```

---

## Resources and References

### Papers
- [Anthropic’s Work on RLHF](https://arxiv.org/abs/2204.05862)
- [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- [OpenAI’s RLHF for ChatGPT](https://arxiv.org/abs/2203.02155)

### Libraries
- [CleanRL’s PPO Implementation](https://github.com/vwxyzjn/cleanrl/tree/master)
- [Hugging Face’s TRL](https://github.com/huggingface/trl)

### Tutorials and Videos
- [Attention? Attention!](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html) - A blog post explaining attention mechanisms in detail.
- [Hugging Face RLHF Course](https://www.youtube.com/watch?v=2MBJOuVq380)
- [PPO Explained](https://www.youtube.com/watch?v=5P7I-xPq8u8)
- [RLHF Blog Post](https://huggingface.co/blog/rlhf)
- [Self-Attention in NLP (YouTube)](https://www.youtube.com/watch?v=5vcj8kSwBCY) - Video explanation of self-attention in NLP.
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - A visual guide to understanding the transformer architecture.
