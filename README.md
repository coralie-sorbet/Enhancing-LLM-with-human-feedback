# Enhancing Large Language Models with Human Feedback via Reinforcement Learning

## Overview
This project demonstrates how human feedback can be integrated to enhance the training of large language models (LLMs) using Reinforcement Learning (RL). The primary goal is to improve the contextual relevance and performance of Transformer-based models, such as GPT, through Reinforcement Learning with Human Feedback (RLHF).

The Reinforcement Learning with Human Feedback (RLHF) process involves several components working together to align a language model with human preferences. The key steps include:

- **Initial Model**: A pre-trained language model is the starting point.
- **Reward Model**: Built to evaluate outputs based on human preferences.
- **Reinforcement Learning**: Fine-tunes the model using techniques like **Proximal Policy Optimization (PPO)**.

The diagram below illustrates the high-level workflow of RLHF. It highlights how the initial language model is tuned iteratively using a reward model and RL techniques to align outputs with human preferences:

<div align="center">
  <img src="https://github.com/user-attachments/assets/02219518-24c9-4246-859d-9b85404da0f7" alt="RLHF Workflow" width="400">
</div>

While full-scale RLHF training can be resource-intensive, this project simplifies the process to demonstrate **core concepts** in a more **resource-efficient manner**, making it accessible for experimentation and learning.


---

## Goals
- **Enhance LLMs using human feedback**: Improve language models' alignment with human expectations.
- **Demonstrate RLHF concepts**: Provide a simplified implementation that balances performance and computational efficiency.
- **Share findings**: Highlight the impact of RLHF on LLM performance and its limitations.

---

## Methodology Overview
This project leverages **Transformers** and integrates **Reinforcement Learning with Human Feedback (RLHF)** to fine-tune large language models.  
- Transformers process sequential data with self-attention mechanisms ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)).
- RLHF aligns model outputs with human preferences via a reward model and RL techniques such as PPO.  
For more details, see [REPORT.MD](./REPORT.MD)

---

## Key Findings
- **Improved Alignment**: RLHF enables LLMs to produce responses better aligned with human preferences.
- **Trade-offs**: Computational cost remains a significant barrier for large-scale implementations.
- **Simplified Tools**: Libraries like TRL streamline RLHF processes for smaller-scale experimentation.

---

## Installation

Install the `trl` library using `pip`:

```bash
pip install trl
```

---
## Examples

### RewardTrainer Example

This is a basic example of how to use the `RewardTrainer` from the `trl` library to train a model.

#### Code Example

```python
from trl import RewardConfig, RewardTrainer
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = AutoModelForSequenceClassification.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct", num_labels=1
)
model.config.pad_token_id = tokenizer.pad_token_id

# Load dataset
dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

# Set up training arguments
training_args = RewardConfig(output_dir="Qwen2.5-0.5B-Reward", per_device_train_batch_size=2)

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
This example assumes you have a dataset and models ready.

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
- [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- [OpenAI’s RLHF for ChatGPT](https://arxiv.org/abs/2203.02155)
- [Anthropic’s Work on RLHF](https://arxiv.org/abs/2204.05862)

### Libraries
- [Hugging Face’s TRL](https://github.com/huggingface/trl)
- [CleanRL’s PPO Implementation](https://github.com/vwxyzjn/cleanrl/tree/master)

### Tutorials and Videos
- [PPO Explained](https://www.youtube.com/watch?v=5P7I-xPq8u8)
- [RLHF Blog Post](https://huggingface.co/blog/rlhf)
- [Hugging Face RLHF Course](https://www.youtube.com/watch?v=2MBJOuVq380)

