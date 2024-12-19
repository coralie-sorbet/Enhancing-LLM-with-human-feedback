# Enhancing Large Language Models with Human Feedback via Reinforcement Learning

## Overview
This project investigates the integration of human feedback into training Large Language Models (LLMs) using Reinforcement Learning (RL). The objective is to enhance the performance and contextual relevance of Transformer-based models, such as GPT, through Reinforcement Learning with Human Feedback (RLHF).

While full-scale training with RLHF is computationally expensive, the aim of this project is to design a training pipeline that effectively demonstrates the core concepts without requiring excessive resources.

---

## Objectives

### 1. Understanding and Utilizing Transformers
- **Transformer Architecture**:
  - Transformers, introduced in [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762), rely on the self-attention mechanism to capture dependencies in sequential data. Unlike RNNs or LSTMs, they enable parallelization for efficient training.
  - Key components:
    - **Multi-Head Attention**: Captures relationships between tokens in input sequences.
    - **Feedforward Layers**: Process representations outputted by the attention mechanism.
    - **Positional Encoding**: Adds sequence information since Transformers process input as a whole.

- **Causal Transformers**:
  - Causal Transformers, like GPT and LLAMA, are autoregressive models trained with a left-to-right context. They predict the next token in a sequence using only prior context, making them ideal for language generation tasks.

### 2. Reinforcement Learning with Human Feedback
- **RLHF**: A framework to fine-tune LLMs by incorporating human preferences. The process involves:
  1. Training a reward model (RM) to quantify the quality of model outputs based on human feedback.
  2. Using RL (e.g., PPO) to optimize the LLM's outputs for higher rewards from the RM.
  3. Fine-tuning the LLM to align its responses with human expectations.

- **Challenges and Libraries**:
  - Implementing RLHF is resource-intensive but can be simplified using libraries like [Hugging Face's TRL](https://github.com/huggingface/trl).

### 3. Training a Reward Model
- Utilize GPT-2 or similar to train a reward model that scores model outputs.
- Example script: [Reward Modeling Example](https://github.com/huggingface/trl/blob/main/examples/scripts/reward_modeling.py).
- **Steps**:
  - Prepare a dataset of prompt-response pairs with associated scores.
  - Fine-tune GPT-2 to predict these scores.
  - Evaluate performance using metrics like correlation with human-labeled scores.

### 4. Optimization with Proximal Policy Optimization (PPO)
- PPO is a policy-gradient RL algorithm effective for optimizing LLMs with RLHF.
- Resources:
  - Hugging Face’s [TRL Quickstart](https://huggingface.co/blog/rlhf).
  - CleanRL’s [PPO Implementation](https://github.com/vwxyzjn/cleanrl/tree/master).

- **Workflow**:
  1. Define a reward function using the trained RM.
  2. Fine-tune the LLM using PPO to maximize rewards.
  3. Evaluate outputs for fluency, relevance, and alignment with human preferences.

---


## RewardTrainer Example

This is a basic example of how to use the `RewardTrainer` from the `trl` library to train a model.

### Code Example

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
## PPOTrainer Example

This is a basic example of how to use the `PPOTrainer` from the `trl` library for Proximal Policy Optimization (PPO).

### Code Example

```python
from trl import PPOConfig, PPOTrainer
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

# Configure PPO training parameters
ppo_config = PPOConfig(
    num_train_epochs=1,  # Number of training epochs
    batch_size=1,  # Batch size for training
    output_dir="PPO_results"  # Directory to save results
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
## Project Structure

```
project/
├── Model_training.ipynb              # Notebook with the training of the models
├── Model_testing.ipynb               # Notebook with the testing on unseen texts of the models
├── README.md                         # Overview of the project
└── REPORT.md                         # Detailed explanation
```

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


