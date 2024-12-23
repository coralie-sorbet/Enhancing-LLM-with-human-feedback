# Enhancing Large Language Models with Human Feedback via Reinforcement Learning

## Overview
This project demonstrates how human feedback can be integrated to enhance the training of large language models (LLMs) using Reinforcement Learning (RL). The primary goal is to improve the contextual relevance and performance of Transformer-based models, such as GPT, through Reinforcement Learning with Human Feedback (RLHF).

The Reinforcement Learning with Human Feedback (RLHF) process involves several components working together to align a language model with human preferences. The key steps include:

- **Initial Model**: A pre-trained language model is the starting point.
- **Reward Model**: Built to evaluate outputs based on human preferences.
- **Reinforcement Learning**: Fine-tunes the model using techniques like **Proximal Policy Optimization (PPO)**.

The diagram below, taken from the [RLHF Blog Post](https://huggingface.co/blog/rlhf), illustrates the high-level workflow of RLHF. It highlights how the initial language model is tuned iteratively using a reward model and RL techniques to align outputs with human preferences:

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

**Improved Alignment**  
RLHF enables LLMs to produce responses better aligned with human preferences.

**Trade-offs**  
Computational cost remains a significant barrier for large-scale implementations.

**Simplified Tools**  
Libraries like TRL streamline RLHF processes for smaller-scale experimentation.

The reward model achieved the following evaluation results:

- **`eval_loss`**: **0.6931**  
  This represents the cross-entropy loss during evaluation. A lower loss indicates better performance, though the ideal value depends on the specific task and dataset.

- **`eval_accuracy`**: **0.7473**  
  The model correctly classified **74.73%** of the evaluation samples. While this is a good starting point, there's still room for improvement.

**Comparison of Scores**  
- **Average Reward Model Score**: **-0.5434042** (evaluated on the entire training dataset)  
- **Average PPO Model Score**: **0.33177295** (evaluated on 10% of the training dataset for memory and time considerations, with potential for improvement if the sample size is increased).

The PPO model demonstrates a significant improvement in reward scores compared to the reward model. This suggests that optimizing the policy with Proximal Policy Optimization (PPO) enables the model to better align with the desired reward function, even when evaluated on a smaller subset of the data. This indicates that PPO-optimized policies not only adhere more closely to human preferences but also show promise for scalability with further training.

### Visualization and Interpretation of Scores

Below, a barplot and a boxplot  illustrate the comparative scores of the Reward Model and PPO Model.

<img width="422" alt="image" src="https://github.com/user-attachments/assets/3548f5c1-e134-4758-916e-f9f84a010b83" />

<img width="416" alt="image" src="https://github.com/user-attachments/assets/e592cbb4-72a9-453b-bc9f-84454466a7cf" />

#### Boxplot: Comparison of Scores
The boxplot provides a clear comparison of the score distributions for the Reward Model and PPO Model. It highlights several key insights:
- The Reward Model has a median score below zero, indicating a general tendency toward lower alignment with the reward function. Additionally, the spread of its scores is narrower, with significant clustering around negative values and a few outliers in both directions.
- In contrast, the PPO Model shows a positive median score, suggesting a noticeable improvement in alignment with the reward function. Its broader interquartile range reflects greater variability, which may result from exploring diverse policies during optimization.

The improvement in median and overall distribution suggests that PPO successfully optimizes the policy toward producing outputs more aligned with human preferences.

#### Histogram: Distribution of Scores
The histogram complements the boxplot by visualizing the frequency distribution of scores for both models:
- The Reward Model's scores are concentrated around negative values, with a peak close to -1. This indicates that many samples do not meet the desired reward function criteria.
- The PPO Model's scores shift toward positive values, with a peak near zero and a heavier tail on the positive side. This demonstrates that PPO optimization leads to better overall alignment with the reward function for a larger portion of the dataset.

Together, these plots emphasize that PPO-optimized policies not only outperform the baseline Reward Model in terms of alignment but also exhibit a broader and more promising distribution of scores. This reinforces the value of PPO in improving the effectiveness of RLHF.

**Model Sizes**  
- **Reward Model**: **479.31 MB**  
- **PPO-optimized Policy Model**: **474.72 MB**
- 
The slightly smaller size of the PPO model highlights its efficiency while retaining strong performance.


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

