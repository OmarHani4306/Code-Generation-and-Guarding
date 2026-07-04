<div align="center">
  <h1>🛡️ Code Generation & Guarding: LLM Alignment</h1>
  
  <p>
    An experimental deep dive into aligning Large Language Models (LLMs) balancing code generation capabilities with behavioral safety controls.
  </p>

<p>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Hugging_Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face" />
  <img src="https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=for-the-badge&logo=weightsandbiases&logoColor=white" alt="WandB" />
</p>
</div>

---

## 📖 Project Overview

This repository contains the implementation for **Lab #4: Code Generation and Guarding** for the NLP course at Alexandria University. 

The fundamental challenge in modern LLM development is balancing capability with control. This project explores three distinct post-training pipelines to observe how different training paradigms impact model compliance, skill acquisition, and safety:
1. **Full Fine-Tuning (FFT)** on a small encoder model.
2. **Parameter-Efficient Fine-Tuning (PEFT)** for skill acquisition (Supervised Fine-Tuning - SFT).
3. **Direct Preference Optimization (DPO)** for behavioral alignment.

## 🧰 Tech Stack & Environment

* **Frameworks:** `transformers`, `peft`, `trl`, `bitsandbytes`
* **Hardware:** Optimized for a single Google Colab T4 GPU (16GB VRAM) / Kaggle.
* **Logging:** Weights & Biases (wandb) for tracking loss curves and alignment metrics.

---

## 🔬 Methodology & Training Phases

### Part I: The Baseline (Full Fine-Tuning)
Implementing a traditional FFT approach to establish a robust, unaligned baseline.

* **Base Models:** `FacebookAI/xlm-roberta-base` (0.3B) or `xlm-roberta-large` (0.6B)
* **Architecture:** Automatically attached Language Modeling (LM) head.

| Parameter | Value |
|---|---|
| Per Device Train Batch Size | 4 |
| Gradient Accumulation Steps | 1 |
| Learning Rate | 5e-5 |
| Optimizer | adamw_torch |
| Precision | bf16 |
| Gradient Checkpointing | True |
| Epochs | 2 |

### Part II: Skill Acquisition (SFT with Q-LoRA)
Teaching the model the specific skill of generating functional, compilable Python code using Parameter-Efficient Fine-Tuning.

* **Base Model:** `Qwen/Qwen2-1.5B-Instruct` (1.5B parameters)
* **Quantization:** 4-bit Q-LoRA (`nf4` quantization type) via `BitsAndBytesConfig`
* **Dataset:** `flytech/python-codes-25k` (>2000 samples)

| Parameter | Value |
|---|---|
| LoRA R | 16 |
| Target Modules | all-linear |
| Per Device Train Batch Size | 1 |
| Gradient Accumulation Steps | 4 |
| Learning Rate | 2e-4 |
| Optimizer | paged_adamw_8bit |
| Max Length | 1024 |
| Epochs | 1 |

### Part III: Behavioral Alignment (DPO)
Correcting dangerous compliance learned during the SFT phase. The model learns when to refuse a request or correct an unsafe instruction, even if the code itself would function perfectly.

* **Dataset:** `jondurbin/truthy-dpo-v0.1` (>4000 samples)
* **Objective Function:**

$$ \mathcal{L}_{DPO}(\pi_\theta; \pi_{ref}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right] $$

| Parameter | Value |
|---|---|
| Beta ($\beta$) Test Values | 0.1, 0.5, 0.8, 1.0 |
| Per Device Train Batch Size | 1 |
| Gradient Accumulation Steps | 4 |
| Learning Rate | 1e-5 |
| Max Prompt Length | 512 |
| Epochs | 3 |
| Gradient Checkpointing | True |

---

## ⚙️ Technical Implementation Notes

* **Completion-Only Training:** During the SFT phase, the loss is computed *only* on the model's generated answers, masking the system prompt and query.
* **Learning Rate Scheduling:** Implements warm-up and cool-down phases to ensure training stability and optimal convergence for pre-trained models.
* **Experiment Tracking:** All training runs ($\mathcal{L}_{SFT}$ and $\mathcal{L}_{DPO}$) are logged via `wandb` to analyze the impact of different $\beta$ values on the alignment process.

## 🎓 Academic Context

**Alexandria University** | Faculty of Engineering | Computer and Systems Department
* **Course:** CSE: NLP (Dec 2025)
* **Supervision:** Dr. Ayman Khalafallah
* **Teaching Assistants:** Eng. Hossam Elkordi, Eng. Ahmed Sakr, Eng. Sajed Almorsy
