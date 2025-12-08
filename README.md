# Parameter Efficient Fine Tuning (PEFT)

Parameter-Efficient Fine-Tuning (PEFT) refers to a set of techniques that fine-tune large pretrained models by training only a small number of additional parameters, instead of updating the entire model.

Large language models (LLMs) such as BERT, GPT, T5, and LLaMA contain millions or billions of parameters. Fully fine-tuning them is costly, slow, and memory-intensive. PEFT offers a much cheaper alternative by freezing the main model and learning only task-specific parameters.

![PEFT Image](https://github.com/user-attachments/assets/03ed8edf-1d7d-432c-81a8-60a29b4c0439)


## Core Idea of PEFT

The idea is simple:
* Freeze the original pretrained model.
* Add small, lightweight trainable components.
* Train only these new components.
* Keep the base model unchanged.

This results in:
* Reduced memory usage
* Faster training
* Fewer trainable parameters
* Easier storage of multiple task-specific models
![PEFT Diagram](https://github.com/user-attachments/assets/e69a882b-7399-4ce7-a179-0c2d122b44c8)

## Why PEFT Works?

Large pretrained models already learn rich general-purpose representations from massive datasets. For downstream tasks, the model only needs small adjustments to adapt to:

* Task-specific patterns
* Domain knowledge
* New instructions
* New label distributions
These adjustments can be captured by a small number of additional parameters without modifying the entire model.

## Types of PEFT Methods
1. LoRA (Low-Rank Adaptation)
2. Adapter Tuning
3. Prefix Tuning
4. Prompt Tuning (Soft Prompt Tuning)
5. QLoRA (Quantized LoRA)
---

## 1. LoRA (Low-Rank Adaptation)

LoRA introduces a low-rank update to the model’s weight matrices while keeping the original weights frozen.  
Instead of updating the full weight matrix W, LoRA learns two small matrices A and B such that:

W' = W + BA

### Key Points:
- Reduces trainable parameters by orders of magnitude.
- Base model remains unchanged; only LoRA layers are trained.
- Works exceptionally well for large language models.
- Easy to merge trained LoRA weights into the main model.
- Reduces GPU memory usage and speeds up training.

---

## 2. Adapter Tuning

Adapter tuning inserts small bottleneck layers (adapters) inside each transformer block.  
During fine-tuning, the original model parameters are frozen and only the adapter weights are trained.

### Key Points:
- Modular approach: multiple adapters for multiple tasks.
- Helps avoid catastrophic forgetting.
- Lightweight compared to full fine-tuning.
- Effective for multilingual and multi-task learning.
- Does not modify pretrained model weights.

---

## 3. Prefix Tuning

Prefix tuning prepends a sequence of learnable key–value vectors to the attention layers of a transformer.  
These prefixes influence the model’s behavior without modifying its internal parameters.

### Key Points:
- Very parameter-efficient.
- Particularly effective for text generation tasks.
- Adds trainable prefixes to each transformer layer.
- Does not require changing or replacing model weights.
- Works well for tasks such as summarization and dialogue.

---

## 4. Prompt Tuning (Soft Prompt Tuning)

Prompt tuning learns continuous prompt embeddings that are added to the input sequence.  
The model remains frozen, and only the prompt vectors are optimized.

### Key Points:
- Suitable for classification and instruction-based tasks.
- Extremely small number of trainable parameters.
- Influences model behavior through learned prompts.
- Works across different model sizes.
- Often used when training data is limited.

---

## 5. QLoRA (Quantized LoRA)

QLoRA combines 4-bit quantization with LoRA to enable fine-tuning of very large LLMs on low-resource hardware.  
The base model is stored in 4-bit precision, while LoRA layers are trained normally.

### Key Points:
- Allows fine-tuning models like LLaMA-2 and LLaMA-3 on a single GPU.
- Maintains performance comparable to full fine-tuning.
- Reduces memory footprint using NF4 (NormalFloat4) quantization.
- Only LoRA layers are trained; quantized model weights stay frozen.
- Ideal for large-scale instruction tuning and domain adaptation.

---
