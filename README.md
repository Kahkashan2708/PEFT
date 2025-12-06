# Parameter Efficient Fine Tuning (PEFT)

Parameter-Efficient Fine-Tuning (PEFT) refers to a set of techniques that fine-tune large pretrained models by training only a small number of additional parameters, instead of updating the entire model.

Large language models (LLMs) such as BERT, GPT, T5, and LLaMA contain millions or billions of parameters. Fully fine-tuning them is costly, slow, and memory-intensive. PEFT offers a much cheaper alternative by freezing the main model and learning only task-specific parameters.


<img src="https://github.com/user-attachments/assets/03ed8edf-1d7d-432c-81a8-60a29b4c0439" width="300">
<img src="https://github.com/user-attachments/assets/e69a882b-7399-4ce7-a179-0c2d122b44c8" width="300">




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

## Why PEFT Works?

Large pretrained models already learn rich general-purpose representations from massive datasets. For downstream tasks, the model only needs small adjustments to adapt to:

* Task-specific patterns
* Domain knowledge
* New instructions
* New label distributions
These adjustments can be captured by a small number of additional parameters without modifying the entire model.

4. Types of PEFT Methods
