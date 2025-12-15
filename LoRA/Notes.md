# LoRA (Low-Rank Adaptation)

LoRA (Low-Rank Adaptation) is a Parameter-Efficient Fine-Tuning (PEFT) technique used to adapt large pre-trained models, especially Large Language Models (LLMs) and Transformer-based architectures, to new tasks without updating all model parameters. Instead of fine-tuning the full weight matrices of the model, LoRA learns small low-rank matrices that are added to the frozen pre-trained weights. This allows efficient task adaptation while keeping most of the original model unchanged.

---

## Motivation

Modern models such as GPT, BERT, and LLaMA contain billions of parameters. Fully fine-tuning these models is computationally expensive, requires large GPU memory, and is often impractical in resource-constrained environments. LoRA addresses these challenges by freezing the original model weights and training only a small number of additional parameters. Despite training far fewer parameters, LoRA often achieves performance close to full fine-tuning, making it a cost-effective and scalable solution.

---
## Mathematical Formulation of LoRA

Consider a weight matrix \( W \in \mathbb{R}^{d \times k} \) in a neural network layer.  
Instead of updating \( W \) directly during fine-tuning, LoRA represents the update as an additive low-rank decomposition:

\[
W' = W + \Delta W
\]

where

\[
\Delta W = A B
\]

Here,

\[
A \in \mathbb{R}^{d \times r}, \quad
B \in \mathbb{R}^{r \times k}
\]

and

\[
r \ll \min(d, k)
\]

which enforces a low-rank constraint.

Only the matrices \( A \) and \( B \) are trained during fine-tuning, while the original weight matrix \( W \) remains frozen.  
This significantly reduces the number of trainable parameters while still allowing effective task adaptation.

---

## Where is LoRA Applied?

LoRA is typically applied to Transformer models, particularly within the attention mechanism. It is commonly used on the projection matrices of the attention layers, including Query (Q), Key (K), Value (V), and the output projection. In practice, applying LoRA to the Query and Value matrices is often sufficient to achieve strong performance. In some cases, LoRA can also be applied to MLP (feed-forward) layers depending on the task requirements.

---
## Training Process

* Load a pre-trained model such as BERT, GPT, or LLaMA that has already been trained on a large dataset.

* Freeze all the original model weights so that they are not updated during training. This ensures that the knowledge learned during pre-training is preserved.
* Insert (inject) LoRA adapters into selected layers of the model, usually in the attention layers such as Query and Value projections.

* Start the training process. During training, only the low-rank matrices introduced by LoRA are updated, while the rest of the model remains unchanged.

* Complete training once the model learns the task-specific patterns using the LoRA parameters.

* For inference or deployment, either keep the LoRA adapters as separate modules or merge the LoRA weights with the base model weights, depending on memory and performance requirements.

---

## Advantages 

1: LoRA reduces memory usage by training only a small number of additional parameters instead of updating the entire model.

2: Because fewer parameters are trained, the training process is faster compared to full fine-tuning.

3: Reduced computation means LoRA requires less GPU power, making it suitable for low-resource environments.

4: Multiple LoRA adapters can be trained for different tasks using the same base model.

5: Switching between tasks becomes easy by loading a different LoRA adapter without retraining the full model.

---

## Limitations
1: LoRA may not always match the performance of full fine-tuning, especially for complex or highly specialized tasks.

2: If the new task is very different from the data used during pre-training, LoRA’s low-rank updates may not be sufficient.

3: Choosing an inappropriate rank value can limit the model’s ability to learn task-specific features.

4: Selecting the wrong target layers for LoRA insertion can reduce adaptation quality.

5: Careful tuning of hyperparameters is required to achieve optimal performance.

---


