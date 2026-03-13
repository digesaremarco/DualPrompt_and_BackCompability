# DualPrompt: Rehearsal-Free Continual Learning and Backward-Compatible Representations

## Overview

This notebook implements **DualPrompt**, a rehearsal-free continual learning framework designed to preserve **backward compatibility** in feature embeddings of a pre-trained ViT backbone.  

DualPrompt separates task-invariant and task-specific knowledge using two types of prompts:

- **G-Prompt (General Prompt):** Encodes task-agnostic instructions shared across all tasks.  
- **E-Prompt (Expert Prompt):** Encodes task-specific instructions associated with a learnable task key.  

Prompts are attached to multiple self-attention layers, allowing sequential task learning without storing past data, while preserving the structure of learned embeddings.

---

## Theory

The key idea is to guide the backbone with lightweight prompts to maintain a **stable and discriminative feature space** across tasks.  
Backward compatibility is evaluated by ensuring that embeddings from a new model can be directly compared with those from a previous model using a **query–gallery protocol**:

1. Extract query embeddings from the current model.  
2. Extract gallery embeddings from the previous model.  
3. Compute **cosine similarity** and perform **k-NN classification**.  
4. Store results in a **lower-triangular compatibility matrix**.

**Feature Alignment (optional):** A distillation loss can be added to minimize drift across consecutive tasks:

$$
\mathcal{L} = \mathcal{L}_{CE} + \lambda \left(1 - \frac{f_{\text{new}} \cdot f_{\text{old}}}{\|f_{\text{new}}\| \, \|f_{\text{old}}\|}\right)
$$

**CKA (Centered Kernel Alignment)** is also computed to quantify similarity between feature embeddings across tasks:

$$
\text{CKA}(X, Y) = \frac{\| Y^\top X \|_F^2}{\| X^\top X \|_F \, \| Y^\top Y \|_F}
$$

---

## Experiments

1. **Baseline DualPrompt:** Standard training on CIFAR-100 (10 tasks × 10 classes).  
2. **Feature Alignment via Distillation:** Adds a feature-level distillation loss to stabilize embeddings.  
3. **Out-of-Distribution Evaluation:** Evaluates embeddings on BloodMNIST to test robustness under distribution shifts.  
4. **5-Task, 20-Class Splits:** Coarser task granularity to test compatibility when increasing the number of classes per task.  

Visualizations include **compatibility matrices**, **t-SNE/PCA plots**, and **CKA heatmaps** to track embedding evolution across tasks.

---

## Key Observations

- Baseline achieves high backward compatibility across tasks (off-diagonal > 0.92).  
- Distillation further reduces embedding drift (CKA > 0.99).  
- OOD evaluation shows some shifts in the last task (~0.80 CKA), while early tasks remain aligned.  
- Coarser task granularity (5-task split) does not negatively affect compatibility.  
- Efficient: ~330k prompt parameters added to an 86M parameter backbone.  
- Prompt placement in early ViT layers preserves both local and global representations, supporting consistent embeddings across tasks.

---

## Usage

Run the notebook sequentially to:

1. Train DualPrompt on CIFAR-100.  
2. Compute query–gallery embeddings for backward compatibility.  
3. Visualize feature evolution via t-SNE, PCA, and CKA.  
4. Evaluate OOD performance and alternative task splits.  

This notebook demonstrates **rehearsal-free continual learning** while maintaining a robust and interpretable embedding space.