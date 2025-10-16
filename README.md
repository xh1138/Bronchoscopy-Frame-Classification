# Bronchoscopy Frame Classification with Contrastive Learning and Pseudo-Labeling

This repository presents the **core design and logic** behind a medical image classification project,
which aims to distinguish *informative* vs *uninformative* bronchoscopy frames.
The code is structured to highlight model design, contrastive loss, pseudo-label integration,
and feature-center updates under noisy and unlabeled conditions.

> **Note**
> This repository is a *demonstration-only* version for academic purposes.  
> It shows the algorithmic and structural design, not a fully runnable implementation.  
> All data paths and lab-specific components have been removed for confidentiality.

---

### 🔍 Core Ideas
- Class-center guided **contrastive learning** to improve robustness under noisy supervision  
- **Pseudo-labeling** for leveraging unlabeled medical frames  
- Dynamic **class center updates** during training for representation consistency  
- Joint optimization of classification and contrastive objectives  

---

### 🧠 Project Structure
| File | Description |
|------|--------------|
| `model.py` | Defines the adapted Transformer-based classifier (simplified) |
| `train.py` | Training loop integrating class-center updates and contrastive learning |
| `evaluate.py` | Evaluation metrics (accuracy, precision, recall, F1) |
| `contrastive_center.py` | Feature-center initialization and update functions |
| `pseudo_labeling.py` | Pseudo-label generation and dataset merging logic |
| `main.py` | Entry point combining all components |
| `LICENSE` | MIT License |

---

### ⚖️ License
Released under the MIT License.
