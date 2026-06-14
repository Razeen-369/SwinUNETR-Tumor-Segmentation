# SwinUNETR-Tumor-Segmentation

## Setup Instructions

1. Clone the repository:
   git clone <repo-link>

2. Navigate to project:
   cd ANEES_HONOURS

3. Create virtual environment:
   python -m venv tumor_env

4. Activate environment:
   Windows:
   tumor_env\Scripts\activate

   Mac/Linux:
   source tumor_env/bin/activate

5. Install dependencies:
   pip install -r requirements.txt

6. Run the application:
   python app.py


**Automated Pituitary Tumor Segmentation using Hybrid Transformer-CNN Models**.

Pituitary tumors can be challenging to diagnose and delineate accurately from MRI scans. To support AI-assisted medical imaging, our team developed and evaluated hybrid deep learning architectures that combine the strengths of **Transformers** and **Convolutional Neural Networks (CNNs)** for automated tumor segmentation.

🔍 **What we explored**

• Swin UNETR architecture
• Custom Swin V-Net architecture
• Automated segmentation of pituitary tumors from 2D MRI scans
• Comparative performance analysis on the BRISC 2025 MRI dataset

📊 **Key Results**

**Swin UNETR**

✅ Dice Score: **0.8349**
✅ Validation Accuracy: **99.5%**
✅ Precision: **0.82**
✅ Recall: **0.85**
✅ F1-Score: **0.835**
✅ ROC-AUC: **0.96**

**Custom Swin V-Net**

✅ Dice Score: **0.8153**
✅ Validation Accuracy: **98.0%**
✅ Precision: **0.80**
✅ Recall: **0.83**
✅ F1-Score: **0.815**
✅ ROC-AUC: **0.97**

These results demonstrate the potential of hybrid Transformer-CNN architectures for accurate and reliable MRI-based tumor segmentation, contributing to the advancement of AI-driven healthcare solutions.
