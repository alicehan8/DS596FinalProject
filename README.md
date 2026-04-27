# EpiModX


## 🚀 Abstract

Epigenetic modifications play a vital role in the pathogenesis of human diseases, particularly neurodegenerative disorders such as Alzheimer's disease (AD), where dysregulated histone modifications are strongly implicated in disease mechanisms. While recent advances underscore the importance of accurately identifying these modifications to elucidate their contribution to AD pathology, existing computational methods remain limited by their generic approaches that overlook disease-specific epigenetic signatures.
To bridge this gap, we developed a novel large language model (LLM)-based deep learning framework tailored for disease-contextual prediction of histone modifications and variant effects. Focusing on AD as a case study, we integrated epigenomic data from multiple patient samples to construct a comprehensive, disease-specific histone modification dataset, enabling our model to learn AD-associated molecular signatures. A key innovation of our approach is the incorporation of a Mixture of Experts (MoE) architecture, which effectively distinguishes between disease and healthy epigenetic states, allowing for precise identification of AD-relevant epigenetic modification patterns. Our model demonstrates robust performance in disease-specific histone modification prediction, achieving mean area under receiver-operating curves (AUROCs) ranging from 0.7863 to 0.9142, significantly outperforming existing state-of-the-art methods that lack disease context. Beyond accurate modification site prediction, our framework provides important biological insights by successfully prioritizing AD-associated genetic variants, which show significant enrichment in disease-relevant pathways, supporting their potential functional role in AD pathogenesis. These findings suggest that differential modification loci identified by our model may represent key regulatory elements in AD.
Our framework establishes a powerful new paradigm for epigenetic research that can be extended to other complex diseases, offering both a valuable tool for variant effect interpretation and a promising strategy for uncovering novel disease mechanisms through epigenetic profiling.


![Framework](https://github.com/user-attachments/assets/a97d4f5f-06dc-4d20-bdb9-d93c1dc19bdc)


## 🔧 Installation

Clone the repository:

```bash
git clone https://github.com/yanwu20/EpiModX.git
cd EpiModX

conda create -n EpiModX python=3.9
conda activate EpiModX
pip install -r requirements.txt

```

## ✅ Requirements

• Python = 3.9

• numpy

• pandas

• scikit-learn

## 📥 Downloading the Human Reference Genome

🧬 The human reference genome (e.g., GRCh38) can be downloaded from GENCODE and ENCODE (https://www.encodeproject.org/data-standards/reference-sequences).

## ✅ Training

To start training the model, run:

```bash
python train_MTL_Moe.py -w True --save_model True 

```

## ✅ Prediction
To run all tests, execute:

```bash
python test_MTL_Moe.py -h H3K27ac

```
## ✅ Contact

For any question and comment regarding the code, please reach out to xiaoyu.wang2@monash.edu
