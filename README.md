# Moroccan Darija SentenceTransformer

This repository contains a [sentence-transformers](https://www.SBERT.net) model finetuned from [answerdotai/ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) on the [al-atlas-moroccan-darija-pretraining-dataset](https://huggingface.co/datasets/atlasia/AL-Atlas-Moroccan-Darija-Pretraining-Dataset). It maps sentences and paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Features
- **Fast and efficient sentence embeddings** for Moroccan Darija.
- **Finetuned from ModernBERT-base**, optimized for dialectal Arabic.
- **Supports multiple NLP tasks** including search, classification, and clustering.

## Model Details
- **Model Type:** Sentence Transformer
- **Base Model:** [answerdotai/ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base)
- **Maximum Sequence Length:** 8196 tokens
- **Output Dimensionality:** 768 dimensions
- **Similarity Function:** Cosine Similarity
- **Training Dataset:** [al-atlas-moroccan-darija-pretraining-dataset](https://huggingface.co/datasets/atlasia/AL-Atlas-Moroccan-Darija-Pretraining-Dataset)
- **Weights:** Available on the [Hugging Face hub](https://huggingface.co/atlasia/MorDernBERT-ep-1-lr-0.005)

## Installation
Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/moroccan-darija-embeddings.git
cd moroccan-darija-embeddings
pip install -r requirements.txt
```

## Usage
### Loading Pre-trained Embeddings
You can load the trained model using `sentence-transformers`:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("path/to/moroccan_darija_model")
embedding = model.encode("جملة بالدارجة")
```

## Roadmap
- ✅ Sentence Transformer model finetuned for Moroccan Darija
- ⏳ Further optimization and finetuning
- ⏳ Evaluation on downstream NLP tasks

## Currently Running
### Abdelaziz:
- Training ModernBERT from scratch on `atlasia/AL-Atlas-Moroccan-Darija-Pretraining-Dataset`
- Training ModernBERT from scratch on `wikipedia-ar`
- Finetuning Arabic BERT on `atlasia/AL-Atlas-Moroccan-Darija-Pretraining-Dataset`
- Finetuning Multilingual BERT on `atlasia/AL-Atlas-Moroccan-Darija-Pretraining-Dataset`
### Abdeljalil
✅ Continuous training XLM Roberta Large [Fill Mask Task] `atlasia/xlm-roberta-large-ft-alatlas`
  - Test space: `atlasia/Darija-Roberta-mask`
## Datasets
### Arabic
- [Wikimedia Arabic Wikipedia (Nov 2023)](https://huggingface.co/datasets/wikimedia/wikipedia/viewer/20231101.ar)

### Moroccan Arabic
- [AL-Atlas Moroccan Darija Pretraining Dataset](https://huggingface.co/datasets/atlasia/AL-Atlas-Moroccan-Darija-Pretraining-Dataset)
- [Social Media Darija Dataset](https://huggingface.co/datasets/atlasia/Social_Media_Darija_DS)

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests to improve the model and codebase.



