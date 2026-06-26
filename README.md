# Hybrid CNN Transformer and BiLSTM CRF Pipeline for Text Restoration in Tamil Palm Leaf Manuscripts

Two deep learning models for Tamil manuscript recognition and text restoration.

## Models

### 1. Hybrid CNN-Transformer (`hybrid_cnn_transformer.py`)
Combines a CNN feature extractor with a Transformer encoder for image classification.
- Input: grayscale images (64×64)
- Output: class logits (125 classes)

### 2. BiLSTM-CRF for Tamil Diacritic & Space Restoration (`bilstm_crf_tamil.py`)
Restores diacritics and spaces in raw Tamil character sequences.
- Input: raw Tamil character sequence (no diacritics/spaces)
- Output: formatted Tamil text

## Requirements
pip install -r requirements.txt

## Usage

```python
# CNN-Transformer
from hybrid_cnn_transformer import HybridCNNTransformer
model = HybridCNNTransformer(num_classes=125)

# BiLSTM-CRF
from bilstm_crf_tamil import BiLSTMCRF, TamilProcessor, infer
```
