# Arabic to English Neural Machine Translation

A neural machine translation system that translates Arabic text to English using a Transformer-based architecture.

## Overview

This project implements a Transformer model for Arabic to English translation. It includes both the training pipeline and a web interface built with Streamlit for easy interaction with the model.

## Features

- Neural machine translation using Transformer architecture
- Streamlit-based web interface
- SentencePiece tokenization for both Arabic and English
- Support for both CPU and GPU inference
- Interactive example phrases for testing

## Technical Details

### Model Architecture
- Transformer model with 4 encoder and 4 decoder layers
- 8 attention heads
- Embedding dimension: 256
- Feedforward dimension: 512
- Dropout rate: 0.1
- Vocabulary size: 8000 tokens
- Maximum sequence length: 200 tokens

### Requirements

- Python 3.11+
- PyTorch
- Streamlit
- SentencePiece
- Additional dependencies in `requirements.txt`

## Installation

1. Clone the repository
```bash
git clone https://github.com/saadsohail05/Arabic-To-English-NMT-Using-Transformers.git
cd Arabic-To-English-NMT-Using-Transformers
```

2. Create and activate a virtual environment
```bash
python -m venv venv_py311
source venv_py311/bin/activate  # On Windows, use `venv_py311\Scripts\activate`
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Running the Web Interface

Start the Streamlit application:
```bash
streamlit run app.py
```

The web interface will be available at `http://localhost:8501`

### Using the Model Programmatically

```python
from model import initialize_transformer, translate, decode_sentence
import torch
import sentencepiece as spm

# Load model and tokenizers
model = initialize_transformer()
model.load_state_dict(torch.load('ar_en_transformer.pt'))
ar_sp = spm.SentencePieceProcessor()
ar_sp.load('ar_tokenizer.model')

# Translate text
input_text = "مرحبا بالعالم"
translated_text = translate(model, input_text)
```

## Model Training

The model was trained on the Tatoeba Arabic-English parallel corpus. Training details and notebooks can be found in the `Notebooks` directory.

## Project Structure

```
.
├── app.py                  # Streamlit web application
├── model.py               # Model architecture and utilities
├── ar_en_transformer.pt   # Trained model weights
├── ar_tokenizer.model     # Arabic SentencePiece tokenizer
├── en_tokenizer.model     # English SentencePiece tokenizer
├── requirements.txt       # Project dependencies
└── Notebooks/            # Training notebooks and experiments
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Helsinki-NLP for the Tatoeba MT dataset
- The PyTorch team for the deep learning framework
- The Streamlit team for the web application framework
