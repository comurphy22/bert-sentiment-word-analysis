# BERT-Based NLP Applications

A collection of Natural Language Processing (NLP) applications leveraging BERT (Bidirectional Encoder Representations from Transformers) for various text analysis tasks including word analogies, sentiment analysis, and aspect-based sentiment analysis.

## Overview

This project demonstrates the versatility of transformer-based language models through three distinct NLP applications:

1. **Word Analogy Solver**: Uses BERT embeddings to solve word analogy problems (e.g., "man:woman::king:?")
2. **Amazon Review Sentiment Analysis**: Predicts sentiment ratings from product reviews using two different approaches
3. **Aspect-Based Sentiment Analysis**: Identifies and analyzes sentiment toward specific aspects in restaurant/laptop reviews

## Features

- 🎯 **Multiple NLP Tasks**: Word analogies, general sentiment analysis, and aspect-level sentiment analysis
- 🤖 **BERT Integration**: Leverages pre-trained BERT models from Hugging Face Transformers
- 📊 **Comparative Analysis**: Implements both feature extraction and fine-tuning approaches
- 🔄 **GPU Support**: Automatically uses CUDA when available for faster processing
- 📈 **Comprehensive Metrics**: Includes accuracy, F1-score, classification reports, and confusion matrices

## Requirements

### Python Version

- Python 3.8 or higher

### Dependencies

Install all required packages using the provided requirements.txt file:

```bash
pip install -r requirements.txt
```

This will install:

- PyTorch (deep learning framework)
- Transformers (Hugging Face BERT models)
- NumPy and Pandas (data processing)
- Scikit-learn (machine learning utilities)
- SciPy (scientific computing)
- Matplotlib and Seaborn (visualization)
- tqdm (progress bars)
- requests (HTTP library for dataset downloads)

## Project Structure

```
deeplearning-P2/
├── word_analogies/
│   ├── word_analogy_solver.py    # Word analogy solver
│   └── word-test.v1.txt          # Word analogy dataset
├── review_sentiment/
│   ├── review_sentiment_classifier.py  # Amazon review sentiment analysis
│   └── amazon_reviews.csv              # Amazon reviews dataset
├── aspect_sentiment_analysis.py  # Aspect-based sentiment analysis
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

## Usage

### Word Analogies

Solve word analogy problems using BERT embeddings:

```bash
cd word_analogies
python word_analogy_solver.py
```

**What it does:**

- Loads word analogy pairs from `word-test.v1.txt`
- Computes BERT embeddings for words
- Uses cosine similarity and L2 distance to find analogous words
- Evaluates top-k accuracy (k=1, 2, 5, 10, 20)

**Example analogy:** man:woman::king:queen

### Review Sentiment Analysis

Analyze sentiment from Amazon product reviews:

```bash
cd review_sentiment
python review_sentiment_classifier.py
```

**What it does:**

- Loads Amazon review data from `amazon_reviews.csv`
- Implements two approaches:
  1. **Feature Extraction**: Extracts BERT embeddings and trains a Logistic Regression classifier
  2. **Fine-tuning**: Fine-tunes BERT end-to-end for sentiment classification
- Evaluates both approaches and compares performance
- Outputs accuracy metrics and classification reports

**Note:** The dataset should contain columns: `reviewText` (review text) and `overall` (rating 1-5)

### Aspect-Based Sentiment Analysis

Perform aspect-level sentiment analysis on restaurant or laptop reviews:

```bash
python aspect_sentiment_analysis.py
```

**What it does:**

- Downloads SemEval-2014 Task 4 dataset (or uses local files if available)
- Parses XML-formatted review data
- Identifies aspect terms (e.g., "food", "service", "battery life")
- Predicts sentiment (positive, neutral, negative) for each aspect
- Provides detailed evaluation metrics

**Supported domains:**

- Restaurants (default)
- Laptops

The script will automatically create `data/`, `results/`, and `models/` directories for storing datasets, outputs, and trained models.

## GPU Acceleration

All scripts automatically detect and use GPU acceleration if available:

- Check GPU availability: The scripts will print "Using device: cuda" or "Using device: cpu"
- For optimal performance with large datasets, GPU is recommended
- CPU execution is supported but may be slower for fine-tuning tasks

## Output

### Word Analogies Output

- Top-k accuracy metrics for different similarity measures
- Ranked predictions for word analogies
- Performance comparison between cosine similarity and L2 distance

### Review Sentiment Output

- Training and test set sizes
- Rating distribution statistics
- Accuracy scores for both approaches
- Detailed classification reports
- Model checkpoints saved to `./results/` and `./logs/`

### Aspect-Based Sentiment Output

- Dataset statistics (training/testing examples)
- Polarity distribution across aspects
- Model evaluation metrics (accuracy, F1-score)
- Confusion matrices and classification reports
- Saved models in `./models/` directory
- Visualization plots in `./results/` directory

## Implementation Details

### Word Embeddings

- Uses `bert-base-uncased` model
- Implements caching to avoid recomputing embeddings
- Supports multiple distance metrics (cosine similarity, L2 distance)

### Sentiment Classification

- **Approach 1**: Uses frozen BERT for feature extraction + Logistic Regression
- **Approach 2**: Fine-tunes BERT with early stopping to prevent overfitting
- Implements text cleaning and validation
- Handles variable-length texts with truncation/padding

### Aspect-Based Analysis

- Parses SemEval XML format
- Filters out conflicting labels
- Maps sentiments to integers (positive=2, neutral=1, negative=0)
- Custom BERT-based model architecture

## Troubleshooting

### Common Issues

**Out of Memory (OOM) errors:**

- Reduce batch size in training arguments
- Use CPU instead of GPU for smaller models
- Limit text length (default: 512 tokens)

**Dataset not found:**

- Ensure CSV/text files are in the correct directories
- For aspect-based analysis, the script will attempt to download data automatically
- Check file paths and permissions

**BERT model download issues:**

- Ensure stable internet connection for first-time model download
- Models are cached locally after first download
- Check Hugging Face Hub accessibility

## Performance Notes

- First run will download BERT models (~400MB)
- Word embedding computation is cached for efficiency
- Fine-tuning requires more computational resources than feature extraction
- GPU acceleration significantly speeds up training (5-10x faster)

## License

This project uses pre-trained models from Hugging Face, which are subject to their respective licenses.

## Acknowledgments

- BERT models from [Hugging Face Transformers](https://huggingface.co/transformers/)
- SemEval-2014 Task 4 dataset for aspect-based sentiment analysis
- Word analogy dataset for evaluation

---

**Note**: Ensure all required datasets are available before running the scripts. The sentiment analysis script expects `amazon_reviews.csv` with proper formatting, and the word analogy script requires `word-test.v1.txt`.
