import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import EarlyStoppingCallback
from torch.utils.data import Dataset
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create directories for model outputs
    os.makedirs('./results', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)
    
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading dataset...")
    df = pd.read_csv('amazon_reviews.csv')
    
    # Extract the reviews and ratings
    reviews = df['reviewText'].tolist()
    ratings = df['overall'].tolist()
    
    print(f"Total number of reviews: {len(reviews)}")
    print("Rating distribution:")
    rating_counts = pd.Series(ratings).value_counts().sort_index()
    for rating, count in rating_counts.items():
        print(f"Rating {rating}: {count} reviews")
    
    # Split the dataset - 80% train, 20% test
    print("Splitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        reviews, ratings, test_size=0.2, random_state=42, stratify=ratings
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Approach 1: Extract features using BERT and train a simple classifier
    print("\nApproach 1: Feature extraction + classifier")
    approach1(X_train, X_test, y_train, y_test, device)
    
    # Approach 2: Fine-tune BERT for sentiment classification
    print("\nApproach 2: Fine-tuning BERT")
    approach2(X_train, X_test, y_train, y_test, device)

def approach1(X_train, X_test, y_train, y_test, device):
    """Extract features with BERT and train a logistic regression classifier."""
    # Load pre-trained model and tokenizer
    print("Loading BERT model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    model = model.to(device)
    model.eval()
    
    print("Extracting features for training set...")
    X_train_features = extract_features_individually(X_train, tokenizer, model, device)
    
    print("Extracting features for test set...")
    X_test_features = extract_features_individually(X_test, tokenizer, model, device)
    
    # Train and evaluate classifier
    print("Training logistic regression classifier...")
    classifier = LogisticRegression(max_iter=1000, class_weight='balanced')
    classifier.fit(X_train_features, y_train)
    
    print("Evaluating classifier...")
    predictions = classifier.predict(X_test_features)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    
    print(f"Logistic Regression - Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    
    return accuracy, report

def approach2(X_train, X_test, y_train, y_test, device):
    """Fine-tune BERT model for sentiment classification."""
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Create datasets with proper text cleaning
    print("Preparing training dataset...")
    train_texts = [clean_text(text) for text in X_train]
    train_dataset = ReviewDataset(train_texts, y_train, tokenizer)
    
    print("Preparing test dataset...")
    test_texts = [clean_text(text) for text in X_test]
    test_dataset = ReviewDataset(test_texts, y_test, tokenizer)
    
    # Fine-tune the model with early stopping to prevent overfitting
    model_name = "bert-base-uncased"
    fine_tune_accuracy, fine_tune_report = fine_tune_model(model_name, train_dataset, test_dataset, device)
    
    return fine_tune_accuracy, fine_tune_report

def clean_text(text):
    """Clean and validate text for processing."""
    if text is None:
        return ""
    elif not isinstance(text, str):
        return str(text)
    
    # Limit text length to prevent memory issues
    if len(text) > 512:
        return text[:512]
    
    return text

def extract_features_individually(texts, tokenizer, model, device):
    """Extract features one text at a time to avoid batch processing issues."""
    features = []
    
    for text in tqdm(texts):
        # Clean the text
        text = clean_text(text)
        
        # Tokenize and prepare for model
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Extract features
        with torch.no_grad():
            outputs = model(**inputs)
            # Use the [CLS] token representation as the sentence embedding
            feature = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            features.append(feature[0])
    
    return np.array(features)

class ReviewDataset(Dataset):
    """Dataset for sentiment analysis of reviews."""
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
        # Convert tensors to lists for dataset compatibility
        self.encodings = {k: v.tolist() for k, v in self.encodings.items()}
        self.labels = [int(label - 1) for label in labels]  # Convert ratings to 0-4 instead of 1-5
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)

def fine_tune_model(model_name, train_dataset, test_dataset, device):
    """Fine-tune a pre-trained transformer model for sentiment classification."""
    print("Loading model for sequence classification...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)
    model.to(device)
    
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_steps=100,
        save_steps=100,
        do_eval=True,
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )
    
    print("Training model...")
    trainer.train()
    
    print("Evaluating fine-tuned model...")
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")
    
    # Predict on test set
    test_pred = trainer.predict(test_dataset)
    preds = np.argmax(test_pred.predictions, axis=-1)
    
    # Convert predictions back to original scale (1-5)
    preds_original = preds + 1
    y_test_array = np.array([label + 1 for label in test_dataset.labels])
    
    # Calculate accuracy and report
    accuracy = accuracy_score(y_test_array, preds_original)
    report = classification_report(y_test_array, preds_original)
    
    print(f"Fine-tuning - Accuracy: {accuracy:.4f}")
    print("Fine-tuning classification report:")
    print(report)
    
    return accuracy, report

if __name__ == "__main__":
    main()
