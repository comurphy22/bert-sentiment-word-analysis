import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import xml.etree.ElementTree as ET
import requests
from io import BytesIO
import zipfile
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42) if torch.cuda.is_available() else None

# Create output directories
os.makedirs('results', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class SemEvalDataProcessor:
    """
    Class to download and process the SemEval-2014 Task 4 dataset
    """
    def __init__(self, domain='restaurants'):
        self.domain = domain  # 'restaurants' or 'laptops'
        self.dataset_url = "https://alt.qcri.org/semeval2014/task4/data/uploads/semeval2014-absa-traindata.zip"
        self.test_url = "https://alt.qcri.org/semeval2014/task4/data/uploads/semeval2014-absa-testdata.zip"
        self.data_dir = "data"
        os.makedirs(self.data_dir, exist_ok=True)
        
    def download_data(self):
        """Check for local files before attempting download"""
        train_file = os.path.join(self.data_dir, f"{self.domain}_train.xml")
        test_file = os.path.join(self.data_dir, f"{self.domain}_test.xml")
        
        # Check if files already exist
        if os.path.exists(train_file) and os.path.exists(test_file):
            print(f"Dataset files found in {self.data_dir}")
            return train_file, test_file
            
        # For testing purposes, if we have a trial file, copy it to both train and test
        trial_file = os.path.join(self.data_dir, f"{self.domain}-trial.xml")
        if os.path.exists(trial_file):
            print(f"Found trial file. Using it for both training and testing.")
            import shutil
            shutil.copy(trial_file, train_file)
            shutil.copy(trial_file, test_file)
            return train_file, test_file
        
        print("No local dataset files found. Attempting download...")
        
        # Rest of your download code...
        try:
            response = requests.get(self.dataset_url)
            response.raise_for_status()
            
            with zipfile.ZipFile(BytesIO(response.content)) as z:
                for filename in z.namelist():
                    if self.domain in filename.lower() and '.xml' in filename.lower():
                        z.extract(filename, self.data_dir)
                        os.rename(os.path.join(self.data_dir, filename), train_file)
                        print(f"Training data extracted to {train_file}")
                        
            # Download test data
            response = requests.get(self.test_url)
            response.raise_for_status()
            
            with zipfile.ZipFile(BytesIO(response.content)) as z:
                for filename in z.namelist():
                    if self.domain in filename.lower() and '.xml' in filename.lower():
                        z.extract(filename, self.data_dir)
                        os.rename(os.path.join(self.data_dir, filename), test_file)
                        print(f"Test data extracted to {test_file}")
                        
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("Please manually download the dataset from the SemEval-2014 website")
            print("and place the XML files in the data directory.")
            
        return train_file, test_file

    def parse_xml_file(self, file_path):
        """Parse the SemEval XML format into a structured format"""
        print(f"Parsing {file_path}...")
        
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            data = []
            
            for sentence in root.findall(".//sentence"):
                sentence_id = sentence.get('id')
                text = sentence.find('text').text
                
                # Check if the sentence contains aspect terms
                aspectTerms = sentence.find('aspectTerms')
                
                if aspectTerms is not None:
                    for aspectTerm in aspectTerms.findall('aspectTerm'):
                        term = aspectTerm.get('term')
                        polarity = aspectTerm.get('polarity')
                        from_idx = int(aspectTerm.get('from'))
                        to_idx = int(aspectTerm.get('to'))
                        
                        # Skip if the polarity is "conflict"
                        if polarity == "conflict":
                            continue
                        
                        data.append({
                            'sentence_id': sentence_id,
                            'text': text,
                            'aspect_term': term,
                            'polarity': polarity,
                            'from_idx': from_idx,
                            'to_idx': to_idx
                        })
            
            return pd.DataFrame(data)
            
        except Exception as e:
            print(f"Error parsing XML file: {e}")
            return pd.DataFrame()

    def load_data(self):
        """Load and preprocess the SemEval dataset"""
        train_file, test_file = self.download_data()
        
        train_df = self.parse_xml_file(train_file)
        test_df = self.parse_xml_file(test_file)
        
        print(f"Loaded {len(train_df)} training examples and {len(test_df)} testing examples")
        
        # Map polarity labels to integers
        polarity_map = {'positive': 2, 'neutral': 1, 'negative': 0}
        train_df['polarity_label'] = train_df['polarity'].map(polarity_map)
        test_df['polarity_label'] = test_df['polarity'].map(polarity_map)
        
        return train_df, test_df
        
    def prepare_input_data(self, df, tokenizer, max_length=128):
        """
        Prepare input data for the model by highlighting the aspect term in the sentence
        and tokenizing the input
        """
        inputs = []
        masks = []
        token_type_ids = []
        labels = []
        aspect_masks = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Preparing inputs"):
            text = row['text']
            aspect = row['aspect_term']
            from_idx = row['from_idx']
            to_idx = row['to_idx']
            
            # Highlight the aspect by inserting special tokens
            marked_text = text[:from_idx] + "[ASPECT] " + text[from_idx:to_idx] + " [/ASPECT]" + text[to_idx:]
            
            # Tokenize
            encoded = tokenizer(
                marked_text,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            inputs.append(encoded['input_ids'][0])
            masks.append(encoded['attention_mask'][0])
            if 'token_type_ids' in encoded:
                token_type_ids.append(encoded['token_type_ids'][0])
            
            # Create aspect mask (1 for aspect tokens, 0 otherwise)
            aspect_start_marker = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("[ASPECT]"))[0]
            aspect_end_marker = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("[/ASPECT]"))[0]
            
            aspect_mask = torch.zeros(max_length)
            input_ids = encoded['input_ids'][0].numpy()
            
            # Find positions of aspect markers
            aspect_start_positions = np.where(input_ids == aspect_start_marker)[0]
            aspect_end_positions = np.where(input_ids == aspect_end_marker)[0]
            
            if len(aspect_start_positions) > 0 and len(aspect_end_positions) > 0:
                start_pos = aspect_start_positions[0] + 1  # Skip the marker itself
                end_pos = aspect_end_positions[0]
                aspect_mask[start_pos:end_pos] = 1
            
            aspect_masks.append(aspect_mask)
            labels.append(row['polarity_label'])
        
        # Convert to tensors
        inputs = torch.stack(inputs)
        masks = torch.stack(masks)
        aspect_masks = torch.stack(aspect_masks)
        labels = torch.tensor(labels)
        
        if token_type_ids:
            token_type_ids = torch.stack(token_type_ids)
            return inputs, masks, token_type_ids, aspect_masks, labels
        else:
            return inputs, masks, None, aspect_masks, labels

class AspectBasedSentimentDataset(Dataset):
    """Dataset for aspect-based sentiment analysis"""
    def __init__(self, inputs, masks, token_type_ids, aspect_masks, labels):
        self.inputs = inputs
        self.masks = masks
        self.token_type_ids = token_type_ids
        self.aspect_masks = aspect_masks
        self.labels = labels
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        item = {
            'input_ids': self.inputs[idx],
            'attention_mask': self.masks[idx],
            'aspect_mask': self.aspect_masks[idx],
            'labels': self.labels[idx]
        }
        
        if self.token_type_ids is not None:
            item['token_type_ids'] = self.token_type_ids[idx]
            
        return item

class AspectBasedSentimentClassifier(nn.Module):
    """BERT-based model for aspect-based sentiment classification"""
    def __init__(self, num_labels=3, dropout_prob=0.1):
        super(AspectBasedSentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask, aspect_mask, token_type_ids=None):
        # Get BERT embeddings
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Apply aspect mask to focus only on aspect terms
        # Expand aspect mask to match hidden dimension
        expanded_aspect_mask = aspect_mask.unsqueeze(-1).expand(sequence_output.size())
        
        # Get weighted representation of aspect terms
        masked_output = sequence_output * expanded_aspect_mask
        
        # Average the masked embeddings
        aspect_representation = masked_output.sum(dim=1) / (aspect_mask.sum(dim=1, keepdim=True) + 1e-10)
        
        # Apply dropout and classify
        pooled_output = self.dropout(aspect_representation)
        logits = self.classifier(pooled_output)
        
        return logits

def train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, device, num_epochs=5):
    """Train the aspect-based sentiment classification model"""
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'val_f1': []
    }
    
    best_val_f1 = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training
        model.train()
        total_loss = 0
        predictions = []
        true_labels = []
        
        progress_bar = tqdm(train_dataloader, desc="Training")
        
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items() if k != 'token_type_ids'}
            token_type_ids = batch.get('token_type_ids')
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                aspect_mask=batch['aspect_mask'],
                token_type_ids=token_type_ids
            )
            
            # Compute loss
            loss = criterion(outputs, batch['labels'])
            
            # Backward pass and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            # Track predictions
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(batch['labels'].cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Calculate training metrics
        train_loss = total_loss / len(train_dataloader)
        train_acc = accuracy_score(true_labels, predictions)
        
        # Validation
        model.eval()
        val_predictions = []
        val_true_labels = []
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items() if k != 'token_type_ids'}
                token_type_ids = batch.get('token_type_ids')
                
                # Forward pass
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    aspect_mask=batch['aspect_mask'],
                    token_type_ids=token_type_ids
                )
                
                # Compute loss
                loss = criterion(outputs, batch['labels'])
                val_loss += loss.item()
                
                # Track predictions
                _, preds = torch.max(outputs, dim=1)
                val_predictions.extend(preds.cpu().numpy())
                val_true_labels.extend(batch['labels'].cpu().numpy())
        
        # Calculate validation metrics
        val_loss /= len(val_dataloader)
        val_acc = accuracy_score(val_true_labels, val_predictions)
        val_f1 = f1_score(val_true_labels, val_predictions, average='macro')
        
        # Save metrics to history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), 'models/best_absa_model.pt')
            print("Saved best model!")
    
    return history

def evaluate_model(model, test_dataloader, device):
    """Evaluate the model on test data"""
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items() if k != 'token_type_ids'}
            token_type_ids = batch.get('token_type_ids')
            
            # Forward pass
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                aspect_mask=batch['aspect_mask'],
                token_type_ids=token_type_ids
            )
            
            # Get predictions
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(batch['labels'].cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='macro')
    report = classification_report(true_labels, predictions, target_names=['negative', 'neutral', 'positive'])
    conf_matrix = confusion_matrix(true_labels, predictions)
    
    results = {
        'accuracy': accuracy,
        'f1_score': f1,
        'classification_report': report,
        'confusion_matrix': conf_matrix,
        'predictions': predictions,
        'true_labels': true_labels
    }
    
    return results

def plot_training_history(history):
    """Plot training history metrics"""
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.plot(history['val_f1'], label='Val F1 Score')
    plt.title('Model Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/training_history.png')
    plt.close()

def plot_confusion_matrix(conf_matrix):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Neutral', 'Positive'],
                yticklabels=['Negative', 'Neutral', 'Positive'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png')
    plt.close()
    
def save_results(results, history, domain):
    """Save results to file"""
    report = f"""
    Aspect-Based Sentiment Analysis Results ({domain} domain)
    ======================================================
    
    Classification Report:
    {results['classification_report']}
    
    Accuracy: {results['accuracy']:.4f}
    Macro F1 Score: {results['f1_score']:.4f}
    
    Training History:
    ----------------
    """
    
    for epoch in range(len(history['train_loss'])):
        report += f"Epoch {epoch+1}: "
        report += f"Train Loss: {history['train_loss'][epoch]:.4f}, "
        report += f"Val Loss: {history['val_loss'][epoch]:.4f}, "
        report += f"Train Acc: {history['train_acc'][epoch]:.4f}, "
        report += f"Val Acc: {history['val_acc'][epoch]:.4f}, "
        report += f"Val F1: {history['val_f1'][epoch]:.4f}\n"
    
    with open('results/absa_results.txt', 'w') as f:
        f.write(report)
    
    print(f"Results saved to results/absa_results.txt")

def perform_analysis(results, test_df):
    """Perform additional analysis on results"""
    # Map predictions back to original labels
    label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    predictions = [label_map[pred] for pred in results['predictions']]
    
    # Add predictions to test dataframe
    test_df_with_preds = test_df.copy()
    test_df_with_preds['predicted_polarity'] = predictions
    
    # Add column for correct/incorrect predictions
    test_df_with_preds['correct'] = test_df_with_preds['polarity'] == test_df_with_preds['predicted_polarity']
    
    # Calculate accuracy by aspect term frequency
    aspect_counts = test_df_with_preds.groupby('aspect_term').size().to_dict()
    test_df_with_preds['aspect_frequency'] = test_df_with_preds['aspect_term'].map(aspect_counts)
    
    # Create frequency bins
    test_df_with_preds['frequency_bin'] = pd.cut(test_df_with_preds['aspect_frequency'], 
                                             bins=[0, 1, 3, 5, 10, float('inf')], 
                                             labels=['1', '2-3', '4-5', '6-10', '>10'])
    
    # Calculate accuracy by frequency bin
    freq_accuracy = test_df_with_preds.groupby('frequency_bin')['correct'].mean()
    
    # Calculate most common aspect terms
    top_aspects = test_df_with_preds['aspect_term'].value_counts().head(10)
    
    # Calculate per-class accuracy
    class_accuracy = test_df_with_preds.groupby('polarity')['correct'].mean()
    
    # Calculate sentence length
    test_df_with_preds['sentence_length'] = test_df_with_preds['text'].apply(lambda x: len(x.split()))
    
    # Create sentence length bins
    test_df_with_preds['length_bin'] = pd.cut(test_df_with_preds['sentence_length'], 
                                           bins=[0, 10, 15, 20, 25, float('inf')], 
                                           labels=['≤10', '11-15', '16-20', '21-25', '>25'])
    
    # Calculate accuracy by sentence length
    length_accuracy = test_df_with_preds.groupby('length_bin')['correct'].mean()
    
    # Plot accuracy by frequency bin
    plt.figure(figsize=(10, 6))
    freq_accuracy.plot(kind='bar')
    plt.title('Accuracy by Aspect Term Frequency')
    plt.xlabel('Aspect Frequency')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('results/accuracy_by_frequency.png')
    plt.close()
    
    # Plot accuracy by polarity class
    plt.figure(figsize=(8, 6))
    class_accuracy.plot(kind='bar', color=['red', 'gray', 'green'])
    plt.title('Accuracy by Sentiment Polarity')
    plt.xlabel('Polarity')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('results/accuracy_by_polarity.png')
    plt.close()
    
    # Plot top aspects
    plt.figure(figsize=(12, 6))
    top_aspects.plot(kind='bar')
    plt.title('Most Common Aspect Terms')
    plt.xlabel('Aspect Term')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('results/common_aspects.png')
    plt.close()
    
    # Plot accuracy by sentence length
    plt.figure(figsize=(10, 6))
    length_accuracy.plot(kind='bar')
    plt.title('Accuracy by Sentence Length')
    plt.xlabel('Sentence Length (words)')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('results/accuracy_by_length.png')
    plt.close()
    
    # Save problematic examples
    incorrect_examples = test_df_with_preds[~test_df_with_preds['correct']].copy()
    incorrect_examples = incorrect_examples.sort_values('aspect_frequency', ascending=False)
    
    # Save incorrect examples to csv
    incorrect_examples.to_csv('results/incorrect_examples.csv', index=False)
    
    # Save analysis results
    analysis_results = {
        'freq_accuracy': freq_accuracy.to_dict(),
        'class_accuracy': class_accuracy.to_dict(),
        'top_aspects': top_aspects.to_dict(),
        'length_accuracy': length_accuracy.to_dict()
    }
    
    with open('results/analysis_results.json', 'w') as f:
        json.dump(analysis_results, f, indent=4)
    
    # Generate markdown report with insights
    report = f"""
    # Aspect-Based Sentiment Analysis: Additional Insights
    
    ## Performance by Aspect Frequency
    
    Accuracy tends to {'increase' if freq_accuracy.is_monotonic_increasing else 'vary'} with aspect frequency.
    - Single-occurrence aspects: {freq_accuracy['1']:.4f}
    - Very common aspects (>10): {freq_accuracy['>10']:.4f}
    
    ## Performance by Sentiment Class
    
    - Negative sentiment: {class_accuracy.get('negative', 'N/A'):.4f}
    - Neutral sentiment: {class_accuracy.get('neutral', 'N/A'):.4f}
    - Positive sentiment: {class_accuracy.get('positive', 'N/A'):.4f}
    
    The model performs {'best' if class_accuracy.idxmax() == 'positive' else 'worst'} on positive sentiments.
    
    ## Most Common Aspect Terms
    
    Top 5 most frequent aspects:
    {', '.join(top_aspects.index[:5])}
    
    ## Performance by Sentence Length
    
    - Short sentences (≤10 words): {length_accuracy['≤10']:.4f}
    - Long sentences (>25 words): {length_accuracy['>25']:.4f}
    
    ## Error Analysis
    
    Common misclassifications:
    - {incorrect_examples['polarity'].value_counts().idxmax()} sentiments most often misclassified
    - {len(incorrect_examples)} total incorrect predictions
    
    See `incorrect_examples.csv` for a full list of errors.
    """
    
    with open('results/insights.md', 'w') as f:
        f.write(report)
        
    return analysis_results

def main():
    """Main function to run aspect-based sentiment analysis"""
    # Set domain (either 'restaurants' or 'laptops')
    domain = 'laptops'  # You can change this to 'laptops'
    
    print(f"=== Aspect-Based Sentiment Analysis on {domain.capitalize()} Dataset ===")
    
    # Process data
    processor = SemEvalDataProcessor(domain=domain)
    train_df, test_df = processor.load_data()
    
    # Limit dataset size for faster execution
    if len(train_df) > 200:
        train_df = train_df.sample(200, random_state=42)
    if len(test_df) > 50:
        test_df = test_df.sample(50, random_state=42)
    
    # Output dataset statistics
    print("\nDataset Statistics:")
    print(f"Training examples: {len(train_df)}")
    print(f"Test examples: {len(test_df)}")
    print("\nPolarity distribution (training):")
    print(train_df['polarity'].value_counts())
    print("\nPolarity distribution (testing):")
    print(test_df['polarity'].value_counts())
    
    # Split training data into train and validation sets
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42, stratify=train_df['polarity'])
    print(f"\nTraining set: {len(train_df)} examples")
    print(f"Validation set: {len(val_df)} examples")
    print(f"Test set: {len(test_df)} examples")
    
    # Initialize tokenizer
    print("\nInitializing tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Add special tokens
    special_tokens = {'additional_special_tokens': ['[ASPECT]', '[/ASPECT]']}
    tokenizer.add_special_tokens(special_tokens)
    
    # Prepare data
    print("\nPreparing input data...")
    train_inputs, train_masks, train_token_type_ids, train_aspect_masks, train_labels = processor.prepare_input_data(train_df, tokenizer)
    val_inputs, val_masks, val_token_type_ids, val_aspect_masks, val_labels = processor.prepare_input_data(val_df, tokenizer)
    test_inputs, test_masks, test_token_type_ids, test_aspect_masks, test_labels = processor.prepare_input_data(test_df, tokenizer)
    
    # Create datasets
    train_dataset = AspectBasedSentimentDataset(train_inputs, train_masks, train_token_type_ids, train_aspect_masks, train_labels)
    val_dataset = AspectBasedSentimentDataset(val_inputs, val_masks, val_token_type_ids, val_aspect_masks, val_labels)
    test_dataset = AspectBasedSentimentDataset(test_inputs, test_masks, test_token_type_ids, test_aspect_masks, test_labels)
    
    # Create data loaders
    batch_size = 16
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    print("\nInitializing model...")
    model = AspectBasedSentimentClassifier(num_labels=3)
    model.bert.resize_token_embeddings(len(tokenizer))  # Resize for new tokens
    model = model.to(device)
    
    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    total_steps = len(train_dataloader) * 5  # 5 epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Train model
    print("\nTraining model...")
    history = train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, device, num_epochs=2)
    
    # Load best model
    print("\nLoading best model...")
    model.load_state_dict(torch.load('models/best_absa_model.pt'))
    
    # Evaluate model
    print("\nEvaluating model on test set...")
    results = evaluate_model(model, test_dataloader, device)
    
    # Print results
    print("\nTest Results:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1 Score (Macro): {results['f1_score']:.4f}")
    print("\nClassification Report:")
    print(results['classification_report'])
    
    # Plot results
    print("\nPlotting results...")
    plot_training_history(history)
    plot_confusion_matrix(results['confusion_matrix'])
    
    # Save results
    print("\nSaving results...")
    save_results(results, history, domain)
    
    # Additional analysis
    print("\nPerforming additional analysis...")
    analysis_results = perform_analysis(results, test_df)
    
    print("\nAnalysis complete! Results saved to the 'results' directory.")

if __name__ == "__main__":
    main()