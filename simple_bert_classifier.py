"""
Multi-Label Classification of Customer Support Tickets with BERT
=============================================================

This script demonstrates how to use BERT to classify customer support tickets into multiple categories.
It's designed to be beginner-friendly with clear explanations at each step.

Author: [Your Name]
Date: [Current Date]
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, hamming_loss
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import logging
from tqdm import tqdm

# Set up logging to track progress
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

# ===== STEP 1: DATA LOADING AND PREPROCESSING =====

def load_data(file_path, sample_size=1000):
    """
    Load the Amazon Electronics review dataset.
    
    Args:
        file_path: Path to the JSON file
        sample_size: Number of records to load (for faster testing)
    
    Returns:
        DataFrame with the loaded data
    """
    logging.info(f"Loading data from {file_path}...")
    
    try:
        # Load the JSON file
        df = pd.read_json(file_path, lines=True)
        
        # Take a sample if specified
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
            logging.info(f"Loaded {sample_size} records from the dataset")
        else:
            logging.info(f"Loaded {len(df)} records from the dataset")
        
        return df
    
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def clean_text(text):
    """
    Clean and normalize text data.
    
    Args:
        text: The text to clean
    
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    import re
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def create_categories(text):
    """
    Create multi-label categories based on review content.
    
    Args:
        text: The review text
    
    Returns:
        List of categories
    """
    categories = []
    
    # Define keywords for each category
    category_keywords = {
        'Product Quality': ['quality', 'durability', 'build', 'material'],
        'Customer Service': ['service', 'support', 'warranty', 'customer service'],
        'Price': ['price', 'cost', 'value', 'expensive', 'cheap'],
        'Functionality': ['works', 'function', 'features', 'performance'],
        'Ease of Use': ['easy', 'simple', 'intuitive', 'user friendly']
    }
    
    text = text.lower()
    for category, keywords in category_keywords.items():
        if any(keyword in text for keyword in keywords):
            categories.append(category)
    
    # If no categories found, assign 'Other'
    return categories if categories else ['Other']

def preprocess_data(df):
    """
    Preprocess the data for model training.
    
    Args:
        df: DataFrame with the raw data
    
    Returns:
        Processed data ready for model training
    """
    logging.info("Preprocessing data...")
    
    # Combine summary and review text for more context
    df['full_text'] = df['summary'] + ' ' + df['reviewText']
    
    # Clean text
    df['cleaned_text'] = df['full_text'].apply(clean_text)
    
    # Create categories
    df['categories'] = df['cleaned_text'].apply(create_categories)
    
    # Convert categories to multi-hot encoding
    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(df['categories'])
    
    # Get the label classes
    label_classes = mlb.classes_
    
    # Tokenize text using BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Process in batches to avoid memory issues
    batch_size = 100
    all_input_ids = []
    all_attention_masks = []
    
    for i in tqdm(range(0, len(df), batch_size), desc="Tokenizing"):
        batch_texts = df['cleaned_text'].iloc[i:i+batch_size].tolist()
        batch_encodings = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )
        
        all_input_ids.append(batch_encodings['input_ids'])
        all_attention_masks.append(batch_encodings['attention_mask'])
    
    # Combine all batches
    input_ids = torch.cat(all_input_ids, dim=0)
    attention_masks = torch.cat(all_attention_masks, dim=0)
    
    # Create a dictionary with the processed data
    processed_data = {
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'labels': labels,
        'label_classes': label_classes
    }
    
    logging.info("Data preprocessing completed!")
    return processed_data

# ===== STEP 2: CREATE A CUSTOM DATASET =====

class TicketDataset(Dataset):
    """
    Custom Dataset for ticket classification.
    """
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': torch.FloatTensor(self.labels[idx])
        }

# ===== STEP 3: DEFINE THE BERT MODEL =====

class BERTClassifier(nn.Module):
    """
    BERT model for multi-label classification.
    """
    def __init__(self, num_labels, dropout=0.3):
        super(BERTClassifier, self).__init__()
        
        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask):
        # Get BERT embeddings
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token output for classification
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # Get logits for each label
        logits = self.classifier(pooled_output)
        
        return logits

# ===== STEP 4: TRAIN THE MODEL =====

def train_model(model, train_dataloader, val_dataloader, epochs=3, learning_rate=2e-5, device='cuda'):
    """
    Train the BERT model.
    
    Args:
        model: The BERT model
        train_dataloader: DataLoader for training data
        val_dataloader: DataLoader for validation data
        epochs: Number of training epochs
        learning_rate: Learning rate for optimization
        device: Device to use for training (cuda or cpu)
    
    Returns:
        Trained model and training history
    """
    # Move model to device
    model = model.to(device)
    
    # Set up optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    
    # Initialize training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'micro_f1': [],
        'macro_f1': [],
        'hamming_loss': []
    }
    
    # Training loop
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0
        train_steps = 0
        
        train_iterator = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{epochs} [Train]')
        for batch in train_iterator:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            train_steps += 1
            
            # Update progress bar
            train_iterator.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / train_steps
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_steps = 0
        all_preds = []
        all_labels = []
        
        val_iterator = tqdm(val_dataloader, desc='Evaluating')
        with torch.no_grad():
            for batch in val_iterator:
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Forward pass
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                # Update metrics
                val_loss += loss.item()
                val_steps += 1
                
                # Convert logits to predictions
                preds = torch.sigmoid(outputs).cpu().numpy()
                all_preds.append(preds)
                all_labels.append(labels.cpu().numpy())
                
                # Update progress bar
                val_iterator.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate metrics
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        binary_preds = (all_preds > 0.5).astype(int)
        
        micro_f1 = f1_score(all_labels, binary_preds, average='micro')
        macro_f1 = f1_score(all_labels, binary_preds, average='macro')
        ham_loss = hamming_loss(all_labels, binary_preds)
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss / val_steps)
        history['micro_f1'].append(micro_f1)
        history['macro_f1'].append(macro_f1)
        history['hamming_loss'].append(ham_loss)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Log metrics
        logging.info(f'Epoch {epoch + 1}/{epochs} completed in {epoch_time:.2f} seconds')
        logging.info(f'Average training loss: {avg_train_loss:.4f}')
        logging.info(f'Validation loss: {val_loss / val_steps:.4f}')
        logging.info(f'Micro F1: {micro_f1:.4f}, Macro F1: {macro_f1:.4f}, Hamming Loss: {ham_loss:.4f}')
    
    return model, history

# ===== STEP 5: EVALUATE AND VISUALIZE RESULTS =====

def plot_training_history(history):
    """
    Plot training history.
    
    Args:
        history: Dictionary with training metrics
    """
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    
    # Plot F1 scores
    axes[0, 1].plot(history['micro_f1'], label='Micro F1')
    axes[0, 1].plot(history['macro_f1'], label='Macro F1')
    axes[0, 1].set_title('F1 Scores')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].legend()
    
    # Plot Hamming Loss
    axes[1, 0].plot(history['hamming_loss'], label='Hamming Loss')
    axes[1, 0].set_title('Hamming Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Hamming Loss')
    axes[1, 0].legend()
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

# ===== STEP 6: PREDICT ON NEW DATA =====

def predict_categories(model, tokenizer, text, label_classes, device='cuda'):
    """
    Predict categories for a new ticket.
    
    Args:
        model: Trained BERT model
        tokenizer: BERT tokenizer
        text: Text of the ticket
        label_classes: List of label classes
        device: Device to use for prediction
    
    Returns:
        Dictionary with predicted categories and probabilities
    """
    # Clean and tokenize text
    cleaned_text = clean_text(text)
    encodings = tokenizer(
        cleaned_text,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors='pt'
    )
    
    # Move to device
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]
    
    # Get predicted categories
    pred_indices = np.where(probs > 0.5)[0]
    pred_categories = [label_classes[i] for i in pred_indices]
    pred_probs = probs[pred_indices].tolist()
    
    # Create result dictionary
    result = {
        'categories': pred_categories,
        'probabilities': pred_probs
    }
    
    return result

# ===== STEP 7: MAIN FUNCTION =====

def main():
    """
    Main function to run the entire pipeline.
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Define parameters
    sample_size = 1000  # Number of records to use (for faster testing)
    batch_size = 8      # Batch size for training
    epochs = 3          # Number of training epochs
    
    # Define the path to your JSON file
    json_file_path = os.path.join('data', 'electronics_reviews.json')
    
    # Check if file exists
    if not os.path.exists(json_file_path):
        logging.error(f"Data file not found at: {json_file_path}")
        logging.info("Please download the dataset from Kaggle and place it in the 'data' directory")
        return
    
    try:
        # Step 1: Load and preprocess data
        df = load_data(json_file_path, sample_size)
        processed_data = preprocess_data(df)
        
        # Step 2: Create datasets
        # Split data into train and validation sets
        train_inputs, val_inputs, train_labels, val_labels = train_test_split(
            processed_data['input_ids'],
            processed_data['labels'],
            test_size=0.2,
            random_state=42
        )
        
        train_masks, val_masks, _, _ = train_test_split(
            processed_data['attention_mask'],
            processed_data['labels'],
            test_size=0.2,
            random_state=42
        )
        
        # Create datasets
        train_dataset = TicketDataset(train_inputs, train_masks, train_labels)
        val_dataset = TicketDataset(val_inputs, val_masks, val_labels)
        
        # Create dataloaders
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        
        logging.info(f"Created dataloaders with {len(train_dataloader)} training batches and {len(val_dataloader)} validation batches")
        
        # Step 3: Initialize model
        model = BERTClassifier(num_labels=len(processed_data['label_classes']))
        
        # Step 4: Train model
        model, history = train_model(
            model, 
            train_dataloader, 
            val_dataloader, 
            epochs=epochs, 
            device=device
        )
        
        # Step 5: Plot training history
        plot_training_history(history)
        
        # Step 6: Save model
        torch.save(model.state_dict(), 'best_model.pt')
        logging.info("Model saved to 'best_model.pt'")
        
        # Save label classes
        np.save('label_classes.npy', processed_data['label_classes'])
        logging.info("Label classes saved to 'label_classes.npy'")
        
        # Step 7: Example prediction
        example_text = "This product is great quality but a bit expensive. The customer service was excellent though."
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        result = predict_categories(
            model, 
            tokenizer, 
            example_text, 
            processed_data['label_classes'],
            device=device
        )
        
        logging.info(f"Example prediction: {result}")
        
        logging.info("Pipeline completed successfully!")
        
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main() 