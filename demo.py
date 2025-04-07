"""
Multi-Label Classification of Customer Support Tickets - DEMO
===========================================================

This script provides a simple demo of the trained BERT model for classifying customer support tickets.
It allows you to input text and see the predicted categories.

Author: [Your Name]
Date: [Current Date]
"""

import torch
from transformers import BertTokenizer, BertModel
from torch import nn
import numpy as np
import os
import matplotlib.pyplot as plt
import re

# ===== STEP 1: DEFINE THE BERT MODEL =====

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

# ===== STEP 2: HELPER FUNCTIONS =====

def clean_text(text):
    """
    Clean and normalize text data.
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def predict_categories(model, tokenizer, text, label_classes, device='cuda'):
    """
    Predict categories for a new ticket.
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

def visualize_predictions(result):
    """
    Visualize the prediction results.
    """
    if not result['categories']:
        print("No categories predicted.")
        return
    
    # Create a bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(result['categories'], result['probabilities'])
    plt.title('Predicted Categories and Probabilities')
    plt.xlabel('Categories')
    plt.ylabel('Probability')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('prediction_results.png')
    plt.close()
    
    print(f"Visualization saved as 'prediction_results.png'")

# ===== STEP 3: LOAD THE TRAINED MODEL =====

def load_model(model_path, label_classes_path):
    """
    Load the trained model and label classes.
    """
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}")
        print("Please train the model first using simple_bert_classifier.py")
        print("Run: python simple_bert_classifier.py")
        return None, None
    
    # Check if label classes exist
    if not os.path.exists(label_classes_path):
        print(f"Label classes file not found at {label_classes_path}")
        print("Please train the model first using simple_bert_classifier.py")
        print("Run: python simple_bert_classifier.py")
        return None, None
    
    # Load label classes
    label_classes = np.load(label_classes_path, allow_pickle=True)
    
    # Initialize model
    model = BERTClassifier(num_labels=len(label_classes))
    
    # Load model weights
    model.load_state_dict(torch.load(model_path))
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    return model, tokenizer, label_classes, device

# ===== STEP 4: INTERACTIVE DEMO =====

def run_demo():
    """
    Run the interactive demo.
    """
    print("=" * 50)
    print("Multi-Label Classification of Customer Support Tickets")
    print("=" * 50)
    print("\nLoading model...")
    
    # Load model and tokenizer
    model, tokenizer, label_classes, device = load_model('best_model.pt', 'label_classes.npy')
    
    if model is None:
        return
    
    print(f"Model loaded successfully! Using device: {device}")
    print(f"Available categories: {', '.join(label_classes)}")
    print("\nEnter 'quit' to exit the demo.")
    
    # Example texts
    example_texts = [
        "This product is great quality but a bit expensive. The customer service was excellent though.",
        "The product works well but it's difficult to use. The instructions are unclear.",
        "I'm very satisfied with the price and functionality. The build quality is solid.",
        "The product arrived damaged. Customer service was unhelpful and didn't offer a refund."
    ]
    
    print("\nExample texts:")
    for i, text in enumerate(example_texts):
        print(f"{i+1}. {text}")
    
    while True:
        print("\n" + "-" * 50)
        user_input = input("\nEnter your text (or 'example N' to use example N, or 'quit' to exit): ")
        
        if user_input.lower() == 'quit':
            break
        
        # Check if user wants to use an example
        if user_input.lower().startswith('example'):
            try:
                example_num = int(user_input.split()[1])
                if 1 <= example_num <= len(example_texts):
                    text = example_texts[example_num - 1]
                    print(f"\nUsing example {example_num}: {text}")
                else:
                    print(f"Example {example_num} not found. Please enter a number between 1 and {len(example_texts)}.")
                    continue
            except (IndexError, ValueError):
                print("Invalid example format. Please use 'example N' where N is the example number.")
                continue
        else:
            text = user_input
        
        # Make prediction
        result = predict_categories(model, tokenizer, text, label_classes, device)
        
        # Display results
        print("\nPredicted Categories:")
        if result['categories']:
            for category, prob in zip(result['categories'], result['probabilities']):
                print(f"- {category}: {prob:.2f}")
        else:
            print("No categories predicted.")
        
        # Visualize results
        visualize_predictions(result)
    
    print("\nThank you for using the demo!")

if __name__ == "__main__":
    run_demo() 