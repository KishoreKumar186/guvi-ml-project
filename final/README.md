# Multi-Label Classification of Customer Support Tickets

This project implements a multi-label classification system for customer support tickets using fine-tuned BERT. The system automatically classifies customer support tickets into multiple categories based on their content.

## Project Structure

- `main.py`: Main script containing the entire pipeline
- `requirements.txt`: Project dependencies
- `README.md`: Project documentation

## Setup Instructions

1. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download the dataset:

- Visit https://www.kaggle.com/datasets/shivamparab/amazon-electronics-reviews
- Download the dataset and place it in the project directory as `electronics_reviews.json`

4. Run the project:

```bash
python main.py
```

## Project Components

1. Data Preprocessing

   - Loads and cleans the Amazon Electronics reviews dataset
   - Combines review text and summary
   - Creates multi-label categories

2. Model Training

   - Fine-tunes BERT for multi-label classification
   - Handles class imbalance
   - Implements data augmentation

3. Evaluation

   - Calculates various metrics (F1, Hamming Loss, etc.)
   - Provides model performance insights

4. API
   - Simple FastAPI endpoint for model inference

## Model Categories

The model classifies reviews into the following categories:

- Product Quality
- Customer Service
- Price
- Functionality
- Ease of Use

## Evaluation Metrics

- Micro and Macro F1 scores
- Hamming Loss
- Subset Accuracy
- AUC-ROC for each category
- Model inference time
