# Multi-Label Classification of Customer Support Tickets with BERT

## Project Overview

This project demonstrates how to use BERT (Bidirectional Encoder Representations from Transformers) to automatically classify customer support tickets into multiple categories based on their content. The goal is to create a system that can efficiently route incoming tickets to the appropriate department or support team, reducing response times and improving customer satisfaction.

## Key Features

- **Multi-Label Classification**: Classifies tickets into multiple categories simultaneously
- **BERT Fine-Tuning**: Uses pre-trained BERT model fine-tuned for our specific task
- **Interactive Demo**: Allows users to input text and see predicted categories
- **Visualization**: Provides visual representation of prediction results

## Project Structure

- `simple_bert_classifier.py`: Main script for training the BERT model
- `demo.py`: Interactive demo to showcase the trained model
- `data/`: Directory for storing the dataset
- `best_model.pt`: Saved model weights (generated after training)
- `label_classes.npy`: Saved label classes (generated after training)
- `training_history.png`: Visualization of training metrics (generated after training)
- `prediction_results.png`: Visualization of prediction results (generated during demo)

## Dataset

The project uses a subset of Amazon Electronics reviews from Kaggle:

- [Amazon Electronics Reviews Dataset](https://www.kaggle.com/datasets/shivamparab/amazon-electronics-reviews)
- Contains product reviews and metadata from the Electronics category
- Includes fields like Product ID, User ID, Reviewer Name, Review Text, Helpful Votes, Overall Rating, Summary, and Unix Review Time

## Categories

The model classifies tickets into the following categories:

- Product Quality
- Customer Service
- Price
- Functionality
- Ease of Use
- Other (default category if none of the above match)

## How to Run

### Prerequisites

- Python 3.7+
- Required packages (install using `pip install -r requirements.txt`):
  - pandas
  - numpy
  - torch
  - transformers
  - scikit-learn
  - matplotlib
  - tqdm

### Step 1: Download the Dataset

1. Download the [Amazon Electronics Reviews Dataset](https://www.kaggle.com/datasets/shivamparab/amazon-electronics-reviews) from Kaggle
2. Create a `data` directory in the project root
3. Place the downloaded JSON file in the `data` directory and rename it to `electronics_reviews.json`

### Step 2: Train the Model

Run the following command to train the model:

```bash
python simple_bert_classifier.py
```

This will:

- Load and preprocess the data
- Train the BERT model
- Save the trained model to `best_model.pt`
- Save the label classes to `label_classes.npy`
- Generate a visualization of training metrics to `training_history.png`

### Step 3: Run the Web Application

Run the following command to start the Flask web application:

```bash
python app.py
```

Then open your web browser and navigate to:

```
http://localhost:5000
```

You can now:

- Enter your own customer support ticket text
- Select from example texts
- See the predicted categories and probabilities
- View a visualization of the results

### Alternative: Run the Command-Line Demo

If you prefer a command-line interface, you can run:

```bash
python demo.py
```

## How It Works

### Data Preprocessing

1. **Text Cleaning**: Remove special characters, convert to lowercase, etc.
2. **Category Creation**: Assign categories based on keywords in the text
3. **Tokenization**: Convert text to BERT tokens
4. **Multi-Label Encoding**: Convert categories to binary vectors

### Model Architecture

1. **BERT Base**: Pre-trained BERT model for text understanding
2. **Classification Head**: Linear layer to predict multiple categories
3. **Loss Function**: Binary Cross Entropy with Logits for multi-label classification

### Training Process

1. **Data Splitting**: Split data into training and validation sets
2. **Fine-Tuning**: Update BERT weights for our specific task
3. **Evaluation**: Calculate metrics like F1 score and Hamming Loss
4. **Visualization**: Plot training metrics over time

### Prediction

1. **Text Input**: User provides text to classify
2. **Preprocessing**: Clean and tokenize the input text
3. **Model Inference**: Get predictions from the trained model
4. **Thresholding**: Apply threshold to get final categories
5. **Visualization**: Display results with probabilities

## Performance Metrics

The model is evaluated using:

- Micro F1 Score: Overall performance across all categories
- Macro F1 Score: Average performance for each category
- Hamming Loss: Fraction of incorrectly predicted labels

## Future Improvements

- Use a larger dataset for better generalization
- Try different pre-trained models (e.g., RoBERTa, DistilBERT)
- Implement data augmentation techniques
- Create a web interface for easier interaction
- Deploy the model as a REST API

## Acknowledgments

- [Hugging Face Transformers](https://github.com/huggingface/transformers) for the BERT implementation
- [Kaggle](https://www.kaggle.com/) for providing the dataset
- [PyTorch](https://pytorch.org/) for the deep learning framework
