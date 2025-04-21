from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = FastAPI()

# Load the trained model and tokenizer
model = BertForSequenceClassification.from_pretrained('trained_model')
tokenizer = BertTokenizer.from_pretrained('trained_model')
model.eval()

# Define categories
categories = [
    'Product Quality', 'Customer Service', 'Price', 'Functionality',
    'Technical Issues', 'Shipping/Delivery', 'User Experience',
    'Product Compatibility', 'Product Features', 'Others'
]

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    categories: dict

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    # Tokenize input text
    inputs = tokenizer(
        request.text,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.sigmoid(outputs.logits)
        predictions = (predictions > 0.5).float()
    
    # Create response
    result = {
        categories[i]: bool(predictions[0][i])
        for i in range(len(categories))
    }
    
    return {"categories": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
