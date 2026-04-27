import sys
import os
sys.path.append(os.path.dirname(__file__))
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from preprocess import clean_text, load_tokenizer

# Paths
# Paths
MODEL_PATH = os.path.join('models', 'lstm_model.pth')
TOKENIZER_PATH = os.path.join('models', 'tokenizer.pickle')
CONFIG_PATH = os.path.join('models', 'config.json')

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        last_time_step = lstm_out[:, -1, :]
        out = self.fc(last_time_step)
        return out

app = FastAPI(title="LSTM Next Word Predictor")

# Global variables for model and tokenizer
model = None
tokenizer = None
max_sequence_len = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@app.on_event("startup")
def load_assets():
    global model, tokenizer, max_sequence_len
    if not os.path.exists(MODEL_PATH) or not os.path.exists(TOKENIZER_PATH):
        print("Model or Tokenizer not found. Please train the model first.")
        return

    tokenizer = load_tokenizer(TOKENIZER_PATH)
    
    import json
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
            max_sequence_len = config['max_len']
            vocab_size = config['vocab_size']
            emb_dim = config['embedding_dim']
            hid_dim = config['hidden_dim']
    else:
        # Fallback for old style config
        max_sequence_len = 11
        vocab_size = tokenizer.total_words
        emb_dim = 100
        hid_dim = 150

    model = LSTMModel(vocab_size, emb_dim, hid_dim).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"Model and Tokenizer loaded successfully on {device}.")

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    input_text: str
    predicted_word: str

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    seed_text = clean_text(request.text)
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    
    # Pad sequence
    pad_len = (max_sequence_len - 1) - len(token_list)
    if pad_len > 0:
        token_list = [0] * pad_len + token_list
    else:
        token_list = token_list[-(max_sequence_len - 1):]
    
    input_tensor = torch.LongTensor([token_list]).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        predicted_index = torch.argmax(output, dim=1).item()
        
    predicted_word = tokenizer.index_word.get(predicted_index, "Unknown")
    
    return {
        "input_text": request.text,
        "predicted_word": predicted_word
    }

@app.get("/", response_class=HTMLResponse)
def read_root():
    return FileResponse("src/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
