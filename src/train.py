import sys
import os
sys.path.append(os.path.dirname(__file__))
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wikipediaapi
from torch.utils.data import DataLoader, TensorDataset
from preprocess import clean_text, prepare_sequences, save_tokenizer

# Paths
DATA_PATH = os.path.join('data', 'raw_text.txt')
MODEL_PATH = os.path.join('models', 'lstm_model.pth')
TOKENIZER_PATH = os.path.join('models', 'tokenizer.pickle')

def fetch_wikipedia_data(topics):
    wiki = wikipediaapi.Wikipedia(
        user_agent='AI-Sequence-Predictor/1.0 (contact: your-email@example.com)',
        language='en',
        extract_format=wikipediaapi.ExtractFormat.WIKI
    )
    
    full_text = ""
    for topic in topics:
        page = wiki.page(topic)
        if page.exists():
            print(f"Fetching data for topic: {topic}")
            full_text += page.text + " "
        else:
            print(f"Topic not found: {topic}")
    
    return full_text

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

def train():
    # Large list of topics covering all requested categories
    topics = [
        "English language", "Grammar", "Science", "History of science", "Physics", "Theoretical physics", 
        "History", "World history", "History of India", "History of the United Kingdom", 
        "Geography", "Physical geography", "Politics", "Political science", "Democracy", 
        "News", "Journalism", "Technology", "Information technology", "History of technology", 
        "Education", "Higher education", "Health", "Medicine", "Public health", 
        "Money", "Finance", "Economy", "Human", "Evolution", "Human behavior", 
        "Value (ethics)", "Ethics", "Philosophy", "World", "Earth", "God", 
        "Theology", "Evil", "Ethics of evil", "Animal", "Mammal", "Bird", "Ornithology", 
        "Atmosphere of Earth", "Climate", "Chemistry", "Organic chemistry", 
        "Biology", "Molecular biology", "Human sexuality", "Internet", "World Wide Web", 
        "Crime", "Criminology", "Award", "Nobel Prize", "Reward", "Employment", "Labor economics",
        "Mathematics", "Algebra", "Calculus", "Geometry", "Space exploration", "NASA", "SpaceX",
        "Artificial intelligence", "Machine learning", "Deep learning", "Neural network",
        "Sociology", "Psychology", "Anthropology", "Archaeology", "Architecture", "Art history",
        "Literature", "Poetry", "Music theory", "Cinema", "Sports", "Olympics",
        "Environment", "Sustainability", "Renewable energy", "Global warming",
        "Law", "Justice", "Human rights", "International relations", "War", "Peace"
    ]
    
    # Check if local data exists and is sufficient (e.g., > 4MB)
    if not os.path.exists(DATA_PATH) or os.path.getsize(DATA_PATH) < 4000000:
        print("Fetching massive data from Wikipedia (Target: 1,000,000+ words)...")
        text = fetch_wikipedia_data(topics)
        if text:
            # Append if file already exists to accumulate data
            mode = 'a' if os.path.exists(DATA_PATH) else 'w'
            with open(DATA_PATH, mode, encoding='utf-8') as f:
                f.write(text)
    
    with open(DATA_PATH, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    
    cleaned_text = clean_text(text)
    words = cleaned_text.split()
    
    # Target 1,000,000 words for full training on GPU
    limit = 1000000
    if len(words) > limit:
        print(f"Limiting corpus to {limit} words.")
        cleaned_text = " ".join(words[:limit])
    
    print(f"Total words in corpus: {len(cleaned_text.split())}")
    
    # Prepare sequences
    X, y, total_words, max_len, tokenizer = prepare_sequences(cleaned_text, max_sequence_len=10)
    
    # Save config with all hyperparameters
    import json
    config = {
        "max_len": max_len,
        "vocab_size": total_words,
        "embedding_dim": 128,
        "hidden_dim": 256
    }
    if not os.path.exists('models'): os.makedirs('models')
    with open(os.path.join('models', 'config.json'), 'w') as f:
        json.dump(config, f)
        
    print(f"Total unique words: {total_words}")
    print(f"Max sequence length: {max_len}")
    
    # Build Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hyperparameters
    embedding_dim = 128
    hidden_dim = 256
    model = LSTMModel(total_words, embedding_dim, hidden_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    dataset = TensorDataset(torch.LongTensor(X), torch.LongTensor(y))
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)
    
    # Train/Val Split
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)
    
    # Train Model
    print(f"Starting training on {device}...")
    num_epochs = 20
    model.train()
    
    history = {"train_loss": [], "val_loss": []}
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient Clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            total_train_loss += loss.item()
        
        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                total_val_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    # Save History for visualization
    with open(os.path.join('models', 'history.json'), 'w') as f:
        json.dump(history, f)
    
    # Save Model and Tokenizer
    if not os.path.exists('models'):
        os.makedirs('models')
        
    torch.save(model.state_dict(), MODEL_PATH)
    save_tokenizer(tokenizer, TOKENIZER_PATH)
    
    # Save config with all hyperparameters
    import json
    config = {
        "max_len": max_len,
        "vocab_size": total_words,
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim
    }
    with open(os.path.join('models', 'config.json'), 'w') as f:
        json.dump(config, f)
    
    print(f"Model saved to {MODEL_PATH}")
    print(f"Tokenizer saved to {TOKENIZER_PATH}")

if __name__ == "__main__":
    train()
