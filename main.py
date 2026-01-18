import os
import json
import random
import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset

SAMPLE_SIZE = None
BATCH_SIZE = 16
NUM_EPOCHS = 5        
LEARNING_RATE = 0.0001 
MAX_LEN = 20          
EMBED_DIM = 128       

class TextTransformer(nn.Module):
    def __init__(self, vocab_size, num_classes, d_model=128, nhead=4, num_layers=2, max_len=50):
        super(TextTransformer, self).__init__()
        
        # Μετατροπή αριθμών σε διανύσματα
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Θέση της λέξης (Learnable)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        
        # Transformer Block
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classifier Head
        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(0.3)
        
        self.d_model = d_model
        self.max_len = max_len

    def forward(self, x):
       
    
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        
        # Συνδυασμός Word Embedding + Position Embedding
        # Scale embedding by sqrt(d_model) - standard practice in Transformers
        x = self.embedding(x) * math.sqrt(self.d_model) + self.pos_embedding(positions)
        x = self.dropout(x)
        
        # Πέρασμα από τον Transformer
        x = self.transformer_encoder(x)
        
        # Global Average Pooling
        # Παίρνουμε τον μέσο όρο όλων των λέξεων για να βγάλουμε ένα νόημα για την πρόταση
        x = x.mean(dim=1)
        
        # Τελική πρόβλεψη
        x = self.fc(x)
        return x


class ChatbotAssistant:
    def __init__(self):
        self.model = None
        self.word2idx = {}  # Λεξικό: Λέξη -> Αριθμός
        self.idx2word = {}  # Λεξικό: Αριθμός -> Λέξη
        self.intens = []      
        self.intens_responses = {} 

    @staticmethod
    def tokenize_and_lemmatize(sentence):
        lemmatizer = nltk.WordNetLemmatizer()
        tokens = nltk.word_tokenize(str(sentence).lower()) # convert to str just in case
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
        return lemmatized_tokens
    
  
    def text_to_indices(self, tokenized_sentence):
        indices = []
        for word in tokenized_sentence:
            if word in self.word2idx:
                indices.append(self.word2idx[word])
            else:
                indices.append(self.word2idx["<UNK>"]) # Άγνωστη λέξη
        
        # Padding ή Truncating κόψιμο
        if len(indices) < MAX_LEN:
            # Συμπλήρωμα με 0 (PAD) μέχρι το MAX_LEN
            indices += [self.word2idx["<PAD>"]] * (MAX_LEN - len(indices))
        else:
            # Κόψιμο αν είναι μεγάλο
            indices = indices[:MAX_LEN]
            
        return indices

    def build_vocabulary(self, dataset):
        print(f"Building vocabulary from {len(dataset)} examples...")
        
        all_words = []
        # Αρχικοποίηση λιστών intents
        self.intens = []
        self.intens_responses = {}

        # Συλλογή όλων των λέξεων
        for item in dataset:
            instruction = item['instruction']
            intent = item['intent']
            response = item['response']

            if intent not in self.intens:
                self.intens.append(intent)
                self.intens_responses[intent] = []

            if response and response not in self.intens_responses[intent]:
                self.intens_responses[intent].append(response)

            tokens = self.tokenize_and_lemmatize(instruction)
            all_words.extend(tokens)
        
        # Δημιουργία μοναδικών λέξεων
        unique_words = sorted(set(all_words))
        
        # ΕΙΔΙΚΑ TOKENS
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        
        # Γέμισμα του λεξικού
        for i, word in enumerate(unique_words):
            self.word2idx[word] = i + 2 # Ξεκινάμε από το 2
            
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        
        print(f"Vocabulary built: {len(self.word2idx)} tokens (including special). Intents: {len(self.intens)}")

    def create_tensors(self, dataset):
        sequences = []
        labels = []

        for item in dataset:
            instruction = item['instruction']
            intent = item['intent']

            if intent not in self.intens:
                continue

            tokens = self.tokenize_and_lemmatize(instruction)
            
            seq = self.text_to_indices(tokens)
            label_idx = self.intens.index(intent)

            sequences.append(seq)
            labels.append(label_idx)
        
        # Επιστρέφουμε LongTensor για τα indices (ακέραιοι)
        return np.array(sequences), np.array(labels)

    def train_model(self, train_dataset, val_dataset=None):
        if not self.word2idx:
            print("Error: Vocabulary is empty.")
            return

        print("Preparing training tensors...")
        X_train, y_train = self.create_tensors(train_dataset)
        
        X_tensor = torch.tensor(X_train, dtype=torch.long)
        y_tensor = torch.tensor(y_train, dtype=torch.long)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        # Transformer Model
        vocab_size = len(self.word2idx)
        num_classes = len(self.intens)
        
        self.model = TextTransformer(
            vocab_size=vocab_size,
            num_classes=num_classes,
            d_model=EMBED_DIM,
            max_len=MAX_LEN
        )
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

        print(f"Starting Transformer training for {NUM_EPOCHS} epochs...")
        
        for epoch in range(NUM_EPOCHS):
            self.model.train()
            running_loss = 0.0
            
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {running_loss/len(loader):.4f}")
            
            if val_dataset and (epoch+1) % 5 == 0:
                self.evaluate(val_dataset)

    def evaluate(self, dataset):
        print("Evaluating...")
        X_val, y_val = self.create_tensors(dataset)
        X_tensor = torch.tensor(X_val, dtype=torch.long) # LongTensor
        y_tensor = torch.tensor(y_val, dtype=torch.long)
        
        self.model.eval()
        with torch.inference_mode():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == y_tensor).sum().item() / len(y_tensor)
            print(f"Accuracy: {accuracy * 100:.2f}%")

    def save_model(self, model_path="chatbot_transformer.pth", meta_path="transformer_metadata.json"):
        if self.model is None: return
        
        torch.save(self.model.state_dict(), model_path)
        
        metadata = {
            'vocab_size': len(self.word2idx),
            'num_classes': len(self.intens),
            'word2idx': self.word2idx,    # Αποθηκεύουμε όλο το mapping
            'intens': self.intens,          
            'intens_responses': self.intens_responses,
            'max_len': MAX_LEN,
            'embed_dim': EMBED_DIM
        }
        
        with open(meta_path, 'w') as f:
            json.dump(metadata, f)
        print("Transformer model saved.")

    def load_model(self, model_path="chatbot_transformer.pth", meta_path="transformer_metadata.json"):
        if not os.path.exists(model_path) or not os.path.exists(meta_path):
            print("Model files not found.")
            return False

        with open(meta_path, 'r') as f:
            meta = json.load(f)

        self.word2idx = meta['word2idx']
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.intens = meta['intens']
        self.intens_responses = meta.get('intens_responses', {}) 

        self.model = TextTransformer(
            vocab_size=meta['vocab_size'],
            num_classes=meta['num_classes'],
            d_model=meta['embed_dim'],
            max_len=meta['max_len']
        )
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        print("Transformer loaded successfully.")
        return True

    def process_message(self, input_message):
        if self.model is None: return "Model not loaded."
        
        tokens = self.tokenize_and_lemmatize(input_message)
        seq = self.text_to_indices(tokens)
        seq_tensor = torch.tensor([seq], dtype=torch.long)

        self.model.eval()
        with torch.inference_mode():
            logits = self.model(seq_tensor)
            
            # Μετατροπή σε Πιθανότητες (Softmax)
            # Το dim=1 σημαίνει ότι αθροίζει στο 100% για κάθε πρόταση
            probabilities = F.softmax(logits, dim=1)
            
            # Βρες τις 2 πιθανότερες προβλέψεις (Top-K)
            # top_probs: Οι πιθανότητες (π.χ. [0.65, 0.30])
            # top_idxs: Οι θέσεις τους (π.χ. [5, 12])
            top_probs, top_idxs = torch.topk(probabilities, k=2, dim=1)
            
            confidence = top_probs[0][0].item()  # Η σιγουριά του 1ου (π.χ. 0.65)
            best_idx = top_idxs[0][0].item()     # Το ID του 1ου
            
            second_confidence = top_probs[0][1].item() # Η σιγουριά του 2ου
            second_idx = top_idxs[0][1].item()   # Το ID του 2ου

       
        print(f"Debug: Best Intent Index: {best_idx}, Confidence: {confidence:.2f}")
    


        if confidence < 0.50:
            return "I'm sorry, I didn't understand that. Could you please rephrase?"

     
        if (confidence - second_confidence) < 0.10:
            intent1 = self.intens[best_idx]
            intent2 = self.intens[second_idx]
            return f"I'm not sure. Did you mean '{intent1}' or '{intent2}'?"

        
        predicted_intent = self.intens[best_idx]

        if predicted_intent in self.intens_responses:
            return random.choice(self.intens_responses[predicted_intent])
        
        return f"Detected intent: {predicted_intent} (Confidence: {confidence:.2f})"



def main():
    print("--- 1. Loading Dataset ---")
    full_dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train")
    

    full_dataset = full_dataset.shuffle(seed=42)

    if SAMPLE_SIZE:
        print(f"Using subset of {SAMPLE_SIZE}...")
        full_dataset = full_dataset.select(range(SAMPLE_SIZE))

    train_temp_split = full_dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = train_temp_split["train"]
    temp_dataset = train_temp_split["test"]

    val_test_split = temp_dataset.train_test_split(test_size=0.5, seed=42)
    valid_dataset = val_test_split["train"]
    test_dataset  = val_test_split["test"]

    print(f"Train: {len(train_dataset)} | Val: {len(valid_dataset)} | Test: {len(test_dataset)}")
    
    chatbot = ChatbotAssistant()

    # Training Phase 
    print("\n--- 2. Training Transformer ---")
    chatbot.build_vocabulary(train_dataset)
    chatbot.train_model(train_dataset, val_dataset=valid_dataset)
    chatbot.evaluate(test_dataset)
    chatbot.save_model()
    # ----------------------

    # Loading Phase 
    # chatbot.load_model()

    print("\n--- 3. Chatbot Ready (Transformer Edition) ---")
    while True:
        try:
            msg = input("\nYou: ")
            if msg.lower() == "/exit": break
            response = chatbot.process_message(msg)
            print(f"Bot: {response}")
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()