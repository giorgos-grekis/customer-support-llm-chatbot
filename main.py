import random
import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset

# Ρυθμίσεις
SAMPLE_SIZE = None     
BATCH_SIZE = 16 
NUM_EPOCHS = 15
LEARNING_RATE = 0.001


class ChatbotLinearModel(nn.Module):
    def __init__(self, input_size, output_size, first_hidden_size=512, second_hidden_size=128):
        super(ChatbotLinearModel, self).__init__()
        self.fc1 = nn.Linear(input_size, first_hidden_size)
        self.fc2 = nn.Linear(first_hidden_size, second_hidden_size)
        self.fc3 = nn.Linear(second_hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ChatbotAssistant:
    def __init__(self):
        self.model = None
        self.vocabulary = [] 
        self.intens = []      
        self.intens_responses = {} 

    @staticmethod
    def tokenize_and_lemmatize(sentence):
        lemmatizer = nltk.WordNetLemmatizer()
        tokens = nltk.word_tokenize(str(sentence))
        lemmatized_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens]
        return lemmatized_tokens
    
    def bag_of_words(self, tokenized_sentence):
        sentence_words = set(tokenized_sentence)
        # Επιστρέφει 1 αν η λέξη υπάρχει στην πρόταση, αλλιώς 0
        return [1 if word in sentence_words else 0 for word in self.vocabulary]

    def build_vocabulary(self, dataset):
        print(f"Building vocabulary from {len(dataset)} examples...")
        
        all_words = []
        self.intens = []
        self.intens_responses = {}

        for item in dataset:
            instruction = item['instruction']
            intent = item['intent']
            response = item['response']

            if intent not in self.intens:
                self.intens.append(intent)
                self.intens_responses[intent] = []

            if response and response not in self.intens_responses[intent]:
                self.intens_responses[intent].append(response)

            if instruction:
                tokens = self.tokenize_and_lemmatize(instruction)
                all_words.extend(tokens)
        
        self.vocabulary = sorted(set(all_words))
        
        print(f"Vocabulary built: {len(self.vocabulary)} tokens. Intents: {len(self.intens)}")

    def create_tensors(self, dataset):
        # Προσαρμογή για να χρησιμοποιεί το bag_of_words
        X_data = []
        y_data = []

        for item in dataset:
            instruction = item['instruction']
            intent = item['intent']

            if intent not in self.intens:
                continue

            tokens = self.tokenize_and_lemmatize(instruction)
            
            # Δημιουργία BoW vector (λίστα από 0 και 1)
            bow_vector = self.bag_of_words(tokens)
            label_idx = self.intens.index(intent)

            X_data.append(bow_vector)
            y_data.append(label_idx)
        
        return np.array(X_data), np.array(y_data)

    def train_model(self, train_dataset, val_dataset=None):
        if not self.vocabulary:
            print("Error: Vocabulary is empty.")
            return

        print("Preparing training tensors (BoW)...")
        X_train, y_train = self.create_tensors(train_dataset)
        
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.long)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        input_size = len(self.vocabulary)
        output_size = len(self.intens)
        
        self.model = ChatbotLinearModel(input_size=input_size, output_size=output_size)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

        print(f"Starting Linear BoW training for {NUM_EPOCHS} epochs...")
        
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
        X_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_tensor = torch.tensor(y_val, dtype=torch.long)
        
        self.model.eval()
        with torch.inference_mode():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == y_tensor).sum().item() / len(y_tensor)
            print(f"Accuracy: {accuracy * 100:.2f}%")

    

    def process_message(self, input_message):
        if self.model is None: return "Model not loaded."
        
        tokens = self.tokenize_and_lemmatize(input_message)
        # Μετατροπή σε BoW Vector
        bow = self.bag_of_words(tokens)
        
        input_tensor = torch.tensor([bow], dtype=torch.float32)

        self.model.eval()
        with torch.inference_mode():
            logits = self.model(input_tensor)
            probabilities = F.softmax(logits, dim=1)
            top_probs, top_idxs = torch.topk(probabilities, k=2, dim=1)
            
            confidence = top_probs[0][0].item()  
            best_idx = top_idxs[0][0].item()     
             

        print(f"Debug: Best Intent Index: {best_idx}, Confidence: {confidence:.2f}")

        if confidence < 0.60: # Λίγο πιο αυστηρό threshold για BoW
            return "I'm sorry, I didn't understand that. Could you please rephrase?"
     
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
    print("\n--- 2. Training Linear BoW Model ---")
    chatbot.build_vocabulary(train_dataset)
    chatbot.train_model(train_dataset, val_dataset=valid_dataset)
    chatbot.evaluate(test_dataset)

    print("\n--- 3. Chatbot Ready (BoW Edition) ---")
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