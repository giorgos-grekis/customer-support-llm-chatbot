import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer

# --- ΡΥΘΜΙΣΕΙΣ ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {DEVICE}")
BATCH_SIZE = 16
NUM_EPOCHS = 20        
LEARNING_RATE = 1e-4  
MAX_LEN = 50
EMBED_DIM = 256
HEADS = 8
LAYERS = 4
DROPOUT = 0.1

# --- 1. TOKENIZER SETUP ---
print("Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': '<PAD>', 'sep_token': '<SEP>', 'bos_token': '<SOS>', 'eos_token': '<EOS>'})
VOCAB_SIZE = len(tokenizer)
print(f"Vocabulary Size: {VOCAB_SIZE}")

# ---------------------------------------------------------
# 2. DATASET CLASS
# ---------------------------------------------------------
class GPTChatDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_len):
        self.data = []
        for item in hf_dataset:
            q = item['instruction']
            a = item['response']
            
            # Καθαρισμός dataset από placeholders
            a = a.replace("{{Refund Amount}}", "the amount")
            a = a.replace("{{Delivery Country}}", "your country")
            a = a.replace("{{Order Number}}", "the order")
            a = a.replace("{{Date}}", "soon")
            a = a.replace("{{Account Name}}", "your account")
            a = a.replace("{{Customer Support Phone Number}}", "our support line")
            a = a.replace("{{Website URL}}", "our website")
            
            full_text = f"{tokenizer.bos_token} {q} {tokenizer.sep_token} {a} {tokenizer.eos_token}"
            
            encoded = tokenizer(
                full_text,
                truncation=True,
                max_length=max_len,
                padding="max_length",
                return_tensors="pt"
            )
            self.data.append(encoded['input_ids'].squeeze(0))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        token_ids = self.data[idx]
        return token_ids[:-1], token_ids[1:]

# ---------------------------------------------------------
# 3. ΤΟ ΜΟΝΤΕΛΟ (BabyGPT - Optimized)
# ---------------------------------------------------------
class BabyGPT(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, max_len, dropout=0.1):
        super(BabyGPT, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout, norm_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        
        x = self.embedding(x) * math.sqrt(self.d_model) + self.pos_embedding(positions)
        x = self.dropout(x)
        
        tgt_mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(x.device)
        
        output = self.transformer_decoder(x, x, tgt_mask=tgt_mask)
        output = self.layer_norm(output)
        
        return self.fc_out(output)

# ---------------------------------------------------------
# 4. TRAINER & CHAT
# ---------------------------------------------------------
class ChatbotTrainer:
    def __init__(self):
        self.model = None
        self.test_loader = None # Αποθήκευση για χρήση στο run_test

    def train(self):
        print("--- Loading Dataset ---")
        full_dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train")
        # full_dataset = full_dataset.shuffle(seed=42).select(range(5000)) # Uncomment για γρήγορο τεστ
        full_dataset = full_dataset.shuffle(seed=42)
        
        # Split: Train (80%), Val (10%), Test (10%)
        # 1. Κόβουμε 20% για Temp (που θα γίνει Val + Test)
        train_temp = full_dataset.train_test_split(test_size=0.2, seed=42)
        train_ds = train_temp['train']
        temp_ds = train_temp['test']
        
        # 2. Κόβουμε το Temp στη μέση (50/50) για Val και Test
        val_test = temp_ds.train_test_split(test_size=0.5, seed=42)
        val_ds = val_test['train']
        test_ds = val_test['test']
        
        print(f"Sizes -> Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

        # DataLoaders
        train_loader = DataLoader(GPTChatDataset(train_ds, tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(GPTChatDataset(val_ds, tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=False)
        
        # --- Δημιουργία Test Loader ---
        self.test_loader = DataLoader(GPTChatDataset(test_ds, tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=False)
        
        # Init Model
        self.model = BabyGPT(VOCAB_SIZE, EMBED_DIM, HEADS, LAYERS, MAX_LEN, DROPOUT).to(DEVICE)
        
        criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        optimizer = optim.AdamW(self.model.parameters(), lr=LEARNING_RATE)
        
        print("\n--- Starting Training ---")
        for epoch in range(NUM_EPOCHS):
            self.model.train()
            train_loss = 0
            for x, y in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                output = self.model(x)
                loss = criterion(output.view(-1, VOCAB_SIZE), y.view(-1))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    output = self.model(x)
                    loss = criterion(output.view(-1, VOCAB_SIZE), y.view(-1))
                    val_loss += loss.item()
            
            print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")

            # ---Save Model ---
            if (epoch+1) % 2 == 0: # Σώζουμε κάθε 2 εποχές
                torch.save(self.model.state_dict(), "baby_gpt_improved.pth")
                print(" -> Model Checkpoint Saved.")

        # ---  Τρέχουμε το τελικό Test ---
        self.run_test()

    # --- Η συνάρτηση run_test ---
    def run_test(self):
        print("\n--- Running Final Evaluation on Test Set ---")
        if self.test_loader is None:
            print("Test loader not found!")
            return

        self.model.eval()
        criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        test_loss = 0
        
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                output = self.model(x)
                loss = criterion(output.view(-1, VOCAB_SIZE), y.view(-1))
                test_loss += loss.item()
        
        avg_test_loss = test_loss / len(self.test_loader)
        perplexity = math.exp(avg_test_loss)
        
        print(f"Final Test Loss: {avg_test_loss:.4f}")
        print(f"Model Perplexity: {perplexity:.2f}")
        print("(Perplexity closer to 1.0 is better - Aim for < 20)")

    def chat(self, user_input):
        if self.model is None: return "Model not loaded."
        self.model.eval()
        
        temperature = 0.1  
        top_k = 50
        repetition_penalty = 1.2
        
        prompt = f"{tokenizer.bos_token} {user_input} {tokenizer.sep_token}"
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        
        generated = input_ids 

        for _ in range(MAX_LEN):
            with torch.no_grad():
                if generated.size(1) > MAX_LEN: 
                    generated = generated[:, -MAX_LEN:]

                output = self.model(generated)
                next_token_logits = output[0, -1, :]
                
                # Penalize Repetition
                for token in set(generated[0].tolist()):
                    next_token_logits[token] /= repetition_penalty

                # Temperature & Top-K
                next_token_logits = next_token_logits / temperature
                v, _ = torch.topk(next_token_logits, top_k)
                next_token_logits[next_token_logits < v[-1]] = float('-inf')

                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                if next_token.item() == tokenizer.eos_token_id:
                    break
                
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
        
        response_ids = generated[0].tolist()
        try:
            sep_idx = response_ids.index(tokenizer.sep_token_id)
            bot_response_ids = response_ids[sep_idx+1:]
        except ValueError:
            bot_response_ids = response_ids
            
        return tokenizer.decode(bot_response_ids, skip_special_tokens=True)

if __name__ == "__main__":
    bot = ChatbotTrainer()
    
    # 1. Εκπαίδευση
    bot.train()
    
    # 2. Chat
    print("\n--- Generative Chatbot Ready! ---")
    while True:
        msg = input("You: ")
        if msg == "/exit": break
        print(f"Bot: {bot.chat(msg)}")