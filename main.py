# from google.colab import drive
# drive.mount('/content/drive')


import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
import os
import re



EXPERIMENT_ID = "023"
PREVIOUS_EXPERIMENT_ID = "022"

# colbab path
# BASE_PATH = "/content/drive/MyDrive/BabyGPT_Experiments_EMBED_DIM_512"
# RESULTS_DIR = os.path.join(BASE_PATH, f"results/{EXPERIMENT_ID}/")
# RESULTS_CHECKPOINT_DIR = os.path.join(BASE_PATH, f"results/{PREVIOUS_EXPERIMENT_ID}")
# CHECKPOINT_PATH = os.path.join(RESULTS_CHECKPOINT_DIR, "baby_gpt_model.pth")

# kaggle path
# BASE_PATH = "/kaggle/working"
# RESULTS_DIR = os.path.join(BASE_PATH, f"results/{EXPERIMENT_ID}/")
# RESULTS_CHECKPOINT_DIR = os.path.join(BASE_PATH, f"results/{PREVIOUS_EXPERIMENT_ID}")
# CHECKPOINT_PATH = f"/kaggle/input/baby-gpt-checkpoint/{PREVIOUS_EXPERIMENT_ID}baby_gpt_model.pth"

# local path
RESULTS_DIR = f"results/{EXPERIMENT_ID}/"
RESULTS_CHECKPOINT_DIR = f"results/{PREVIOUS_EXPERIMENT_ID}"
CHECKPOINT_PATH = os.path.join(RESULTS_CHECKPOINT_DIR, f"baby_gpt_model.pth")


DEVICE = "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {DEVICE}")
BATCH_SIZE = 16
NUM_EPOCHS = 1         
LEARNING_RATE = 0.0001 
MAX_LEN = 512                # Το μέγιστο όριο tokens που δεχόμαστε σε μία απαντηση. 
EMBED_DIM = 512 
HEADS = 8                   # Σπάμε το διάνυσμα των 512 σε 8 μικρότερα κομμάτια των 64.
LAYERS = 6                  # Τα πρώτα layers μαθαίνουν απλές λέξεις, τα τελευταία μαθαίνουν το νόημα.
DROPOUT = 0.1




# clean noise from dataset
def clean_text(text):
    text = text.replace("{{Refund Amount}}", "the refund amount")
    text = text.replace("{{Money Amount}}", "the amount")
    text = text.replace("{{Currency Symbol}}", "$")
    text = text.replace("{{Billing}}", "your bill")
    text = text.replace("{{Invoice Number}}", "the invoice number")
    text = text.replace("{{Rebate ID}}", "the rebate ID")
    text = text.replace("{{Compensation ID}}", "the compensation ID")
    text = text.replace("{{Transaction ID}}", "the transaction ID")
    text = text.replace("{{Order Number}}", "the order number")
    text = text.replace("{{Order Status}}", "the order status")
    text = text.replace("{{Tracking Number}}", "the tracking number")
    text = text.replace("{{Shipment Tracking Number}}", "the tracking number")
    text = text.replace("{{Shipping Address}}", "your shipping address")
    text = text.replace("{{Delivery Country}}", "your country")
    text = text.replace("{{Delivery City}}", "your city")
    text = text.replace("{{Destination}}", "the destination")
    text = text.replace("{{Store Location}}", "our store")
    text = text.replace("{{Carrier Name}}", "the shipping carrier")
    text = text.replace("{{Estimated Delivery Time}}", "the estimated delivery time")
    text = text.replace("{{Delivery Time}}", "delivery time")
    text = text.replace("{{Account Name}}", "your account")
    text = text.replace("{{Account Number}}", "your account number")
    text = text.replace("{{Account ID}}", "your account ID")
    text = text.replace("{{Customer ID}}", "your customer ID")
    text = text.replace("{{Username}}", "your username")
    text = text.replace("{{Password}}", "your password")
    text = text.replace("{{PIN}}", "your PIN")
    text = text.replace("{{PIN Code}}", "your PIN code")
    text = text.replace("{{Access Key}}", "your access key")
    text = text.replace("{{User Profile}}", "your profile")
    text = text.replace("{{Membership Type}}", "your membership")
    text = text.replace("{{Account Type}}", "your account type")
    text = text.replace("{{Date}}", "the date")
    text = text.replace("{{Year}}", "this year")
    text = text.replace("{{Month}}", "this month")
    text = text.replace("{{Timeframe}}", "a few days")
    text = text.replace("{{Business Hours}}", "business hours")
    text = text.replace("{{Opening Time}}", "9 AM")
    text = text.replace("{{Closing Time}}", "5 PM")
    text = text.replace("{{number of days}}", "a few days")
    text = text.replace("{{soon}}", "shortly")
    text = text.replace("{{Website URL}}", "our website")
    text = text.replace("{{Login Page URL}}", "the login page")
    text = text.replace("{{Support Page URL}}", "our support page")
    text = text.replace("{{Contact Page URL}}", "our contact page")
    text = text.replace("{{Help Center URL}}", "our help center")
    text = text.replace("{{Password Reset Page URL}}", "the password reset page")
    text = text.replace("{{Account Recovery Page URL}}", "the recovery page")
    text = text.replace("{{Live Chat URL}}", "our live chat")
    text = text.replace("{{Link}}", "this link")
    text = text.replace("{{Customer Support Phone Number}}", "our support line")
    text = text.replace("{{Company Phone Number}}", "our office number")
    text = text.replace("{{Toll-Free Number}}", "our toll-free number")
    text = text.replace("{{Email Address}}", "our email")
    text = text.replace("{{Customer Support Email}}", "support@example.com")
    text = text.replace("{{Complaint Email Address}}", "complaints@example.com")
    text = text.replace("{{Contact Us}}", "contact us")
    text = text.replace("{{Company Name}}", "our company")
    text = text.replace("{{CompanyName}}", "our company")
    text = text.replace("{{Agent Name}}", "Alex") 
    text = text.replace("{{Client Name}}", "Customer")
    text = text.replace("{{User Name}}", "Customer")
    text = text.replace("{{Person Name}}", "the person")
    text = text.replace("{{Salutation}}", "Hello")
    text = text.replace("{{Product Name}}", "the product")
    text = text.replace("{{Product/Service Name}}", "the service")
    text = text.replace("{{Cancellation Policy}}", "our cancellation policy")
    text = text.replace("{{Return Policy}}", "our return policy")
    text = text.replace("{{Refund Policy}}", "our refund policy")
    text = text.replace("{{Feature 1}}", "the first feature")
    text = text.replace("{{Feature 2}}", "the second feature")
    text = text.replace("{{Feature 3}}", "the third feature")
    text = re.sub(r'\{\{.*?\}\}', "the details", text)
    text = " ".join(text.split())
    return text


if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
    print(f"Created directory: {RESULTS_DIR}")

# Tokenizer add special tokens
print("Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': '<PAD>', 'sep_token': '<SEP>', 'bos_token': '<SOS>', 'eos_token': '<EOS>'})
VOCAB_SIZE = len(tokenizer)
print(f"Vocabulary Size: {VOCAB_SIZE}")


# DATASET CLASS
class GPTChatDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_len):
        self.data = []
        for item in hf_dataset:
            q = item['instruction']
            a = item['response']
            
            # clean dataset from noise
            a = clean_text(a)
            
            full_text = f"{tokenizer.bos_token} {q} {tokenizer.sep_token} {a} {tokenizer.eos_token}"
            
            encoded = tokenizer(
                full_text,
                truncation=True,
                max_length=max_len,
                padding="max_length",
                return_tensors="pt"
            )
            self.data.append(encoded['input_ids'].squeeze(0)) # [1, 50] -> [50]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        token_ids = self.data[idx]
        # Next-Token Prediction: Input = [<SOS>, Q, <SEP>, A], Target = [Q, <SEP>, A, <EOS>]
        return token_ids[:-1], token_ids[1:]


# Optimized
class BabyGPT(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, max_len, dropout=0.1):
        super(BabyGPT, self).__init__() # Κληρονομεί από το nn.Module
        self.embedding = nn.Embedding(vocab_size, d_model) # Μετατρέπει τα τokens σε διανύσματα.
        self.pos_embedding = nn.Embedding(max_len, d_model) # Προσθέτει πληροφορία θέσης (position) σε κάθε token.
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            batch_first=True, 
            dropout=dropout, 
            norm_first=True, # Pre-Layer Normalization
            activation="gelu")
        # Στοιβάζεις 4 ΤΕΤΟΙΑ ΙΔΙΑ layers το ένα πάνω στο άλλο, τρία διανύσματα (vectors) => Query (Q), Key (K), Value (V)
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
        
        # Κατά την εκπαίδευση κρύβουμε τις επόμενες λέξεις  
        tgt_mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(x.device)
        
        output = self.transformer_decoder(x, x, tgt_mask=tgt_mask)
        output = self.layer_norm(output)
        
        return self.fc_out(output)


# TRAINER & CHAT
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
        
        # Test Loader
        self.test_loader = DataLoader(GPTChatDataset(test_ds, tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=False)
        
        # Init Model
        self.model = BabyGPT(VOCAB_SIZE, EMBED_DIM, HEADS, LAYERS, MAX_LEN, DROPOUT).to(DEVICE)


        if os.path.exists(CHECKPOINT_PATH):
            print(f"\nFound existing model at: {CHECKPOINT_PATH}")
            print("Loading weights to continue training...")
            try:
                self.model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
                print("Model loaded successfully!")
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Starting from scratch...")
        else:
            print(f"\nNo checkpoint found at {CHECKPOINT_PATH}")
            print("Starting training from scratch (New Model)...")
        
        # Softmax: Μετατρέπει τις ακατέργαστες εξόδους του μοντέλου σε πιθανότητες που αθροίζουν στο 100% (1.0) +
        # Log Loss: Συγκρίνει αυτές τις πιθανότητες με την πραγματική απάντηση. 
        # ignore_index=tokenizer.pad_token_id => αγνοούμε το padding <PAD> κατά τον υπολογισμό του loss   
        criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        optimizer = optim.AdamW(self.model.parameters(), lr=LEARNING_RATE)

        log_path = os.path.join(RESULTS_DIR, "train_log.txt")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("Epoch | Train Loss | Val Loss\n")
            f.write("-" * 35 + "\n")
        
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


            # append στο αρχείο log
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"{EXPERIMENT_ID} | {train_loss/len(train_loader):.4f} | {val_loss/len(val_loader):.4f}\n")

            # Save Model στο σωστό φάκελο
            model_save_path = os.path.join(RESULTS_DIR, "baby_gpt_model.pth")
            torch.save(self.model.state_dict(), model_save_path)
            print(f" -> Model Saved at: {model_save_path}")

        self.run_test()

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


        test_results_path = os.path.join(RESULTS_DIR, "test_results.txt")
        with open(test_results_path, "w", encoding="utf-8") as f:
            f.write("--- FINAL TEST RESULTS ---\n")
            f.write(f"Final Test Loss: {avg_test_loss:.4f}\n")
            f.write(f"Model Perplexity: {perplexity:.2f}\n")
            f.write("(Lower is better)\n")
        
        print(f"Test results saved to: {test_results_path}")

    def chat(self, user_input):

        if self.model is None: 
            model_path = os.path.join(RESULTS_DIR, "baby_gpt_model.pth")
            if os.path.exists(model_path):
                 print(f"Loading model from {model_path}...")
                 self.model = BabyGPT(VOCAB_SIZE, EMBED_DIM, HEADS, LAYERS, MAX_LEN, DROPOUT).to(DEVICE)
                 self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            else:
                return f"Model not loaded and no checkpoint found. Path {model_path}."
            
        self.model.eval()
        
        temperature = 0.7 # αυτοπεποίθηση στις προβλέψεις
        top_k = 50
        repetition_penalty = 1.15
        penalty_window = 30
        
        # Προετοιμασία Prompt
        prompt = f"{tokenizer.bos_token} {user_input} {tokenizer.sep_token}"
        # prompt = f"{tokenizer.bos_token} Customer Support Conversation. User: {user_input} Agent:"
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        
        generated = input_ids 

        # Loop παραγωγής λέξη-λέξη autoregressive με repetition penalty, temperature και top-k sampling
        for _ in range(MAX_LEN):
            with torch.no_grad():
                if generated.size(1) > MAX_LEN: 
                    generated = generated[:, -MAX_LEN:]

                output = self.model(generated)
                next_token_logits = output[0, -1, :]

                # Τιμωρούμε μόνο τα tokens που βρίσκονται στο penalty_window
                start_idx = max(0, generated.size(1) - penalty_window)
                current_window_tokens = generated[0, start_idx:].tolist()

                # Penalize Repetition
                for token in set(current_window_tokens):
                    score = next_token_logits[token]

                    # Αν το score είναι αρνητικό (π.χ. -10), πολλαπλασιάζουμε για να γίνει πιο μικρό (-12)
                    if score < 0:
                        next_token_logits[token] = score * repetition_penalty
                    # Αν είναι θετικό, διαιρούμε
                    else:
                        next_token_logits[token] = score / repetition_penalty

                   
                    # next_token_logits[token] /= repetition_penalty

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
    
    # # Εκπαίδευση
    # bot.train()
    
    # Chat
    print("\n--- Generative Chatbot Ready! ---")
    while True:
        msg = input("You: ")
        if msg == "/exit": break
        print(f"Bot: {bot.chat(msg)}")