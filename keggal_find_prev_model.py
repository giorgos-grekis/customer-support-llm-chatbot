import os

PREVIOUS_EXPERIMENT_ID = "005"
CHECKPOINT_PATH = "/kaggle/input/baby-gpt-checkpoint/baby_gpt_model.pth"

print(f"Έλεγχος για το αρχείο: {CHECKPOINT_PATH}\n")

if os.path.exists(CHECKPOINT_PATH):
    file_size_mb = os.path.getsize(CHECKPOINT_PATH) / (1024 * 1024)
    print(f"Το μοντέλο βρέθηκε.")
    print(f"path: {CHECKPOINT_PATH}")
    print(f"size: {file_size_mb:.2f} MB")
    print("-" * 30)
    print("Μπορείς να προχωρήσεις με την εκπαίδευση!")
    
else:
    print(f"ΣΦΑΛΜΑ: Το αρχείο δεν υπάρχει σε αυτή τη διαδρομή.")
    print("\nΤι υπάρχει μέσα στο /kaggle/input/...")
    
    found_any = False
    for root, dirs, files in os.walk("/kaggle/input"):
        for file in files:
            full_path = os.path.join(root, file)
            print(f"Βρέθηκε αρχείο: {full_path}")
            found_any = True

    print("-" * 30)
    if found_any:
        print("Αντίγραψε μία από τις παραπάνω διαδρομές (paths) και βάλε τη στο CHECKPOINT_PATH.")
    else:
        print("Ο φάκελος /kaggle/input είναι άδειος!")