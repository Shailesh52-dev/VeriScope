import pandas as pd
import os
import json # CRITICAL: Added for parsing SciFact JSONL
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
import torch.nn as nn # Import nn for custom loss
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from datasets import Dataset 
import warnings # For ignoring pandas/sklearn warnings

# Suppress warnings that happen during data processing for clarity
warnings.filterwarnings("ignore")

# --- CRITICAL CONFIGURATION: D-DRIVE PATHS ---
OUTPUT_BASE_PATH = 'D:/FactCheck_ML_Output'
OUTPUT_DIR = os.path.join(OUTPUT_BASE_PATH, 'multi_head_results') 
LOGGING_DIR = os.path.join(OUTPUT_BASE_PATH, 'logs')
FINAL_MODEL_SAVE_PATH = os.path.join(OUTPUT_BASE_PATH, "final_multi_head_model") 
DATA_DIR = 'data' 

# --- NEW: DATA LIMIT CONFIGURATION ---
MAX_DATASET_ROWS = 10000 # Total combined articles used for training
# -------------------------------------

# Ensure output and data directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOGGING_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True) 

# --- General Configuration ---
MODEL_NAME = 'distilbert-base-uncased'
NUM_OUTPUT_HEADS = 4 
MAX_LENGTH = 128
EPOCHS = 3
BATCH_SIZE = 8 


# --- CUSTOM MASKED LOSS FUNCTION (CRITICAL for Partial Supervision) ---
# This ignores labels set to the neutral/mask value (0.5)
class MaskedBCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mask_value = 0.5 # Labels set to 0.5 will be ignored
        
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = (target != self.mask_value).float()
        
        # Calculate standard loss (BCEWithLogitsLoss)
        loss = super().forward(input, target)
        
        # Apply mask and normalize loss by the number of unmasked elements
        masked_loss = loss * mask
        return torch.sum(masked_loss) / torch.sum(mask)


# --- SPECIALIZED SCIFACT LOADER (Incorporated from User Request) ---
def load_scifact_data(file_name):
    """
    Loads SciFact claims_train.jsonl, manually extracts evidence labels from nested JSON,
    and returns a DataFrame ready for merging.
    """
    file_path = os.path.join(DATA_DIR, file_name)
    rows = []

    try:
        # Use provided code logic
        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                # Apply data row limit check from the overall script config
                if i >= MAX_DATASET_ROWS // NUM_OUTPUT_HEADS:
                    break

                obj = json.loads(line)
                claim = obj.get("claim", "").strip()
                evidence = obj.get("evidence", {})

                if not claim:
                    continue

                # Determine evidence specificity signal
                if not evidence:
                    evidence_signal = "none"
                else:
                    labels = [
                        ev.get("label")
                        for ev in evidence.values()
                        if isinstance(ev, dict)
                    ]
                    if any(l in ["SUPPORT", "CONTRADICT"] for l in labels):
                        evidence_signal = "specific"
                    else:
                        evidence_signal = "vague" # Treat claims with evidence but no clear verdict as vague

                rows.append({
                    "text": claim,
                    "evidence_signal": evidence_signal
                })
        
        df = pd.DataFrame(rows).dropna().sample(frac=1).reset_index(drop=True)
        
        # Rename column to match pipeline convention (index 1 is Evidence Specificity)
        df.rename(columns={"evidence_signal": "label_1"}, inplace=True)
        return df[['text', "label_1"]]
        
    except FileNotFoundError:
        print(f"WARNING: SciFact file {file_name} not found. Skipping this head.")
        return None
    except Exception as e:
        print(f"ERROR processing SciFact JSONL: {e}")
        return None
# --- END SPECIALIZED SCIFACT LOADER ---


# --- 1. Load and Transform Data (Optimized Multi-Dataset Loading) ---

def load_signal_data(file_name, text_col, label_col, target_label_index, sep=','):
    """Loads one external dataset and extracts the necessary text and label."""
    file_path = os.path.join(DATA_DIR, file_name)
    try:
        # --- File Type Handling ---
        if file_name.endswith('.jsonl'):
            df = pd.read_json(file_path, lines=True)
        elif file_name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path) 
        # CRITICAL: Handle CSV/TSV/Text file formats
        elif file_name.endswith(('.csv', '.txt')) or sep == '\t': 
             # Use header=None for files without headers (like CoNLL)
             df = pd.read_csv(file_path, sep=sep, header=None) 
        else:
            print(f"WARNING: Unknown file format for {file_name}. Skipping.")
            return None
        
        # Select required columns and drop NA values
        df = df[[text_col, label_col]].dropna().sample(frac=1).reset_index(drop=True)
        
        # Downsample to maximum quarter size to meet the total row limit
        df = df.head(MAX_DATASET_ROWS // NUM_OUTPUT_HEADS)
        
        df.rename(columns={text_col: 'text', label_col: f'label_{target_label_index}'}, inplace=True)
        return df[['text', f'label_{target_label_index}']]
    
    except FileNotFoundError:
        print(f"WARNING: Dataset file {file_name} not found. Skipping this head.")
        return None
    except ValueError as e:
        # This will now print the missing key more clearly if it's a KeyError
        print(f"ERROR loading {file_name} with columns '{text_col}' and '{label_col}': {e}")
        return None

print("1. Loading and merging four specialized datasets...")

# --- Dataset Loading Assumptions (ADJUST COL_NAMES IN YOUR LOCAL FILE!) ---
# 0. Plausibility (FEVER: train.jsonl)
# CRITICAL FIX: Retaining 'label' as requested.
plausibility_data = load_signal_data(file_name='train.jsonl', text_col='claim', label_col='label', target_label_index=0)

# 1. Evidence Specificity (SciFact: claims_train.jsonl) - USE SPECIALIZED LOADER
evidence_data = load_scifact_data(file_name='scifact/claims_train.jsonl') # CALLING SPECIALIZED FUNCTION

# 2. Bias (Media Bias/AllSides-like: labeled_dataset.xlsx)
bias_data = load_signal_data(file_name='labeled_dataset.xlsx', text_col='sentence', label_col='Label_bias', target_label_index=2)

# 3. Uncertainty (CoNLL 2010: biomed_articles_trial)
uncertainty_data = load_signal_data(file_name='biomed_articles_trial', text_col=0, label_col=1, target_label_index=3, sep='\t')

# Combine all loaded dataframes
all_dfs = [df for df in [plausibility_data, evidence_data, bias_data, uncertainty_data] if df is not None]

if not all_dfs:
    print("FATAL ERROR: No valid datasets loaded. Cannot continue training.")
    texts = ["Sample text."] * 80
    labels = [[0.5, 0.5, 0.5, 0.5]] * 80 # Neutral placeholder
else:
    # --- Merge with Outer Join for Partial Supervision ---
    merged_df = all_dfs[0]
    for df in all_dfs[1:]:
        merged_df = pd.merge(merged_df, df, on='text', how='outer')

    # --- CRITICAL ROBUSTNESS FIX: Guarantee all four label columns exist ---
    target_cols = [f'label_{i}' for i in range(NUM_OUTPUT_HEADS)]
    for col in target_cols:
        if col not in merged_df.columns:
            # If a loader failed completely, add the column and initialize it with the mask value
            merged_df[col] = 0.5
    
    # --- Preprocessing: Convert Labels to Binary (0 or 1) ---
    def preprocess_label(df, col_index):
        col = f'label_{col_index}'
        if col in df.columns:
            # 0. Plausibility (FEVER: SUPPORTS=1, REFUTES/NIA=0)
            if col_index == 0:
                 # CRITICAL: Mapping the specific label assumed for FEVER 2018 (SUPPORTED)
                 df[col] = (df[col].astype(str).str.lower() == 'supports').astype(float) 
            # 1. Evidence (SciFact: SPECIFIC/VAGUE/NONE)
            elif col_index == 1:
                 # CRITICAL: Mapping the specific string output from load_scifact_data
                 df[col] = (df[col].astype(str).str.lower() == 'specific').astype(float)
            # 2. Bias (Media Bias: "Non-Biased"=1, "Biased"/Other=0)
            elif col_index == 2:
                 df[col] = (df[col].astype(str).str.lower() == 'non-biased').astype(float) 
            # 3. Uncertainty (CoNLL: 'C'ertain=1, 'U'ncertain=0, or similar)
            elif col_index == 3:
                 df[col] = (df[col].astype(str).str.lower().str.contains('c|certain')).astype(float) 
        return df

    for i in range(NUM_OUTPUT_HEADS):
        # NOTE: This preprocess function is now run AFTER the merge, 
        # but it only acts on the target column if it was successfully loaded.
        merged_df = preprocess_label(merged_df, i)

    # CRITICAL: Fill NaNs (missing labels) with 0.5 for the Masked Loss Function
    # This fills NaNs generated by the outer join (where data was missing across sources)
    merged_df[target_cols] = merged_df[target_cols].fillna(0.5)
    
    # Final data selection and preparation
    texts = merged_df['text'].astype(str).tolist()
    labels = merged_df[target_cols].values.tolist()
    
    # Downsample the final list to meet the MAX_DATASET_ROWS limit 
    if len(texts) > MAX_DATASET_ROWS:
        sample_indices = np.random.choice(len(texts), MAX_DATASET_ROWS, replace=False)
        texts = [texts[i] for i in sample_indices]
        labels = [labels[i] for i in sample_indices]

    print(f"Data consolidation complete. Final samples used: {len(texts)}")
    print(f"WARNING: The training uses {np.mean([np.sum(np.array(l) != 0.5) for l in labels]):.2f} active labels per sample (Partial Supervision).")


# Split the data
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, shuffle=True)

# --- 2. Tokenization ---
print("2. Tokenizing data...")
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=MAX_LENGTH)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=MAX_LENGTH)

class MultiOutputDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.float)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx] 
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = MultiOutputDataset(train_encodings, train_labels)
val_dataset = MultiOutputDataset(val_encodings, val_labels)

# 3. Training Setup 
print("3. Checking for existing model for continuous training (Feedback Loop)...")

if os.path.exists(FINAL_MODEL_SAVE_PATH):
    print(f"   -> Existing model found. Loading model for further fine-tuning.")
    model = DistilBertForSequenceClassification.from_pretrained(FINAL_MODEL_SAVE_PATH, num_labels=NUM_OUTPUT_HEADS)
else:
    print("   -> No existing model found. Starting training from base DistilBERT.")
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_OUTPUT_HEADS)

# Custom Trainer to apply the Masked Loss
class CustomTrainer(Trainer):
    # CRITICAL FIX: Added *args and **kwargs to accept internal arguments (like num_items_in_batch)
    def compute_loss(self, model, inputs, return_outputs=False, *args, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        loss_fct = MaskedBCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1, self.model.config.num_labels))
        return (loss, outputs) if return_outputs else loss


# Custom metrics for Multi-Label/Multi-Head
def compute_metrics(p):
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    labels = p.label_ids
    
    # We must filter out the masked 0.5 labels before computing metrics
    valid_mask = (labels != 0.5)
    
    # Sigmoid and threshold (0.5) to get binary predictions
    preds = (torch.sigmoid(torch.tensor(logits)).numpy() > 0.5).astype(int)
    
    metrics = {}
    
    for i, name in enumerate(["plausibility", "evidence", "bias", "uncertainty"]):
        
        # Select only the valid labels and logits for this head (i)
        valid_labels_i = labels[:, i][valid_mask[:, i]]
        valid_logits_i = logits[:, i][valid_mask[:, i]]

        if len(valid_labels_i) > 0 and (np.max(valid_labels_i) != np.min(valid_labels_i)):
            # Only calculate AUC if there is more than one class present
            auc_score = roc_auc_score(valid_labels_i, valid_logits_i, average='macro', labels=[0, 1])
            metrics[f'auc_{name}'] = auc_score
        else:
             metrics[f'auc_{name}'] = 0.5 # Default neutral score

    # Calculate overall weighted F1 score across all valid predictions
    # Flatten everything, filter by mask, then calculate F1
    flat_labels = labels[valid_mask]
    flat_preds = preds[valid_mask]
    
    f1_weighted = f1_score(flat_labels, flat_preds, average='weighted', zero_division=0)
    metrics['f1_weighted'] = f1_weighted
    
    return metrics


# 4. Training
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,             
    per_device_train_batch_size=BATCH_SIZE,   
    per_device_eval_batch_size=BATCH_SIZE, 
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir=LOGGING_DIR, 
    logging_steps=100,
    eval_strategy="epoch", 
    save_strategy="epoch",
    load_best_model_at_end=True,
    # fp16=torch.cuda.is_available(), # Uncomment for GPU
)

trainer = CustomTrainer( # Use CustomTrainer here!
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

print("4. Starting multi-head training...")
trainer.train()

# 5. Save the Model
trainer.save_model(FINAL_MODEL_SAVE_PATH)
tokenizer.save_pretrained(FINAL_MODEL_SAVE_PATH)
print(f"5. Model saved successfully to: {FINAL_MODEL_SAVE_PATH}")