import os

os.environ['HF_HOME'] = r"E:\huggingface_cache"
os.environ['TRANSFORMERS_CACHE'] = os.path.join(os.environ['HF_HOME'], "transformers")
os.environ['HF_DATASETS_CACHE'] = os.path.join(os.environ['HF_HOME'], "datasets")

# Make sure subfolders exist
os.makedirs(os.environ['TRANSFORMERS_CACHE'], exist_ok=True)
os.makedirs(os.environ['HF_DATASETS_CACHE'], exist_ok=True)

import pickle
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    DistilBertTokenizer, DistilBertForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ================= CONFIG =================
class Config:
    MODEL_NAME = "distilbert-base-multilingual-cased"
    MAX_LENGTH = 256           # shorter sequence for CPU
    BATCH_SIZE = 2             # small batch for CPU
    GRAD_ACCUM_STEPS = 4       # effective batch size 8
    LEARNING_RATE = 2e-5
    EPOCHS = 3                 # fewer epochs for CPU
    WEIGHT_DECAY = 0.01
    DROPOUT = 0.3
    DEVICE = torch.device("cpu")  # force CPU
    SAVE_DIR = "bert_model/saved_model"
    DATA_PATH = "bert_model/dataset.csv"
    SEED = 42

os.makedirs(Config.SAVE_DIR, exist_ok=True)
torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)

# ================= DATASET =================
class KMRLDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ================= MODEL =================
class DistilBERTClassifier(nn.Module):
    def __init__(self, model_name, num_labels, dropout_rate=0.3):
        super().__init__()
        self.bert = DistilBertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            dropout=dropout_rate
        )
        self.config = self.bert.config
    
    def forward(self, input_ids, attention_mask, labels=None):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

# ================= METRICS =================
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1_weighted': f1_score(labels, preds, average='weighted'),
        'f1_macro': f1_score(labels, preds, average='macro')
    }

# ================= CUSTOM TRAINER =================
class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.focal_loss = nn.CrossEntropyLoss()  # optional: focal can be added
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self.class_weights is not None:
            weight_tensor = self.class_weights.to(logits.device)
            loss_fct = CrossEntropyLoss(weight=weight_tensor)
            loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        else:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# ================= DATA LOADING =================
def load_data():
    df = pd.read_csv(Config.DATA_PATH, on_bad_lines='skip', encoding='utf-8-sig')
    if 'english_text' in df.columns and 'malayalam_text' in df.columns:
        df['combined_text'] = df['english_text'].fillna('') + ' ' + df['malayalam_text'].fillna('')
        text_col = 'combined_text'
    else:
        text_col = 'text' if 'text' in df.columns else df.columns[0]
    label_col = 'label' if 'label' in df.columns else df.columns[1]

    df = df.dropna(subset=[text_col, label_col])
    df[text_col] = df[text_col].astype(str)
    df[label_col] = df[label_col].astype(str)
    label_encoder = LabelEncoder()
    df['encoded_labels'] = label_encoder.fit_transform(df[label_col])

    with open(os.path.join(Config.SAVE_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(label_encoder, f)

    logger.info(f"Dataset loaded: {len(df)} samples")
    logger.info(f"Classes: {df[label_col].unique().tolist()}")
    return df, text_col, label_col, label_encoder

# ================= CROSS VALIDATION =================
def cross_validate(df, text_col, label_encoder, n_splits=3):  # reduce folds for CPU
    tokenizer = DistilBertTokenizer.from_pretrained(Config.MODEL_NAME)
    X, y = df[text_col].values, df['encoded_labels'].values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=Config.SEED)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        logger.info(f"Fold {fold+1}/{n_splits}")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_dataset = KMRLDataset(X_train, y_train, tokenizer, Config.MAX_LENGTH)
        val_dataset = KMRLDataset(X_val, y_val, tokenizer, Config.MAX_LENGTH)
        class_weights = torch.FloatTensor(compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)).to(Config.DEVICE)

        model = DistilBERTClassifier(Config.MODEL_NAME, num_labels=len(label_encoder.classes_), dropout_rate=Config.DROPOUT)

        training_args = TrainingArguments(
            output_dir=f"{Config.SAVE_DIR}/fold_{fold}",
            num_train_epochs=Config.EPOCHS,
            per_device_train_batch_size=Config.BATCH_SIZE,
            per_device_eval_batch_size=Config.BATCH_SIZE,
            gradient_accumulation_steps=Config.GRAD_ACCUM_STEPS,
            weight_decay=Config.WEIGHT_DECAY,
            learning_rate=Config.LEARNING_RATE,
            logging_steps=20,
            evaluation_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1_weighted",
            greater_is_better=True,
            report_to="none",
            save_total_limit=2
        )

        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            class_weights=class_weights,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )

        trainer.train()
        eval_res = trainer.evaluate()
        cv_scores.append(eval_res['eval_f1_weighted'])
        logger.info(f"Fold {fold+1} F1 Weighted: {eval_res['eval_f1_weighted']:.4f}")
        torch.cuda.empty_cache()

    mean_cv, std_cv = np.mean(cv_scores), np.std(cv_scores)
    logger.info(f"Cross-validation completed! Mean F1: {mean_cv:.4f} ± {std_cv:.4f}")
    return mean_cv, std_cv

# ================= FINAL TRAINING =================
def train_final_model(df, text_col, label_encoder):
    tokenizer = DistilBertTokenizer.from_pretrained(Config.MODEL_NAME)
    X_train, X_val, y_train, y_val = train_test_split(
        df[text_col].values, df['encoded_labels'].values,
        test_size=0.15, stratify=df['encoded_labels'].values,
        random_state=Config.SEED
    )

    train_dataset = KMRLDataset(X_train, y_train, tokenizer, Config.MAX_LENGTH)
    val_dataset = KMRLDataset(X_val, y_val, tokenizer, Config.MAX_LENGTH)
    class_weights = torch.FloatTensor(compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)).to(Config.DEVICE)
    model = DistilBERTClassifier(Config.MODEL_NAME, num_labels=len(label_encoder.classes_), dropout_rate=Config.DROPOUT)

    training_args = TrainingArguments(
        output_dir=Config.SAVE_DIR,
        num_train_epochs=Config.EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        gradient_accumulation_steps=Config.GRAD_ACCUM_STEPS,
        weight_decay=Config.WEIGHT_DECAY,
        learning_rate=Config.LEARNING_RATE,
        logging_steps=20,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_weighted",
        greater_is_better=True,
        report_to="none",
        save_total_limit=3
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()
    eval_results = trainer.evaluate()
    logger.info(f"Final Model F1 Weighted: {eval_results['eval_f1_weighted']:.4f}")

    model.save_pretrained(Config.SAVE_DIR)
    tokenizer.save_pretrained(Config.SAVE_DIR)

    preds = np.argmax(trainer.predict(val_dataset).predictions, axis=1)
    cm = confusion_matrix(y_val, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(Config.SAVE_DIR, 'confusion_matrix.png'), dpi=300)
    plt.close()

    return eval_results['eval_f1_weighted']

# ================= MAIN =================
def main():
    logger.info("Starting DistilBERT KMRL Department Classification Training on CPU")
    df, text_col, label_col, label_encoder = load_data()
    cv_mean, cv_std = cross_validate(df, text_col, label_encoder)
    final_score = train_final_model(df, text_col, label_encoder)
    logger.info("Training completed successfully!")
    logger.info(f"Cross-validation F1 Score: {cv_mean:.4f} ± {cv_std:.4f}")
    logger.info(f"Final Model F1 Score: {final_score:.4f}")
    logger.info(f"Model saved to: {Config.SAVE_DIR}")

if __name__ == "__main__":
    main()
