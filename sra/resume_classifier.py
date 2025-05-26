import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import BertForSequenceClassification, BertTokenizerFast, Trainer, TrainingArguments


class ResumeClassifier:
    def __init__(self, labels: list[str]):
        self.labels = labels
        self.label2id = {label: i for i, label in enumerate(labels)}
        self.id2label = {i: label for i, label in enumerate(labels)}
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=len(labels),
            id2label=self.id2label,
            label2id=self.label2id,
        )

    def preprocess(self, examples):
        # Tokenize the text
        tokenized = self.tokenizer(examples["resume_text"], truncation=True, padding=True, max_length=256)

        # Ensure labels are integers, not lists
        if isinstance(examples["labels"][0], list):
            tokenized["labels"] = [label[0] if isinstance(label, list) else label for label in examples["labels"]]
        else:
            tokenized["labels"] = examples["labels"]

        return tokenized

    def prepare_dataset(self, df: pd.DataFrame) -> Dataset:
        # Clean the dataframe
        df = df.dropna(subset=["resume_text", "label"]).copy()

        # Map labels to IDs
        df["labels"] = df["label"].map(self.label2id)

        # Check for any unmapped labels
        if df["labels"].isna().any():
            print("Warning: Some labels couldn't be mapped. Dropping those rows.")
            df = df.dropna(subset=["labels"])

        # Convert labels to integers
        df["labels"] = df["labels"].astype(int)

        # Create dataset
        dataset = Dataset.from_pandas(df[["resume_text", "labels"]])

        # Apply preprocessing
        dataset = dataset.map(self.preprocess, batched=True)

        # Remove original text column to save memory and avoid conflicts
        dataset = dataset.remove_columns(["resume_text"])

        # Set format for PyTorch
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        return dataset

    def train(self, train_dataset: Dataset, eval_dataset: Dataset = None, output_dir="resume-bert-classifier"):
        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch" if eval_dataset is not None else "no",
            save_strategy="epoch",
            per_device_train_batch_size=8,
            num_train_epochs=3,  # Reduced from 5 to speed up training
            weight_decay=0.01,
            learning_rate=2e-5,  # Added learning rate
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            save_total_limit=2,
            load_best_model_at_end=True if eval_dataset is not None else False,
            metric_for_best_model="eval_loss" if eval_dataset is not None else None,
            remove_unused_columns=False,  # Important: keep this False
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )

        trainer.train()
        self.trainer = trainer

    def train_from_dataframe(self, df: pd.DataFrame, validation_split: float = 0.2, output_dir: str = "resume-bert-classifier"):
        """Train directly from a pandas DataFrame."""
        # Ensure label mapping exists
        if not self.label2id:
            self.labels = df["label"].unique().tolist()
            self.label2id = {label: i for i, label in enumerate(self.labels)}
            self.id2label = {i: label for label, i in self.label2id.items()}

        # Split DataFrame
        train_df, eval_df = train_test_split(df, test_size=validation_split, stratify=df["label"], random_state=42)

        # Convert to HuggingFace datasets
        train_dataset = self.prepare_dataset(train_df)
        eval_dataset = self.prepare_dataset(eval_df)

        # Train using existing logic
        self.train(train_dataset, eval_dataset=eval_dataset, output_dir=output_dir)

    def predict(self, texts: list[str]) -> list[str]:
        # Get the device the model is on
        device = next(self.model.parameters()).device

        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)

        # Move inputs to the same device as the model
        inputs = {k: v.to(device) for k, v in inputs.items()}

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)

        preds = torch.argmax(outputs.logits, dim=1)
        return [self.id2label[i.item()] for i in preds]

    def predict_cpu(self, texts: list[str]) -> list[str]:
        """Alternative predict method that forces CPU usage to avoid MPS issues"""
        # Move model to CPU temporarily
        original_device = next(self.model.parameters()).device
        self.model.to("cpu")

        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)

        preds = torch.argmax(outputs.logits, dim=1)
        result = [self.id2label[i.item()] for i in preds]

        # Move model back to original device
        self.model.to(original_device)

        return result

    def save(self, path: str):
        self.tokenizer.save_pretrained(path)
        self.model.save_pretrained(path)

    def load(self, path: str):
        self.tokenizer = BertTokenizerFast.from_pretrained(path)
        self.model = BertForSequenceClassification.from_pretrained(path)
