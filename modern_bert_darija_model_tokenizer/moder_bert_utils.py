import wandb
import torch
from dotenv import load_dotenv
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)


def info(message):
  print("="*30+f"[INFO] {message}"+"="*30)

# dataset to iterator
def batch_iter(ds,batch_size=1000):
  for i in range(0,len(ds),batch_size):
    yield ds[i:i+batch_size]["text"]

def process(examples,new_tokenizer,configs):
  return new_tokenizer(
      examples["text"],
      turncation=True,
      max_length=configs.max_length)

# prepare configs
class Config:
  def __init__(self):
    self.model_name="answerdotai/ModernBERT-base"
    self.max_lenght=4096*2
    self.new_vocab_size=64000
    self.mlm_probability=0.15 # Masked Language Proba (15% of input will be masked)
    self.base_dir="./DarijaModern"
    self.output_dir=self.base_dir+"/model"
    self.num_train_epochs=3
    self.per_device_train_batch_size=32
    self.per_device_eval_batch_size=8
    self.evaluation_strategy="steps"
    self.eval_steps=5000
    self.logging_steps=100
    self.save_steps=5000
    self.save_total_limit=2
    self.learning_rate=5e-2
    self.warmup_steps=500
    self.weight_decay=0.01
    self.report_to="wandb"
    self.run_name="modernbert-darija"
    self.overwrite_output_dir = True
