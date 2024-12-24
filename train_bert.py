import os
import torch

from transformers import (
        AutoTokenizer,
        AutoModelForMaskedLM,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling
    )

from datasets import load_dataset


if __name__ == "__main__":

    # Define hyperparameters
    lr = 5e-3
    batch_size = 128
    warmup_ratio = 0.05
    n_epochs = 3
    max_length = 512
    eval_steps = 5000
    save_steps = 5000
    logging_steps = 5000

    # Define model and dataset paths
    BASE_MODEL = "google-bert/bert-base-multilingual-cased" # "asafaya/bert-large-arabic"
    DATASET_PATH = "atlasia/AL-Atlas-Moroccan-Darija-Pretraining-Dataset"
    
    # Load dataset
    dataset = load_dataset(DATASET_PATH)
    
    # Initialize tokenizer and model from English BERT
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForMaskedLM.from_pretrained(
                                            BASE_MODEL, 
                                            torch_dtype = torch.bfloat16,
            )

    def tokenize_function(examples, max_length=512):

        tokens = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
            return_special_tokens_mask=True
        )
        # print(tokens)
        return tokens

    # Tokenize the dataset
    tokenized_train_dataset = dataset['train'].map(
        tokenize_function,
        remove_columns=dataset['train'].column_names,
        batched=True,
    )
    
    tokenized_eval_dataset = dataset['test'].map(
        tokenize_function,
        remove_columns=dataset['test'].column_names,
        batched=True,
    )

    # Create data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./arabic_bert",
        overwrite_output_dir=True,
        num_train_epochs=n_epochs,
        per_device_train_batch_size=batch_size,
        prediction_loss_only=True,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=warmup_ratio,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        logging_steps=logging_steps,
        gradient_accumulation_steps=1,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        data_collator=data_collator,
    )

    # Train the model
    trainer.train()

    # Save the fine-tuned model and tokenizer
    model_save_path = "./multilingual_bert_finetuned"
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)