import torch
from transformers import AutoConfig, AdamW
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from torch.utils.data import DataLoader

if __name__ == "__main__":
    
    lr = 5e-3
    batch_size = 128
    warmup_ratio = 0.05
    n_epochs = 1
    eval_steps = 5000
    save_steps = 5000
    logging_steps = 5000
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    
    dataset = load_dataset("atlasia/AL-Atlas-Moroccan-Darija-Pretraining-Dataset")
    
    model_name = "answerdotai/ModernBERT-base"
    model_shortname = model_name.split("/")[-1]

    # Load model with optimized settings
    model = SentenceTransformer(
        model_name, 
        model_kwargs={
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2"
        },
        config_kwargs={"config": AutoConfig.from_pretrained(model_name)},
    ).to(device)

    model.max_seq_length = 8196
    
    dataset_dict = dataset
    train_dataset = dataset_dict["train"]
    eval_dataset = dataset_dict["test"]
    
    # Optimize data loading
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    
    loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=batch_size).to(device)

    run_name = f"{model_shortname}-{lr}"
    args = SentenceTransformerTrainingArguments(
        output_dir=f"output/{model_shortname}/{run_name}",
        num_train_epochs=n_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        fp16=False,  # Enable mixed precision training
        bf16=True,
        # batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        logging_steps=logging_steps,
        run_name=run_name,
        gradient_accumulation_steps=1,  # Increased for larger effective batch size
    )

    # Use a more efficient optimizer
    optimizer = AdamW(model.parameters(), lr=lr, correct_bias=False)

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        optimizers=(optimizer, None),
    )

    trainer.train()

    model.save_pretrained(f"output/{model_shortname}/{run_name}/final")
    model.push_to_hub(run_name, private=False)
