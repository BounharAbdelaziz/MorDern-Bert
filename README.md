# Approach

1. **Training from scratch**
   - **a)** On Arabic data **only**  
   - **b)** On Moroccan Arabic data **only**  
   - **c)** Mix of both  

2. **Full Finetuning** *(applies only to model 1-a)*  
   - **a)** On Moroccan Arabic data **only**  
   - **b)** Mix of both  

3. **LoRA Finetuning** *(applies only to model 1-a)*  
   - **a)** On Moroccan Arabic data **only**  
   - **b)** Mix of both  

4. **Finetuning of the base model**  
   *(trained on English data: `answerdotai/ModernBERT-base`)*  
   - Perform steps 2 and 3  

# Datasets

1. **Arabic**
    - https://huggingface.co/datasets/wikimedia/wikipedia/viewer/20231101.ar

2. **Moroccan Arabic**
    - atlasia/AL-Atlas-Moroccan-Darija-Pretraining-Dataset
    - atlasia/Social_Media_Darija_DS

# Currently Running
    - Abdelaziz: 
        - Training ModernBERT from scratch on `atlasia/AL-Atlas-Moroccan-Darija-Pretraining-Dataset`
        - Training ModernBERT from scratch on `wikipedia-ar`
        - Finetuning of arabic BERT on `atlasia/AL-Atlas-Moroccan-Darija-Pretraining-Dataset
        - Finetuning of multilingual BERT on `atlasia/AL-Atlas-Moroccan-Darija-Pretraining-Dataset`
