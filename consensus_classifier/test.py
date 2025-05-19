import transformers
def load_model(model_path):
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=2,
        trust_remote_code=True,
    )
    model.eval()
    

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=250,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )
    return model, tokenizer


model, tokenizer = load_model('/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/finetune/outputs/1000_perspecies/acinetobacter_baumannii/best/')
model, tokenizer = load_model('/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/finetune/outputs/thresh_5e-2_blast_80_pv4/best')
