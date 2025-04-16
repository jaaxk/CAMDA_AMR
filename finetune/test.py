import torch
from transformers import AutoTokenizer, AutoModel
from transformers.models.bert.configuration_bert import BertConfig

#tokenizer = AutoTokenizer.from_pretrained("/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/finetune/output_1/checkpoint-11800", trust_remote_code=True)
config = BertConfig.from_pretrained("/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/finetune/output_1/checkpoint-11800")
model = AutoModel.from_pretrained("/gpfs/scratch/jvaska/CAMDA_AMR/CAMDA_AMR/finetune/output_1/checkpoint-11800", trust_remote_code=True, config=config)

print(model)
