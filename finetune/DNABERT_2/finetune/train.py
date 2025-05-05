import os
import csv
import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, Sequence, Tuple, List, Union

import torch
import transformers
import sklearn
import numpy as np
from torch.utils.data import Dataset
import wandb

from transformers import TrainerCallback

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    use_lora: bool = field(default=False, metadata={"help": "whether to use LoRA"})
    lora_r: int = field(default=8, metadata={"help": "hidden dimension for LoRA"})
    lora_alpha: int = field(default=32, metadata={"help": "alpha for LoRA"})
    lora_dropout: float = field(default=0.05, metadata={"help": "dropout rate for LoRA"})
    lora_target_modules: str = field(default="query,value", metadata={"help": "where to perform LoRA"})
    res_weight: float = field(default=None, metadata={"help": "weight for resistance"})
    int_weight: float = field(default=None, metadata={"help": "weight for intermediate"})
    sus_weight: float = field(default=None, metadata={"help": "weight for susceptible"})


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    kmer: int = field(default=-1, metadata={"help": "k-mer for input sequence. -1 means not using k-mer."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    run_name: str = field(default="run")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length."})
    gradient_accumulation_steps: int = field(default=1)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    num_train_epochs: int = field(default=1)
    fp16: bool = field(default=False)
    logging_steps: int = field(default=100)
    save_steps: int = field(default=100)
    eval_steps: int = field(default=100)
    evaluation_strategy: str = field(default="steps"),
    warmup_steps: int = field(default=50)
    weight_decay: float = field(default=0.01)
    learning_rate: float = field(default=1e-4)
    save_total_limit: int = field(default=3)
    load_best_model_at_end: bool = field(default=True)
    output_dir: str = field(default="output")
    find_unused_parameters: bool = field(default=False)
    checkpointing: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=False)
    eval_and_save_results: bool = field(default=True)
    save_model: bool = field(default=False)
    seed: int = field(default=42)
    

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


"""
Get the reversed complement of the original DNA sequence.
"""
def get_alter_of_dna_sequence(sequence: str):
    MAP = {"A": "T", "T": "A", "C": "G", "G": "C"}
    # return "".join([MAP[c] for c in reversed(sequence)])
    return "".join([MAP[c] for c in sequence])

"""
Transform a dna sequence to k-mer string
"""
def generate_kmer_str(sequence: str, k: int) -> str:
    """Generate k-mer string from DNA sequence."""
    return " ".join([sequence[i:i+k] for i in range(len(sequence) - k + 1)])


"""
Load or generate k-mer string for each DNA sequence. The generated k-mer string will be saved to the same directory as the original data with the same name but with a suffix of "_{k}mer".
"""
def load_or_generate_kmer(data_path: str, texts: List[str], k: int) -> List[str]:
    """Load or generate k-mer string for each DNA sequence."""
    kmer_path = data_path.replace(".csv", f"_{k}mer.json")
    if os.path.exists(kmer_path):
        logging.warning(f"Loading k-mer from {kmer_path}...")
        with open(kmer_path, "r") as f:
            kmer = json.load(f)
    else:        
        logging.warning(f"Generating k-mer...")
        kmer = [generate_kmer_str(text, k) for text in texts]
        with open(kmer_path, "w") as f:
            logging.warning(f"Saving k-mer to {kmer_path}...")
            json.dump(kmer, f)
        
    return kmer

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, 
                 data_path: str, 
                 tokenizer: transformers.PreTrainedTokenizer, 
                 kmer: int = -1):

        super(SupervisedDataset, self).__init__()

        # load data from the disk
        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]
        if len(data[0]) == 2:
            # data is in the format of [text, label]
            logging.warning("Perform single sequence classification...")
            texts = [d[0] for d in data]
            labels = [int(d[1]) for d in data]
        elif len(data[0]) == 3:
            # data is in the format of [text1, text2, label]
            logging.warning("Perform sequence-pair classification...")
            texts = [[d[0], d[1]] for d in data]
            labels = [int(d[2]) for d in data]
        elif len(data[0]) == 5:
            # full dataset
            logging.warning("Perform classification with num_hits and species...")
            texts = [d[0] for d in data]
            label_mapping = {'Resistant': 0, 'Intermediate': 1, 'Susceptible': 2}
            labels = [label_mapping[d[-1]] for d in data]
            num_hits = [int(d[1]) for d in data]
            species = [d[3] for d in data]
            # Use a fixed mapping for species for consistency with inference
            species_mapping = {
                'klebsiella_pneumoniae': 0,
                'streptococcus_pneumoniae': 1,
                'escherichia_coli': 2,
                'campylobacter_jejuni': 3,
                'salmonella_enterica': 4,
                'neisseria_gonorrhoeae': 5,
                'staphylococcus_aureus': 6,
                'pseudomonas_aeruginosa': 7,
                'acinetobacter_baumannii': 8 
            }
            try:
                species = [species_mapping[s] for s in species]
            except KeyError as e:
                print('WARNING: Species input is numeric, skipping species mapping')
                species = [int(s) for s in species]
            # Hardcoded normalization for num_hits: range 0-496
            num_hits = [(float(nh) - 0.0) / (496.0 - 0.0) for nh in num_hits]
            self.num_hits = num_hits
            self.species = species
            # Save mapping for reference:
            self.species_mapping = species_mapping

        elif len(data[0]) == 6:
            # sequence	num_hits	accession	species	antibiotic	phenotype
            # num_hits already normalized ((float(num_hits) - 0.0) / (496.0 - 0.0)) and species, antibiotic, and phenotype converted to integer:
            """ species_mapping = {
                'klebsiella_pneumoniae': 0,
                'streptococcus_pneumoniae': 1,
                'escherichia_coli': 2,
                'campylobacter_jejuni': 3,
                'salmonella_enterica': 4,
                'neisseria_gonorrhoeae': 5,
                'staphylococcus_aureus': 6,
                'pseudomonas_aeruginosa': 7,
                'acinetobacter_baumannii': 8 
                }
            antibiotic_mapping = {'GEN': 1, 
                'ERY': 2,
                'CAZ': 3,
                'TET': 4,
                'tetracycline': 4
            }

            label_mapping = {'Resistant': 0, 'Intermediate': 0, 'Susceptible': 1} #treating intermediate as resistant

            DONT UNCOMMENT THIS, THESE ARE ALREADY CONVERTED
            
            """

            logging.warning("Performing classification with num_hits, species, and antibiotic...")
            texts = [d[0] for d in data]
            num_hits = [float(d[1]) for d in data]
            species = [int(d[3]) for d in data]
            antibiotic = [int(d[4]) for d in data]
            labels = [int(d[-1]) for d in data]
            self.num_hits = num_hits
            self.species = species
            self.antibiotic = antibiotic

        else:
            raise ValueError("Data format not supported.")
        
        if kmer != -1:
            # only write file on the first process
            if torch.distributed.get_rank() not in [0, -1]:
                torch.distributed.barrier()

            logging.warning(f"Using {kmer}-mer as input...")
            texts = load_or_generate_kmer(data_path, texts, kmer)

            if torch.distributed.get_rank() == 0:
                torch.distributed.barrier()

        output = tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )

        self.input_ids = output["input_ids"]
        self.attention_mask = output["attention_mask"]
        self.labels = labels
        self.num_labels = len(set(labels))

        logging.warning("Number of labels:", self.num_labels)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i], num_hits=self.num_hits[i], species=self.species[i], antibiotic=self.antibiotic[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    class_weights: Optional[torch.Tensor] = None

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))

        # New: Extract additional fields
        num_hits = [instance["num_hits"] for instance in instances]
        species = [instance["species"] for instance in instances]
        antibiotic = [instance["antibiotic"] for instance in instances]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )

        num_hits = torch.tensor(num_hits, dtype=torch.float).unsqueeze(1)  # shape: (batch_size, 1)
        species = torch.tensor(species, dtype=torch.long).unsqueeze(1)
        antibiotic = torch.tensor(antibiotic, dtype=torch.long).unsqueeze(1)
        labels = torch.Tensor(labels).long()

        return dict(
            input_ids=input_ids,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            labels=labels,
            num_hits=num_hits,
            antibiotic=antibiotic,
            species=species,
            class_weights=self.class_weights
        )
"""
Manually calculate the accuracy, f1, matthews_correlation, precision, recall with sklearn.
"""
def calculate_metric_with_sklearn(predictions: np.ndarray, labels: np.ndarray):
    valid_mask = labels != -100  # Exclude padding tokens (assuming -100 is the padding token ID)
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]

    calculate_metric_with_sklearn.last_preds = predictions
    calculate_metric_with_sklearn.last_labels = labels

    return {
        "accuracy": sklearn.metrics.accuracy_score(valid_labels, valid_predictions),
        "f1": sklearn.metrics.f1_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "matthews_correlation": sklearn.metrics.matthews_corrcoef(
            valid_labels, valid_predictions
        ),
        "precision": sklearn.metrics.precision_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "recall": sklearn.metrics.recall_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "confusion_matrix": sklearn.metrics.confusion_matrix(
            valid_labels, valid_predictions
        ).tolist(),
    }

# from: https://discuss.huggingface.co/t/cuda-out-of-memory-when-using-trainer-with-compute-metrics/2941/13
def preprocess_logits_for_metrics(logits:Union[torch.Tensor, Tuple[torch.Tensor, Any]], _):
    if isinstance(logits, tuple):  # Unpack logits if it's a tuple
        logits = logits[0]

    if logits.ndim == 3:
        # Reshape logits to 2D if needed
        logits = logits.reshape(-1, logits.shape[-1])

    return torch.argmax(logits, dim=-1)


"""
Compute metrics used for huggingface trainer.
""" 
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return calculate_metric_with_sklearn(predictions, labels)



def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    class_weights = None
    if model_args.res_weight is not None:   
        class_weights = [model_args.res_weight, model_args.sus_weight]
        class_weights = torch.tensor(class_weights, dtype=torch.float)
        logging.warning('USING CLASS WEIGHTS: ', class_weights)

    try:
        if torch.distributed.get_rank() == 0:
            wandb.init(
            project="AMR-DNABERT2-finetune",
            name= os.getenv('WANDB_NAME', default=f'lr: {training_args.learning_rate}'),
            config={
            "learning_rate": training_args.learning_rate,
            "epochs": training_args.num_train_epochs,
            "per_device_train_batch_size": training_args.per_device_train_batch_size,
        }
        )
        #config = wandb.config
    except RuntimeError:
        print('Not using DDP!')
        wandb.init(
            project="AMR-DNABERT2-finetune",
            name= os.getenv('WANDB_NAME', default=f'lr: {training_args.learning_rate}'),
            config={
            "learning_rate": training_args.learning_rate,
            "epochs": training_args.num_train_epochs,
            "per_device_train_batch_size": training_args.per_device_train_batch_size,
            }
        )

    # load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

    if "InstaDeepAI" in model_args.model_name_or_path:
        tokenizer.eos_token = tokenizer.pad_token

    # define datasets and data collator
    train_dataset = SupervisedDataset(tokenizer=tokenizer, 
                                      data_path=os.path.join(data_args.data_path, "train.csv"), 
                                      kmer=data_args.kmer)
    val_dataset = SupervisedDataset(tokenizer=tokenizer, 
                                     data_path=os.path.join(data_args.data_path, "dev.csv"), 
                                     kmer=data_args.kmer)
    test_dataset = SupervisedDataset(tokenizer=tokenizer, 
                                     data_path=os.path.join(data_args.data_path, "test.csv"), 
                                     kmer=data_args.kmer)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, class_weights=class_weights)


    # load model
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        num_labels=train_dataset.num_labels,
        trust_remote_code=True,
    )

    # configure LoRA
    if model_args.use_lora:
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=list(model_args.lora_target_modules.split(",")),
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="SEQ_CLS",
            inference_mode=False,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # define trainer
    trainer = transformers.Trainer(model=model,
                                   tokenizer=tokenizer,
                                   args=training_args,
                                   preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                                   compute_metrics=compute_metrics,
                                   train_dataset=train_dataset,
                                   eval_dataset=val_dataset,
                                   data_collator=data_collator,
                                   callbacks=[WandbConfusionMatrixCallback()])
    trainer.train()

    if training_args.save_model:
        trainer.save_state()
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    # get the evaluation results from trainer
    if training_args.eval_and_save_results:
        results_path = os.path.join(training_args.output_dir, "results", training_args.run_name)
        results = trainer.evaluate(eval_dataset=test_dataset)
        os.makedirs(results_path, exist_ok=True)
        with open(os.path.join(results_path, "eval_results.json"), "w") as f:
            json.dump(results, f)

    # Always save the best model at the end to output_dir/best -- NEW
    best_dir = os.path.join(training_args.output_dir, "best")
    os.makedirs(best_dir, exist_ok=True)
    trainer.model.save_pretrained(best_dir)
    if hasattr(trainer, 'tokenizer') and trainer.tokenizer is not None:
        trainer.tokenizer.save_pretrained(best_dir)

class WandbConfusionMatrixCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        try:
            if torch.distributed.get_rank() == 0 and hasattr(calculate_metric_with_sklearn, "last_preds"):
                wandb.log({
                    "eval/confusion_matrix": wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=calculate_metric_with_sklearn.last_labels,
                        preds=calculate_metric_with_sklearn.last_preds,
                        class_names = ['Resistant', 'Susceptible']
                            )
                        })
        except RuntimeError:
            if hasattr(calculate_metric_with_sklearn, "last_preds"):
                wandb.log({
                    "eval/confusion_matrix": wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=calculate_metric_with_sklearn.last_labels,
                        preds=calculate_metric_with_sklearn.last_preds,
                        class_names = ['Resistant', 'Susceptible']
                            )
                        })



if __name__ == "__main__":
    train()
