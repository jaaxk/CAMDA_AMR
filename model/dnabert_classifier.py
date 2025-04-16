import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers.models.bert.configuration_bert import BertConfig
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
from typing import Dict, List, Tuple
import os
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

class AMRDataset(Dataset):
    def __init__(self, sequences: List[str], num_hits: List[int], species: List[str], labels: List[str], tokenizer):
        logging.info('Initializing AMR Dataset...')
        # Set max length according to DNABERT2 recommendation (0.25 * sequence_length)
        self.max_length = 125  # For 500bp sequences
        logging.info(f'Using max_length of {self.max_length} for sequences')
        self.sequences = sequences
        self.num_hits = torch.tensor(num_hits, dtype=torch.float32)
        
        # Convert species to integers
        unique_species = sorted(set(species))  # Sort for consistent mapping
        species_map = {s: i for i, s in enumerate(unique_species)}
        self.species = torch.tensor([species_map[s] for s in species], dtype=torch.long)
        logging.info(f'Species mapping: {species_map}')
        self.num_species = len(unique_species)
        
        # Convert labels to integers
        label_map = {"Susceptible": 0, "Intermediate": 1, "Resistant": 2}
        self.labels = torch.tensor([label_map[label] for label in labels])
        
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        encoding = self.tokenizer(seq, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'num_hits': self.num_hits[idx],
            'species': self.species[idx],
            'labels': self.labels[idx]
        }

class DNABertClassifier(nn.Module):
    def __init__(self, model_path: str, num_classes: int = 3, num_species: int = 9):
        super().__init__()
    
        logging.info('Loading DNABERT2 configuration...')
        config = BertConfig.from_pretrained(model_path, trust_remote_code=True)
        
        logging.info('Attempting to load DNABERT2 model')
        self.bert = AutoModel.from_pretrained(model_path, trust_remote_code=True, config=config)
        logging.info('DNABERT2 model loaded successfully.')

        # Species embedding (8 dimensions is sufficient for 9 categories)
        self.species_embedding = nn.Embedding(num_species, 8)
        
        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(768 + 1 + 8, 512),  # 768 for BERT, 1 for num_hits, 8 for species embedding
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, input_ids, attention_mask, num_hits, species):
        # Get DNABERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs[0]  # [batch_size, sequence_length, 768]
        
        # Mean pool the embeddings across sequence length dimension
        pooled = torch.mean(hidden_states, dim=1)  # [batch_size, 768]
        
        # Get species embeddings
        species_emb = self.species_embedding(species)  # [batch_size, 8]
        
        # Combine all features
        combined = torch.cat([
            pooled,                    # [batch_size, 768]
            num_hits.unsqueeze(1),     # [batch_size, 1]
            species_emb                 # [batch_size, 8]
        ], dim=1)  # [batch_size, 777]
        
        # Pass through classifier
        logits = self.classifier(combined)
        return logits

def load_and_preprocess_data(data_path: str) -> Tuple[pd.DataFrame, StandardScaler]:
    """Load and preprocess the data, including sanity checks."""
    logging.info('Loading data from CSV...')
    df = pd.DataFrame(pd.read_csv(data_path))
    
    # Sanity checks
    logging.info("\n=== Data Statistics ===")
    
    # Check phenotype distribution
    phenotype_dist = df['phenotype'].value_counts()
    logging.info(f"\nPhenotype distribution:\n{phenotype_dist}")
    
    # Check sequence lengths
    seq_lengths = df['sequence'].str.len()
    logging.info(f"\nSequence length statistics:")
    logging.info(f"Mean: {seq_lengths.mean():.2f}")
    logging.info(f"Min: {seq_lengths.min()}")
    logging.info(f"Max: {seq_lengths.max()}")
    if not all(seq_lengths == 500):
        logging.warning("Not all sequences are 500bp!")
    
    # Check num_hits distribution
    logging.info(f"\nnum_hits statistics:")
    logging.info(f"Mean: {df['num_hits'].mean():.2f}")
    logging.info(f"Min: {df['num_hits'].min()}")
    logging.info(f"Max: {df['num_hits'].max()}")
    
    # Normalize num_hits
    scaler = StandardScaler()
    df['num_hits_normalized'] = scaler.fit_transform(df[['num_hits']])
    
    logging.info('Data preprocessing complete.')
    return df, scaler

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    max_grad_norm: float = 1.0,  # Add gradient clipping
) -> Dict:
    # Get process rank
    rank = torch.distributed.get_rank()
    """Train the model and return training history."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    # Create progress bar for epochs
    epoch_pbar = tqdm(range(num_epochs), desc='Epochs')
    for epoch in range(num_epochs):
        # Set epoch for samplers
        train_loader.sampler.set_epoch(epoch)
        val_loader.sampler.set_epoch(epoch)
        
        # Training
        model.train()
        train_loss = 0
        # Create progress bar for batches (only on rank 0)
        if rank == 0:
            batch_pbar = tqdm(train_loader, desc=f'Training Epoch {epoch+1}/{num_epochs}', leave=False)
            iterator = batch_pbar
        else:
            iterator = train_loader
        
        for batch in iterator:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            num_hits = batch['num_hits'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, num_hits, batch['species'].to(device))
            loss = criterion(outputs, labels)
            loss.backward()
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            
            train_loss += loss.item()
            # Update batch progress bar with current loss (only on rank 0)
            if rank == 0:
                batch_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            if rank == 0:
                val_pbar = tqdm(val_loader, desc=f'Validation Epoch {epoch+1}/{num_epochs}', leave=False)
                iterator = val_pbar
            else:
                iterator = val_loader
            
            for batch in iterator:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                num_hits = batch['num_hits'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask, num_hits, batch['species'].to(device))
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Update validation progress bar (only on rank 0)
                if rank == 0:
                    current_acc = 100. * correct / total
                    val_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{current_acc:.2f}%'})
        
        # Log metrics
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(accuracy)
        
        # Gather metrics from all processes
        train_loss_tensor = torch.tensor([train_loss], device=device)
        val_loss_tensor = torch.tensor([val_loss], device=device)
        correct_tensor = torch.tensor([correct], device=device)
        total_tensor = torch.tensor([total], device=device)
        
        torch.distributed.all_reduce(train_loss_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(val_loss_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(correct_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(total_tensor, op=torch.distributed.ReduceOp.SUM)
        
        # Calculate final metrics
        train_loss = train_loss_tensor.item() / torch.distributed.get_world_size() / len(train_loader)
        val_loss = val_loss_tensor.item() / torch.distributed.get_world_size() / len(val_loader)
        accuracy = 100. * correct_tensor.item() / total_tensor.item()
        
        # Update history (only on rank 0)
        if rank == 0:
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(accuracy)
            
            # Log progress
            logging.info(f'Epoch {epoch+1}/{num_epochs}:')
            logging.info(f'Train Loss: {train_loss:.4f}')
            logging.info(f'Val Loss: {val_loss:.4f}')
            logging.info(f'Val Accuracy: {accuracy:.2f}%')
    
    return history
