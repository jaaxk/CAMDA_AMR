import torch
from transformers import AutoTokenizer, logging as transformers_logging
from torch.utils.data import DataLoader
import argparse
import logging
from dnabert_classifier import DNABertClassifier, AMRDataset, load_and_preprocess_data, train_model
import os
import gc
import psutil
import sys

def print_memory_stats(stage):
    process = psutil.Process()
    logging.info(f'=== Memory Stats at {stage} ===')    
    logging.info(f'CPU Memory used: {process.memory_info().rss / 1024 / 1024:.2f} MB')
    if torch.cuda.is_available():
        logging.info(f'GPU Memory allocated: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB')
        logging.info(f'GPU Memory cached: {torch.cuda.memory_reserved() / 1024 / 1024:.2f} MB')

# Reduce transformers logging
transformers_logging.set_verbosity_error()

def main():
    parser = argparse.ArgumentParser(description='Train DNABERT AMR Classifier')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the training CSV file')
    parser.add_argument('--model_path', type=str, default="zhihan1996/DNABERT-2-117M",
                      help='Path to DNABERT model (local path or Hugging Face model ID)')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the model')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device and memory settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        # Set memory settings for A100
        torch.cuda.empty_cache()
        gc.collect()
        torch.backends.cudnn.benchmark = True
        # Enable TF32 for better performance on A100
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    logging.info(f'Using device: {device}')
    
    logging.info('=== Starting data loading and preprocessing ===')    
    # Load and preprocess data
    df, scaler = load_and_preprocess_data(args.data_path)
    print_memory_stats('after data loading')
    
    # Split data
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(
        df, test_size=args.val_split, random_state=args.seed,
        stratify=df['phenotype']  # Maintain class distribution
    )
    
    logging.info('=== Initializing tokenizer ===')    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    print_memory_stats('after tokenizer init')
    
    logging.info('=== Creating datasets ===')    
    try:
        # Create datasets
        train_dataset = AMRDataset(
            sequences=train_df['sequence'].tolist(),
            num_hits=train_df['num_hits_normalized'].tolist(),
            labels=train_df['phenotype'].tolist(),
            tokenizer=tokenizer
        )
        print_memory_stats('after train dataset creation')
    except Exception as e:
        logging.error(f'Error during dataset creation: {str(e)}')
        sys.exit(1)
    
    val_dataset = AMRDataset(
        sequences=val_df['sequence'].tolist(),
        num_hits=val_df['num_hits_normalized'].tolist(),
        labels=val_df['phenotype'].tolist(),
        tokenizer=tokenizer
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    logging.info('=== Initializing model ===')    
    try:
        # Initialize model
        model = DNABertClassifier(args.model_path)
        print_memory_stats('after model creation')
        
        model = model.to(device)
        print_memory_stats('after moving model to device')
    except Exception as e:
        logging.error(f'Error during model initialization: {str(e)}')
        sys.exit(1)
    
    # Train model
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate
    )
    
    # Save model and scaler
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'history': history,
        'args': args,
    }, os.path.join(args.output_dir, 'model_checkpoint.pt'))
    
    logging.info(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
