import torch
from transformers import AutoTokenizer, logging as transformers_logging
from transformers.models.bert.configuration_bert import BertConfig
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp
import argparse
import logging
from dnabert_classifier import DNABertClassifier, AMRDataset, load_and_preprocess_data, train_model
import os
import gc
import psutil
import sys
from sklearn.model_selection import train_test_split

def print_memory_stats(stage):
    process = psutil.Process()
    logging.info(f'=== Memory Stats at {stage} ===')    
    logging.info(f'CPU Memory used: {process.memory_info().rss / 1024 / 1024:.2f} MB')
    if torch.cuda.is_available():
        logging.info(f'GPU Memory allocated: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB')
        logging.info(f'GPU Memory cached: {torch.cuda.memory_reserved() / 1024 / 1024:.2f} MB')

# Reduce transformers logging
transformers_logging.set_verbosity_error()

def setup_ddp():
    """Initialize DDP process group using environment variables set by torchrun"""
    init_process_group(backend='nccl')

def cleanup_ddp():
    """Clean up DDP process group"""
    destroy_process_group()

def main(local_rank: int = 0):
    """Main training function.
    Args:
        local_rank (int): Local rank passed by torchrun
    """
    # Parse arguments
    args = parse_args()
    # Initialize DDP
    setup_ddp()
    
    # Get rank and world size from environment
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    local_rank = int(os.environ["LOCAL_RANK"])
    
    # Ensure model/data are on the right device
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    # Set up logging only on rank 0
    if rank == 0:
        logging.info(f'Using device: {device}')
        logging.info(f'CUDA available: {torch.cuda.is_available()}')
        if torch.cuda.is_available():
            logging.info(f'CUDA device count: {torch.cuda.device_count()}')
            logging.info(f'CUDA current device: {torch.cuda.current_device()}')
            logging.info(f'CUDA device name: {torch.cuda.get_device_name(device)}')
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output directory
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Set memory settings
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    if rank == 0:
        logging.info('=== Starting data loading and preprocessing ===')
    
    # Load and preprocess data
    df, scaler = load_and_preprocess_data(args.data_path)
    if rank == 0:
        print_memory_stats('after data loading')
    
    # Split data
    train_df, val_df = train_test_split(
        df, test_size=args.val_split, random_state=args.seed,
        stratify=df['phenotype']  # Maintain class distribution
    )
    
    if rank == 0:
        logging.info('=== Initializing tokenizer ===')
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True
    )
    if rank == 0:
        print_memory_stats('after tokenizer init')
    
    if rank == 0:
        logging.info('=== Creating datasets ===')
    
    # Create datasets
    train_dataset = AMRDataset(
        sequences=train_df['sequence'].tolist(),
        num_hits=train_df['num_hits_normalized'].tolist(),
        species=train_df['species'].tolist(),
        labels=train_df['phenotype'].tolist(),
        tokenizer=tokenizer
    )
    
    val_dataset = AMRDataset(
        sequences=val_df['sequence'].tolist(),
        num_hits=val_df['num_hits_normalized'].tolist(),
        species=val_df['species'].tolist(),
        labels=val_df['phenotype'].tolist(),
        tokenizer=tokenizer
    )
    
    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    if rank == 0:
        logging.info('=== Initializing model ===')
    
    try:
        # Initialize model
        num_species = len(train_df['species'].unique())
        model = DNABertClassifier(
            model_path=args.model_path,
            num_classes=3,
            num_species=num_species
        )
        
        # Move model to device
        model = model.to(device)
        
        # Wrap model with DDP
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        
        if rank == 0:
            print_memory_stats('after model creation')
    
    except Exception as e:
        logging.error(f'Error during model initialization: {str(e)}')
        cleanup_ddp()
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
    
    # Save model and scaler (only on rank 0)
    if rank == 0:
        torch.save({
            'model_state_dict': model.module.state_dict(),  # Save the inner model's state
            'scaler': scaler,
            'history': history,
            'args': args,
        }, os.path.join(args.output_dir, 'model_checkpoint.pt'))
        
        logging.info(f"Model saved to {args.output_dir}")
    
    # Cleanup
    cleanup_ddp()

def parse_args():
    parser = argparse.ArgumentParser(description='Train DNABERT AMR Classifier')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the training CSV file')
    parser.add_argument('--model_path', type=str, default="zhihan1996/DNABERT-2-117M",
                      help='Path to DNABERT model (local path or Hugging Face model ID)')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size per GPU for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers per GPU')
    return parser.parse_args()

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
    
    logging.info('=== Initializing model ===')    
    try:
        # Initialize model
        num_species = len(train_df['species'].unique())
        model = DNABertClassifier(
            model_path=args.model_path,
            num_classes=3,
            num_species=num_species
        )
        print_memory_stats('after model creation')
        
        model = model.to(device)
        print_memory_stats('after moving model to device')
    except Exception as e:
        logging.error(f'Error during model initialization: {str(e)}')
        sys.exit(1)
    
    logging.info('=== Creating datasets ===')    
    try:
        # Create datasets
        train_dataset = AMRDataset(
            sequences=train_df['sequence'].tolist(),
            num_hits=train_df['num_hits_normalized'].tolist(),
            species=train_df['species'].tolist(),
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
        species=val_df['species'].tolist(),
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
    # Set up logging format
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Call main with local rank from environment
    main()
