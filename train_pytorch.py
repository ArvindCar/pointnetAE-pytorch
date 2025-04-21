import argparse
import math
from datetime import datetime
import numpy as np
import torch
import torch.optim as optim
import socket
import os
import sys
import wandb
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import importlib


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

# Import your modules
from ShapeNetMeshDataLoader import get_dataloader

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]. For multiple GPUs, use comma-separated list, e.g., "0,1,2"')
parser.add_argument('--model', default='model_pytorch', help='Model name [default: model_pytorch]')
parser.add_argument('--log_dir', default='log_pytorch', help='Log dir [default: log_pytorch]')
parser.add_argument('--resume', default=None, help='Path to checkpoint file for resuming training')
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 2048]')
parser.add_argument('--max_epoch', type=int, default=201, help='Epoch to run [default: 201]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--shapenet_dir', required=True, help='Path to ShapeNetCore.v2 directory')
parser.add_argument('--cache_dir', default=None, help='Directory to cache sampled point clouds')
parser.add_argument('--rotation', default='aligned', choices=['aligned', 'z', 'so3'], 
                   help='Type of rotation augmentation: aligned (none), z (rotation around z-axis), so3 (random 3D rotation)')
parser.add_argument('--normalize', action='store_true', help='Normalize point clouds to unit sphere')
parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
FLAGS = parser.parse_args()

EPOCH_CNT = 0

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

LOG_DIR = FLAGS.log_dir + '_' + datetime.now().strftime('%Y%m%d_%H%M%S')
if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
os.system(f'cp models/{FLAGS.model}.py {LOG_DIR}')
os.system(f'cp {sys.argv[0]} {LOG_DIR}')  # Copy training script
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

HOSTNAME = socket.gethostname()

# import the model dynamically based on the command line argument
model_module = importlib.import_module(f'models.{FLAGS.model}')
PointNetAE = model_module.PointNetAE
get_loss = model_module.get_loss

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def init_wandb(model=None):
    if not is_main_process():
        return
    wandb.init(
        project="pointnet-AE",
        config=vars(FLAGS),
        name=f"{os.path.basename(LOG_DIR)}",
        dir=LOG_DIR
    )
    # Log model architecture as a graph
    # wandb.watch(model)

def log_string(out_str):
    if not is_main_process():
        return
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

# Learning rate adjustment function
def adjust_learning_rate(optimizer, batch_idx):
    lr = BASE_LEARNING_RATE * (DECAY_RATE ** (batch_idx / DECAY_STEP))
    lr = max(lr, 0.00001)  # CLIP THE LEARNING RATE
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def train():
    
    # Set device
    if torch.cuda.is_available():
        local_rank = int(os.environ['LOCAL_RANK'])
        print("Available GPUs:", torch.cuda.device_count())
        print("local_rank:", local_rank)
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
        log_string("Using CPU")
    log_string(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    
    # Create model and optimizer
    model = PointNetAE(num_point=NUM_POINT).to(device)

    
    if OPTIMIZER == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=BASE_LEARNING_RATE)
    else:
        optimizer = optim.SGD(model.parameters(), lr=BASE_LEARNING_RATE, momentum=MOMENTUM)
    
    # Create DataLoaders using ShapeNetMeshDataLoader
    log_string(f"Creating dataloaders for ShapeNet dataset at {FLAGS.shapenet_dir}")

    train_loader, train_sampler = get_dataloader(
        root_dir=FLAGS.shapenet_dir,
        batch_size=BATCH_SIZE,
        n_points=NUM_POINT,
        split='train',
        normalize=FLAGS.normalize,
        rotation=FLAGS.rotation,
        cache_dir=FLAGS.cache_dir,
        shuffle=True,
        num_workers=FLAGS.num_workers,
        seed=FLAGS.seed,
        distributed=True
    )
    
    val_loader,val_sampler = get_dataloader(
        root_dir=FLAGS.shapenet_dir,
        batch_size=BATCH_SIZE,
        n_points=NUM_POINT,
        split='val',
        normalize=FLAGS.normalize,
        rotation='aligned',  # No rotation augmentation for validation
        cache_dir=FLAGS.cache_dir,
        shuffle=False,
        num_workers=FLAGS.num_workers,
        seed=FLAGS.seed,
        distributed=True
    )
    
    test_loader, test_sampler = get_dataloader(
        root_dir=FLAGS.shapenet_dir,
        batch_size=BATCH_SIZE,
        n_points=NUM_POINT,
        split='test',
        normalize=FLAGS.normalize,
        rotation='aligned',  # No rotation augmentation for testing
        cache_dir=FLAGS.cache_dir,
        shuffle=False,
        num_workers=FLAGS.num_workers,
        seed=FLAGS.seed,
        distributed=True
    )
    
    log_string(f"Train dataset size: {len(train_loader.dataset)}")
    log_string(f"Validation dataset size: {len(val_loader.dataset)}")
    log_string(f"Test dataset size: {len(test_loader.dataset)}")
    # Variables for resuming training
    global_step = 0
    best_loss = 1e20
    start_epoch = 0
    
    # Resume from checkpoint if specified
    if FLAGS.resume:
        if os.path.isfile(FLAGS.resume):
            log_string(f"Loading checkpoint from {FLAGS.resume}")
            checkpoint = torch.load(FLAGS.resume, map_location=device)
            
            # Load model state
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Restore training state
            start_epoch = checkpoint.get('epoch', 0) + 1  # Start from next epoch
            best_loss = checkpoint.get('best_loss', 1e20)
            global_step = checkpoint.get('global_step', 0)
            
            log_string(f"Resuming training from epoch {start_epoch}")
            log_string(f"Loaded checkpoint with best validation loss: {best_loss}")
        else:
            log_string(f"No checkpoint found at {FLAGS.resume}")

    model = DDP(model, device_ids=[local_rank])
    # Initialize wandb
    init_wandb(model)
    
    # Training loop
    global_step = 0
    best_loss = 1e20
    
    for epoch in range(MAX_EPOCH):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        log_string('**** EPOCH %03d ****' % (epoch))
        sys.stdout.flush()
        
        # Train one epoch
        train_loss, train_pcloss = train_one_epoch(
            model, train_loader, optimizer, device, 
            global_step
        )
        global_step += len(train_loader)
        
        log_string(f'Train Epoch {epoch} mean loss: {train_loss:.6f}')
        log_string(f'Train Epoch {epoch} mean pc loss: {train_pcloss:.6f}')
        
        # Evaluate on validation set
        val_loss, val_pcloss = eval_one_epoch(
            model, val_loader, device, 
            "Validation"
        )
        
        log_string(f'Validation Epoch {epoch} mean loss: {val_loss:.6f}')
        log_string(f'Validation Epoch {epoch} mean pc loss: {val_pcloss:.6f}')
        
        # Evaluate on test set (less frequently to save time)
        if epoch % 5 == 0:
            test_loss, test_pcloss = eval_one_epoch(
                model, test_loader, device, 
                "Test"
            )
            
            log_string(f'Test Epoch {epoch} mean loss: {test_loss:.6f}')
            log_string(f'Test Epoch {epoch} mean pc loss: {test_pcloss:.6f}')
            
            # Visualize some reconstructions
            if epoch % 10 == 0:
                with torch.no_grad():
                    for point_clouds in test_loader:
                        if point_clouds is None:
                            continue
                        point_clouds = point_clouds.to(device)
                        reconstructed_pc, _ = model(point_clouds)
                        
                        # Visualize first model in batch (wandb has 3D point cloud support)
                        if point_clouds.size(0) > 0:
                            # Convert to numpy for wandb
                            original = point_clouds[0].cpu().numpy()
                            recon = reconstructed_pc[0].cpu().numpy()
                            
                            # Log point clouds to wandb
                            if is_main_process():  
                                wandb.log({
                                    f"pointcloud/epoch_{epoch}_original": wandb.Object3D(original),
                                    f"pointcloud/epoch_{epoch}_reconstructed": wandb.Object3D(recon),
                                    "epoch": epoch
                                })
                        break  # Only visualize one batch
        
        # Save best model based on validation loss
        if val_loss < best_loss:
            best_loss = val_loss
            if is_main_process():
                save_path = os.path.join(LOG_DIR, f"best_model_epoch_{epoch:03d}.pth")
                # Handle the case where model is wrapped with DataParallel
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save(model_to_save.state_dict(), save_path)
                log_string(f"Best model saved to {save_path}")
            
                # Log model to wandb
                wandb.save(save_path)
                # Log that this is the best model so far
                wandb.run.summary["best_val_loss"] = best_loss
                wandb.run.summary["best_epoch"] = epoch
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            save_path = os.path.join(LOG_DIR, f"checkpoint_epoch_{epoch:03d}.pth")
            # Handle the case where model is wrapped with DataParallel or DDP
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
                'global_step': global_step,
                'seed': FLAGS.seed,
                'lr': BASE_LEARNING_RATE
            }, save_path)
            log_string(f"Checkpoint saved to {save_path}")
            
        global EPOCH_CNT
        EPOCH_CNT += 1
    if is_main_process():
        wandb.finish()

def train_one_epoch(model, dataloader, optimizer, device, global_step):
    """Train for one epoch."""
    model.train()
    train_loss_sum = 0
    train_pcloss_sum = 0
    batch_count = 0
    
    for i, point_clouds in enumerate(dataloader):
        if point_clouds is None:  # Skip empty batches
            continue
            
        point_clouds = point_clouds.to(device)
        
        # Adjust learning rate
        lr = adjust_learning_rate(optimizer, global_step + i)
        
        # Forward pass
        optimizer.zero_grad()
        reconstructed_pc, end_points = model(point_clouds)
        loss, end_points = get_loss(reconstructed_pc, point_clouds, end_points)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track statistics
        train_loss_sum += loss.item()
        train_pcloss_sum += end_points['pcloss'].item()
        batch_count += 1
        
        # Log metrics periodically
        if (i+1) % 10 == 0:
            if is_main_process():
                # Log to wandb
                wandb.log({
                    'batch/loss': loss.item(),
                    'batch/pcloss': end_points['pcloss'].item(),
                    'batch/learning_rate': lr,
                    'batch_idx': global_step + i
                })
            
            log_string(' -- Batch %03d / %03d --' % (i+1, len(dataloader)))
            log_string('batch loss: %f' % loss.item())
            log_string('batch pc loss: %f' % end_points['pcloss'].item())
    
    # Return average metrics for the epoch
    avg_loss = train_loss_sum / max(batch_count, 1)
    avg_pcloss = train_pcloss_sum / max(batch_count, 1)
    
    # Log average metrics for the epoch
    if is_main_process():
        # Log to wandb
        wandb.log({
            'train/epoch_loss': avg_loss,
            'train/epoch_pcloss': avg_pcloss,
            'epoch': EPOCH_CNT
        })
    
    return avg_loss, avg_pcloss

def eval_one_epoch(model, dataloader, device, phase="Validation"):
    """Evaluate for one epoch."""
    model.eval()
    eval_loss_sum = 0
    eval_pcloss_sum = 0
    batch_count = 0
    
    with torch.no_grad():
        for point_clouds in dataloader:
            if point_clouds is None:  # Skip empty batches
                continue
                
            point_clouds = point_clouds.to(device)
            
            # Forward pass
            reconstructed_pc, end_points = model(point_clouds)
            loss, end_points = get_loss(reconstructed_pc, point_clouds, end_points)
            
            # Track statistics
            eval_loss_sum += loss.item()
            eval_pcloss_sum += end_points['pcloss'].item()
            batch_count += 1
    
    # Return average metrics for the epoch
    avg_loss = eval_loss_sum / max(batch_count, 1)
    avg_pcloss = eval_pcloss_sum / max(batch_count, 1)
    
    # Log average metrics for the epoch
    if is_main_process():
        # Log to wandb
        wandb.log({
            f'{phase.lower()}/epoch_loss': avg_loss,
            f'{phase.lower()}/epoch_pcloss': avg_pcloss,
            'epoch': EPOCH_CNT
        })
    
    return avg_loss, avg_pcloss

if __name__ == "__main__":
    log_string('pid: %s' % (str(os.getpid())))
    log_string('Starting training...')
    train()
    LOG_FOUT.close()