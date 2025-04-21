import argparse
import numpy as np
import torch
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from utils import show3d_balls
from models.model_pytorch import PointNetAE
from ShapeNetMeshDataLoader import get_dataloader

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 2048]')
parser.add_argument('--category', default=None, help='Which single class to train on [default: None]')
parser.add_argument('--model_path', default='log_pytorch/best_model.pth', help='Model checkpoint file path [default: log_pytorch/best_model.pth]')
parser.add_argument('--num_group', type=int, default=1, help='Number of groups of generated points [default: 1]')
parser.add_argument('--shapenet_dir', required=True, help='Path to ShapeNetCore.v2 directory')
parser.add_argument('--cache_dir', default=None, help='Directory to cache sampled point clouds')
parser.add_argument('--normalize', action='store_true', help='Normalize point clouds to unit sphere')
parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to visualize [default: 10]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference [default: 1]')
parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
FLAGS = parser.parse_args()

def get_model(model_path, device):
    """Load the pre-trained model."""
    model = PointNetAE(num_point=FLAGS.num_point).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set to evaluation mode
    return model

def inference(model, pc, device):
    """Run inference on the given point cloud."""
    with torch.no_grad():
        pc_tensor = torch.tensor(pc, dtype=torch.float32).to(device)
        reconstructed_pc, _ = model(pc_tensor)
        return reconstructed_pc.cpu().numpy()

if __name__ == '__main__':
    # Set device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{FLAGS.gpu}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    print(f"Using device: {device}")
    
    # Initialize colors for visualization
    num_group = FLAGS.num_group
    color_list = []
    for i in range(num_group):
        color_list.append(np.random.random((3,)))
    
    # Load the pre-trained model
    model = get_model(FLAGS.model_path, device)
    
    # Create test dataset
    test_loader, _ = get_dataloader(
        root_dir=FLAGS.shapenet_dir,
        batch_size=FLAGS.batch_size,
        n_points=FLAGS.num_point,
        split='test',
        normalize=FLAGS.normalize,
        rotation='aligned',  # No rotation for testing
        cache_dir=FLAGS.cache_dir,
        shuffle=True,  # Shuffle to get random samples
        num_workers=FLAGS.num_workers,
        seed=42,
        distributed=False  # No distributed inference
    )
    
    print(f"Test dataset size: {len(test_loader.dataset)}")
    
    # Visualize a subset of examples
    sample_count = 0
    
    for batch in test_loader:
        if batch is None:
            continue
            
        # Move data to device
        point_clouds = batch.to(device)
        
        # Run inference
        reconstructed_pc, _ = model(point_clouds)
        
        # Process each example in the batch
        for i in range(point_clouds.shape[0]):
            # Get original and reconstructed point clouds
            original_pc = point_clouds[i].cpu().numpy()
            pred_pc = reconstructed_pc[i].cpu().numpy()
            
            # Visualize original point cloud
            print(f"Visualizing sample {sample_count+1}/{FLAGS.num_samples}")
            show3d_balls.showpoints(original_pc, ballradius=8, title="Original")
            
            # Visualize reconstructed point cloud
            show3d_balls.showpoints(pred_pc, ballradius=8, title="Reconstructed")
            
            # Visualize with colors if more than one group
            if num_group > 1:
                c_gt = np.zeros_like(pred_pc)
                for j in range(num_group):
                    start_idx = j * FLAGS.num_point // num_group
                    end_idx = (j + 1) * FLAGS.num_point // num_group
                    c_gt[start_idx:end_idx, :] = color_list[j]
                show3d_balls.showpoints(pred_pc, c_gt=c_gt, ballradius=8, title="Reconstructed (Colored)")
            
            sample_count += 1
            if sample_count >= FLAGS.num_samples:
                sys.exit(0)  # Exit after showing the requested number of samples