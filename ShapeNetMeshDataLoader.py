import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

# Python Version Flag: If > 3.6, you don't need to do this
import contextlib
if not hasattr(contextlib, 'nullcontext'):
    class nullcontext:
        def __init__(self, enter_result=None):
            self.enter_result = enter_result
        def __enter__(self):
            return self.enter_result
        def __exit__(self, *excinfo):
            pass
    contextlib.nullcontext = nullcontext

from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.transforms import random_rotations
from pytorch3d.datasets import ShapeNetCore
import math
import random
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ShapeNetPointCloudDataset(Dataset):
    """
    Dataset for loading ShapeNetCore.v2 models as point clouds.
    
    Samples points from 3D meshes and applies optional normalization and rotations.
    """
    
    def __init__(self, 
                 root_dir, 
                 n_points=2048, 
                 split='train', 
                 split_ratio=[0.8, 0.1, 0.1],
                 normalize=True, 
                 rotation='aligned', 
                 transforms=None,
                 cache_dir=None,
                 seed=42):
        """
        Initialize the ShapeNet point cloud dataset.
        
        Args:
            root_dir (str): Path to ShapeNetCore.v2 directory
            n_points (int): Number of points to sample from each mesh
            split (str): One of 'train', 'val', 'test'
            split_ratio (list): Ratio for train/val/test split as [train, val, test]
            normalize (bool): Whether to normalize point clouds to unit sphere
            rotation (str): Rotation type: 'aligned' (no rotation), 'z' (random z rotation), 'so3' (random 3D rotation)
            transforms (callable): Optional transform to apply to the point clouds
            cache_dir (str): Optional directory to cache sampled point clouds
            seed (int): Random seed for reproducibility
        """
        logger.info(f"Initializing ShapeNet dataset from {root_dir}")
        
        self.root_dir = root_dir
        self.n_points = n_points
        self.normalize = normalize
        self.rotation = rotation
        self.transforms = transforms
        self.cache_dir = cache_dir
        
        # Create cache directory if specified
        if self.cache_dir and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            logger.info(f"Created cache directory at {self.cache_dir}")
            
        # Validate inputs
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"Split must be one of 'train', 'val', 'test', got {split}")
            
        if rotation not in ['aligned', 'z', 'so3']:
            raise ValueError(f"Rotation must be one of 'aligned', 'z', 'so3', got {rotation}")
            
        if sum(split_ratio) != 1.0:
            raise ValueError(f"Split ratio must sum to 1.0, got {split_ratio}")
            
        # Set random seed for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        self.dataset = ShapeNetCore(root_dir, version=2, load_textures=False)

        # Apply train/val/test split
        num_total = len(self.dataset)
        indices = np.random.permutation(num_total)
        train_end = int(split_ratio[0] * num_total)
        val_end = train_end + int(split_ratio[1] * num_total)

        if split == 'train':
            self.indices = indices[:train_end]
        elif split == 'val':
            self.indices = indices[train_end:val_end]
        else:  # test
            self.indices = indices[val_end:]
            
        # # Find all model paths
        # logger.info("Scanning for model files...")
        # self.model_paths = self._find_model_paths()
        # logger.info(f"Found {len(self.model_paths)} models")
        
        # # Split dataset
        # self._split_dataset(split, split_ratio)
        # logger.info(f"Using {len(self.model_paths)} models for {split} split")
        
        # # Initialize statistics
        # self.stats = {
        #     'loaded': 0,
        #     'errors': 0,
        #     'cached': 0
        # }
        
    # def _find_model_paths(self):
    #     """Find all valid model paths in the ShapeNet directory."""
    #     model_paths = []
    #     synset_dirs = [d for d in os.listdir(self.root_dir) 
    #                   if os.path.isdir(os.path.join(self.root_dir, d))]
        
    #     for synset_id in synset_dirs:
    #         synset_path = os.path.join(self.root_dir, synset_id)
    #         model_ids = [d for d in os.listdir(synset_path) 
    #                     if os.path.isdir(os.path.join(synset_path, d))]
            
    #         for model_id in model_ids:
    #             model_path = os.path.join(synset_path, model_id)
                
    #             # Check both possible model file locations
    #             model_file = os.path.join(model_path, 'models', 'model_normalized.obj')
    #             if not os.path.exists(model_file):
    #                 model_file = os.path.join(model_path, 'model_normalized.obj')
                
    #             if os.path.exists(model_file):
    #                 model_paths.append({
    #                     'synset_id': synset_id,
    #                     'model_id': model_id,
    #                     'obj_path': model_file
    #                 })
        
    #     return model_paths
    
    # def _split_dataset(self, split, split_ratio):
    #     """Split dataset into train/val/test sets."""
    #     # Shuffle model paths
    #     indices = np.random.permutation(len(self.model_paths))
        
    #     # Calculate split indices
    #     train_end = int(split_ratio[0] * len(indices))
    #     val_end = train_end + int(split_ratio[1] * len(indices))
        
    #     # Select appropriate indices
    #     if split == 'train':
    #         selected_indices = indices[:train_end]
    #     elif split == 'val':
    #         selected_indices = indices[train_end:val_end]
    #     else:  # test
    #         selected_indices = indices[val_end:]
            
    #     # Update model paths to only include the selected split
    #     self.model_paths = [self.model_paths[i] for i in selected_indices]
    
    def _get_cache_path(self, model_info):
        """Get the cache path for a model."""
        if not self.cache_dir:
            return None
            
        cache_filename = f"{model_info['synset_id']}_{model_info['model_id']}.pt"
        return os.path.join(self.cache_dir, cache_filename)
    
    def _normalize_pointcloud(self, points):
        """Normalize point cloud to unit sphere."""
        centroid = torch.mean(points, dim=0)
        points = points - centroid
        max_dist = torch.max(torch.norm(points, dim=1))
        points = points / max_dist if max_dist > 0 else points
        return points
    
    def _apply_rotation(self, points):
        """Apply rotation based on the specified type."""
        if self.rotation == 'z':
            # Random rotation around z-axis
            angle = torch.rand(1) * 2 * math.pi
            cos_a, sin_a = torch.cos(angle), torch.sin(angle)
            rot_matrix = torch.tensor([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ], dtype=points.dtype).squeeze(0)
            
            points = torch.matmul(points, rot_matrix.T)
            
        elif self.rotation == 'so3':
            # Random 3D rotation
            rot_matrix = random_rotations(1)[0]
            points = torch.matmul(points, rot_matrix.T)
            
        return points
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        try:
            model = self.dataset[self.indices[idx]]
            verts, faces = model["verts"], model["faces"]
            mesh = Meshes(verts=[verts], faces=[faces])
            pointcloud = sample_points_from_meshes(mesh, num_samples=self.n_points).squeeze(0)
            
            if self.normalize:
                pointcloud = self._normalize_pointcloud(pointcloud)
            pointcloud = self._apply_rotation(pointcloud)
            
            if self.transforms:
                pointcloud = self.transforms(pointcloud)
            
            return pointcloud
        except Exception as e:
            logger.error(f"Error loading model at index {idx}: {e}")
            fallback = torch.randn(self.n_points, 3)
            fallback = fallback / torch.norm(fallback, dim=1, keepdim=True)
            return fallback

        # model_info = self.model_paths[idx]
        # obj_path = model_info['obj_path']
        # cache_path = self._get_cache_path(model_info)
        
        # # Try to load from cache first
        # if cache_path and os.path.exists(cache_path):
        #     try:
        #         pointcloud = torch.load(cache_path)
        #         self.stats['cached'] += 1
                
        #         # Apply rotation (we don't cache rotations to allow variety)
        #         pointcloud = self._apply_rotation(pointcloud)
                
        #         # Apply transforms if any
        #         if self.transforms:
        #             pointcloud = self.transforms(pointcloud)
                    
        #         return pointcloud
        #     except Exception as e:
        #         logger.warning(f"Failed to load from cache, regenerating: {e}")
        
        # # Try to load and process the model
        # try:
        #     # Load the mesh
        #     with open(obj_path, 'r') as f:
        #         verts, faces, _ = load_obj(
        #             f,
        #             load_textures=False,
        #             create_texture_atlas=False,
        #             texture_atlas_size=1
        #         )
            
        #     # Create mesh and sample points
        #     mesh = Meshes(verts=[verts], faces=[faces.verts_idx])
        #     pointcloud = sample_points_from_meshes(mesh, num_samples=self.n_points).squeeze(0)
            
        #     # Normalize if requested
        #     if self.normalize:
        #         pointcloud = self._normalize_pointcloud(pointcloud)
                
        #     # Apply rotation
        #     pointcloud = self._apply_rotation(pointcloud)
            
        #     # Save to cache if enabled
        #     if cache_path:
        #         # Save without rotation to allow for different rotations
        #         # when loading from cache
        #         torch.save(pointcloud, cache_path)
            
        #     # Apply transforms if any
        #     if self.transforms:
        #         pointcloud = self.transforms(pointcloud)
                
        #     self.stats['loaded'] += 1
        #     return pointcloud
            
        # except Exception as e:
        #     logger.error(f"Error loading {obj_path}: {str(e)}")
        #     self.stats['errors'] += 1
            
        #     # Return a fallback point cloud (random points on a sphere)
        #     pointcloud = torch.randn(self.n_points, 3)
        #     pointcloud = pointcloud / torch.norm(pointcloud, dim=1, keepdim=True)
        #     return pointcloud
            
    def get_stats(self):
        """Get dataset loading statistics."""
        return self.stats


def get_dataloader(
    root_dir,
    batch_size=32,
    n_points=2048,
    split='train',
    split_ratio=[0.8, 0.1, 0.1],
    normalize=True,
    rotation='aligned',
    transforms=None,
    cache_dir=None,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    seed=42,
    distributed=False 
):
    """
    Create a DataLoader for ShapeNet point clouds.
    
    Args:
        root_dir (str): Path to ShapeNetCore.v2 directory
        batch_size (int): Batch size
        n_points (int): Number of points to sample per model
        split (str): One of 'train', 'val', 'test'
        split_ratio (list): Ratio for train/val/test split as [train, val, test]
        normalize (bool): Whether to normalize point clouds
        rotation (str): Rotation type: 'aligned', 'z', 'so3'
        transforms (callable): Optional transform to apply
        cache_dir (str): Optional directory to cache point clouds
        shuffle (bool): Whether to shuffle the dataset
        num_workers (int): Number of worker processes for data loading
        pin_memory (bool): Whether to pin memory for faster GPU transfer
        seed (int): Random seed for reproducibility
        
    Returns:
        torch.utils.data.DataLoader: DataLoader for the specified dataset
    """
    dataset = ShapeNetPointCloudDataset(
        root_dir=root_dir,
        n_points=n_points,
        split=split,
        split_ratio=split_ratio,
        normalize=normalize,
        rotation=rotation,
        transforms=transforms,
        cache_dir=cache_dir,
        seed=seed
    )

    def collate_fn(batch):
        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            return None
        return torch.utils.data.dataloader.default_collate(batch)

    # Setup sampler
    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False  # Sampler handles it

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=False
    )

    return dataloader, sampler if distributed else dataloader

# Example usage:
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ShapeNet DataLoader Test")
    parser.add_argument("--root", required=True, help="Path to ShapeNetCore.v2 directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--n_points", type=int, default=2048, help="Points per model")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--cache_dir", default=None, help="Cache directory")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    
    args = parser.parse_args()
    
    # Create dataloader
    dataloader = get_dataloader(
        root_dir=args.root,
        batch_size=args.batch_size,
        n_points=args.n_points,
        split=args.split,
        cache_dir=args.cache_dir,
        num_workers=args.num_workers
    )
    
    # Test dataloader
    logger.info(f"Created dataloader with {len(dataloader.dataset)} samples")
    
    for i, batch in enumerate(dataloader):
        if i == 0:
            logger.info(f"First batch shape: {batch.shape}")
        
        if (i+1) % 10 == 0:
            logger.info(f"Loaded {i+1} batches")
            
        if i >= 49:  # Load 50 batches as a test
            break
    
    # Print statistics
    stats = dataloader.dataset.get_stats()
    logger.info(f"Dataset loading stats: {stats}")