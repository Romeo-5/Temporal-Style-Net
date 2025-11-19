"""
Training script with multi-GPU support
Implements distributed training using PyTorch DistributedDataParallel
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.style_transfer import StyleTransferNet, Encoder
from src.models.temporal_consistency import TemporalConsistencyModule


class ContentStyleDataset(Dataset):
    """Dataset for style transfer training"""
    
    def __init__(self, content_dir, style_dir, image_size=256):
        self.content_paths = list(Path(content_dir).glob('*.jpg')) + \
                            list(Path(content_dir).glob('*.png'))
        self.style_paths = list(Path(style_dir).glob('*.jpg')) + \
                          list(Path(style_dir).glob('*.png'))
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomCrop(image_size),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.content_paths)
    
    def __getitem__(self, idx):
        # Load content
        content = Image.open(self.content_paths[idx]).convert('RGB')
        content = self.transform(content)
        
        # Random style
        style_idx = np.random.randint(len(self.style_paths))
        style = Image.open(self.style_paths[style_idx]).convert('RGB')
        style = self.transform(style)
        
        return content, style


class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG features"""
    
    def __init__(self, encoder):
        super(PerceptualLoss, self).__init__()
        self.encoder = encoder
        self.criterion = nn.MSELoss()
    
    def forward(self, output, target):
        """Compute perceptual loss at multiple layers"""
        output_features = self.encoder(output)
        target_features = self.encoder(target)
        
        loss = 0.0
        for out_feat, target_feat in zip(output_features, target_features):
            loss += self.criterion(out_feat, target_feat)
        
        return loss


class StyleLoss(nn.Module):
    """Style loss using Gram matrices"""
    
    def __init__(self, encoder):
        super(StyleLoss, self).__init__()
        self.encoder = encoder
        self.criterion = nn.MSELoss()
    
    def gram_matrix(self, features):
        """Compute Gram matrix"""
        N, C, H, W = features.size()
        features = features.view(N, C, H * W)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (C * H * W)
    
    def forward(self, output, style):
        """Compute style loss"""
        output_features = self.encoder(output)
        style_features = self.encoder(style)
        
        loss = 0.0
        for out_feat, style_feat in zip(output_features, style_features):
            out_gram = self.gram_matrix(out_feat)
            style_gram = self.gram_matrix(style_feat)
            loss += self.criterion(out_gram, style_gram)
        
        return loss


class Trainer:
    """Main training class with multi-GPU support"""
    
    def __init__(self, config, rank=0, world_size=1):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.is_main = (rank == 0)
        
        # Set device
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{rank}')
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device('cpu')
        
        # Initialize models
        self.setup_models()
        
        # Setup data
        self.setup_data()
        
        # Setup training
        self.setup_training()
        
        # Tensorboard
        if self.is_main:
            self.writer = SummaryWriter(config['log_dir'])
    
    def setup_models(self):
        """Initialize models"""
        # Style transfer network
        self.encoder = Encoder(pretrained=True).to(self.device)
        self.style_net = StyleTransferNet(encoder=self.encoder).to(self.device)
        
        # Temporal consistency (optional)
        if self.config.get('use_temporal', False):
            self.temporal_module = TemporalConsistencyModule().to(self.device)
        else:
            self.temporal_module = None
        
        # Wrap with DDP if using multiple GPUs
        if self.world_size > 1:
            self.style_net = DDP(
                self.style_net,
                device_ids=[self.rank],
                find_unused_parameters=True
            )
            if self.temporal_module is not None:
                self.temporal_module = DDP(
                    self.temporal_module,
                    device_ids=[self.rank]
                )
    
    def setup_data(self):
        """Setup dataloaders"""
        dataset = ContentStyleDataset(
            content_dir=self.config['content_dir'],
            style_dir=self.config['style_dir'],
            image_size=self.config['image_size']
        )
        
        # Use distributed sampler if multi-GPU
        if self.world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True
            )
        else:
            sampler = None
        
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )
    
    def setup_training(self):
        """Setup optimizer and losses"""
        # Only train decoder
        params = list(self.style_net.module.decoder.parameters() if self.world_size > 1 
                     else self.style_net.decoder.parameters())
        
        self.optimizer = torch.optim.Adam(
            params,
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 1e-5)
        )
        
        # Losses
        self.content_loss = PerceptualLoss(self.encoder)
        self.style_loss = StyleLoss(self.encoder)
        
        # Loss weights
        self.content_weight = self.config.get('content_weight', 1.0)
        self.style_weight = self.config.get('style_weight', 10.0)
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.style_net.train()
        
        if self.world_size > 1:
            self.dataloader.sampler.set_epoch(epoch)
        
        total_loss = 0.0
        progress = tqdm(self.dataloader, desc=f"Epoch {epoch}") if self.is_main else self.dataloader
        
        for batch_idx, (content, style) in enumerate(progress):
            content = content.to(self.device)
            style = style.to(self.device)
            
            # Forward pass
            output = self.style_net(content, style)
            
            # Compute losses
            c_loss = self.content_loss(output, content)
            s_loss = self.style_loss(output, style)
            
            loss = self.content_weight * c_loss + self.style_weight * s_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Logging
            if self.is_main and batch_idx % 100 == 0:
                step = epoch * len(self.dataloader) + batch_idx
                self.writer.add_scalar('Loss/total', loss.item(), step)
                self.writer.add_scalar('Loss/content', c_loss.item(), step)
                self.writer.add_scalar('Loss/style', s_loss.item(), step)
        
        avg_loss = total_loss / len(self.dataloader)
        
        if self.is_main:
            print(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")
        
        return avg_loss
    
    def save_checkpoint(self, epoch, path):
        """Save model checkpoint"""
        if not self.is_main:
            return
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        model = self.style_net.module if self.world_size > 1 else self.style_net
        
        torch.save({
            'epoch': epoch,
            'decoder': model.decoder.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config
        }, path)
        
        print(f"Checkpoint saved: {path}")
    
    def train(self):
        """Main training loop"""
        print(f"Starting training on rank {self.rank}")
        print(f"Total epochs: {self.config['epochs']}")
        print(f"Batch size: {self.config['batch_size']} per GPU")
        print(f"Effective batch size: {self.config['batch_size'] * self.world_size}")
        
        for epoch in range(1, self.config['epochs'] + 1):
            avg_loss = self.train_epoch(epoch)
            
            # Save checkpoint
            if self.is_main and epoch % self.config['save_interval'] == 0:
                checkpoint_path = os.path.join(
                    self.config['checkpoint_dir'],
                    f"checkpoint_epoch_{epoch}.pth"
                )
                self.save_checkpoint(epoch, checkpoint_path)
        
        if self.is_main:
            print("Training complete!")
            final_path = os.path.join(self.config['checkpoint_dir'], 'final_model.pth')
            self.save_checkpoint(self.config['epochs'], final_path)
    
    def cleanup(self):
        """Cleanup distributed training"""
        if self.world_size > 1:
            dist.destroy_process_group()


def setup_distributed(rank, world_size):
    """Setup distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group(
        backend='nccl' if torch.cuda.is_available() else 'gloo',
        rank=rank,
        world_size=world_size
    )


def main_worker(rank, world_size, config):
    """Worker process for distributed training"""
    # Setup distributed
    if world_size > 1:
        setup_distributed(rank, world_size)
    
    # Create trainer
    trainer = Trainer(config, rank=rank, world_size=world_size)
    
    # Train
    try:
        trainer.train()
    finally:
        trainer.cleanup()


def main():
    parser = argparse.ArgumentParser(description='Train style transfer network')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Path to config file')
    parser.add_argument('--gpus', type=int, default=1,
                       help='Number of GPUs to use')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set number of GPUs
    world_size = min(args.gpus, torch.cuda.device_count())
    
    print(f"Training with {world_size} GPU(s)")
    
    if world_size > 1:
        # Multi-GPU training
        import torch.multiprocessing as mp
        mp.spawn(
            main_worker,
            args=(world_size, config),
            nprocs=world_size,
            join=True
        )
    else:
        # Single GPU training
        main_worker(0, 1, config)


if __name__ == "__main__":
    main()
