# ===== src/hexagonal_research.py =====
"""
Hexagonal Vision Transformer Research Implementation
Dr. Hillary Danan, 2025

Exploring the unexplored: Hexagonal patch extraction for vision transformers

Research Question: Despite hexagonal packing being mathematically proven 15.47% more 
efficient than square packing (Thue, 1890; Fejes Tóth, 1940) and universal in biological 
vision systems, no research exists on hexagonal vision transformers. Why?

This implementation provides tools to investigate this gap.

References:
- Zhao et al. (2021). HexCNN: 42.2% training time reduction, 41.7% memory savings
- Hoogeboom et al. (2018). HexaConv: Group Equivariant CNNs on Hexagonal Lattices
- Dosovitskiy et al. (2021). An Image is Worth 16x16 Words (ViT)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import time
import json
from datetime import datetime

@dataclass
class ResearchConfig:
    """Configuration for reproducible research experiments"""
    image_size: int = 224
    hex_radius: float = 8.0
    square_patch_size: int = 16
    batch_size: int = 32
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed: int = 42
    
    def __post_init__(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        # Compute theoretical values
        self.hex_density = np.pi / (2 * np.sqrt(3))  # ≈ 0.9069
        self.square_density = np.pi / 4  # ≈ 0.7854
        self.theoretical_improvement = (self.hex_density / self.square_density - 1) * 100

class HexagonalGrid:
    """
    Research implementation of hexagonal grid for patch extraction.
    Not optimized - intended for exploration and measurement.
    """
    
    def __init__(self, image_size: int, hex_radius: float):
        self.image_size = image_size
        self.hex_radius = hex_radius
        
        # Mathematical constants from Grünbaum & Shephard (1987)
        self.hex_height = np.sqrt(3) * hex_radius
        self.hex_width = 2 * hex_radius
        
        # Density calculations
        self.hex_density = np.pi / (2 * np.sqrt(3))  # ≈ 0.9069
        self.square_density = np.pi / 4  # ≈ 0.7854
        self.theoretical_improvement = (self.hex_density / self.square_density - 1) * 100
        
        self.centers = self._generate_hex_centers()
        self.num_hexagons = len(self.centers)
        
    def _generate_hex_centers(self) -> List[Tuple[float, float]]:
        """Generate hexagon centers using optimal packing pattern"""
        centers = []
        
        # Pointy-top orientation spacing
        col_spacing = 1.5 * self.hex_radius
        row_spacing = self.hex_height
        
        n_cols = int(np.ceil(self.image_size / col_spacing)) + 1
        n_rows = int(np.ceil(self.image_size / row_spacing)) + 1
        
        for col in range(n_cols):
            for row in range(n_rows):
                x = col * col_spacing
                y = row * row_spacing
                
                # Hexagonal packing offset
                if col % 2 == 1:
                    y += row_spacing / 2
                    
                # Include centers within bounds (with margin)
                if -self.hex_radius <= x <= self.image_size + self.hex_radius:
                    if -self.hex_radius <= y <= self.image_size + self.hex_radius:
                        centers.append((x, y))
                        
        return centers
    
    def compute_theoretical_metrics(self) -> Dict[str, float]:
        """Compute theoretical coverage and efficiency metrics"""
        
        # Count fully contained hexagons
        fully_contained = 0
        for cx, cy in self.centers:
            if (cx - self.hex_radius >= 0 and 
                cx + self.hex_radius <= self.image_size and
                cy - self.hex_radius >= 0 and 
                cy + self.hex_radius <= self.image_size):
                fully_contained += 1
                
        # Compare with square patches
        square_size = int(self.hex_radius * 2)
        squares_per_row = self.image_size // square_size
        num_squares = squares_per_row ** 2
        
        return {
            'theoretical_hex_density': self.hex_density,
            'theoretical_square_density': self.square_density,
            'theoretical_improvement_percent': self.theoretical_improvement,
            'num_hexagons': self.num_hexagons,
            'num_squares': num_squares,
            'fully_contained_hexagons': fully_contained,
            'boundary_hexagons': self.num_hexagons - fully_contained,
            'hex_to_square_ratio': self.num_hexagons / num_squares
        }

class HexagonalPatchExtractor(nn.Module):
    """
    Research implementation of hexagonal patch extraction.
    Simple, unoptimized approach for measurement and comparison.
    """
    
    def __init__(self, image_size: int, hex_radius: float):
        super().__init__()
        self.grid = HexagonalGrid(image_size, hex_radius)
        self.patch_size = int(2 * hex_radius)
        
    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Extract patches and return with extraction statistics"""
        B, C, H, W = images.shape
        device = images.device
        
        patches = torch.zeros(
            B, self.grid.num_hexagons, C * self.patch_size * self.patch_size,
            device=device
        )
        
        stats = {
            'valid_extractions': 0,
            'boundary_cases': 0,
            'extraction_time': 0
        }
        
        start_time = time.time()
        
        for idx, (cx, cy) in enumerate(self.grid.centers):
            # Simple bounding box extraction (not true hexagonal)
            x1 = int(max(0, cx - self.grid.hex_radius))
            x2 = int(min(W, cx + self.grid.hex_radius))
            y1 = int(max(0, cy - self.grid.hex_radius))
            y2 = int(min(H, cy + self.grid.hex_radius))
            
            if x2 > x1 and y2 > y1:
                patch = images[:, :, y1:y2, x1:x2]
                
                # Resize to standard size
                if patch.shape[2] != self.patch_size or patch.shape[3] != self.patch_size:
                    patch = F.interpolate(
                        patch, 
                        size=(self.patch_size, self.patch_size),
                        mode='bilinear',
                        align_corners=False
                    )
                    stats['boundary_cases'] += 1
                    
                patches[:, idx] = patch.flatten(1)
                stats['valid_extractions'] += 1
                
        stats['extraction_time'] = time.time() - start_time
        
        return patches, stats

class SquarePatchExtractor(nn.Module):
    """Standard square patch extraction for baseline comparison"""
    
    def __init__(self, image_size: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        self.unfold = nn.Unfold(
            kernel_size=patch_size,
            stride=patch_size
        )
        
    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Extract square patches using standard PyTorch operations"""
        start_time = time.time()
        
        patches = self.unfold(images)  # [B, C*P*P, N]
        patches = patches.transpose(1, 2)  # [B, N, C*P*P]
        
        stats = {
            'valid_extractions': self.num_patches,
            'boundary_cases': 0,
            'extraction_time': time.time() - start_time
        }
        
        return patches, stats

class MinimalTransformer(nn.Module):
    """
    Minimal transformer for research comparison.
    Not a full ViT - just enough to measure patch extraction impact.
    """
    
    def __init__(
        self, 
        num_patches: int,
        patch_dim: int,
        embed_dim: int = 384,
        depth: int = 6
    ):
        super().__init__()
        
        self.patch_embed = nn.Linear(patch_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)
        
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=6,
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            )
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(patches)
        x = x + self.pos_embed[:, :x.size(1)]
        
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        return x.mean(dim=1)  # Global average pooling

class ResearchExperiment:
    """Run controlled experiments comparing hexagonal vs square patches"""
    
    def __init__(self, config: ResearchConfig):
        self.config = config
        self.results = {
            'config': config.__dict__,
            'timestamp': datetime.now().isoformat()
        }
        
    def analyze_coverage(self) -> Dict:
        """Analyze theoretical coverage properties"""
        print("Analyzing coverage efficiency...")
        
        hex_grid = HexagonalGrid(self.config.image_size, self.config.hex_radius)
        hex_metrics = hex_grid.compute_theoretical_metrics()
        
        # Add square comparison
        square_metrics = {
            'patch_size': self.config.square_patch_size,
            'patches_per_row': self.config.image_size // self.config.square_patch_size,
            'total_patches': (self.config.image_size // self.config.square_patch_size) ** 2
        }
        
        return {
            'hexagonal': hex_metrics,
            'square': square_metrics,
            'comparison': {
                'hex_to_square_patch_ratio': hex_metrics['num_hexagons'] / square_metrics['total_patches'],
                'theoretical_coverage_improvement': hex_metrics['theoretical_improvement_percent']
            }
        }
    
    def benchmark_extraction(self, num_iterations: int = 100) -> Dict:
        """Benchmark extraction performance"""
        print(f"Running extraction benchmark ({num_iterations} iterations)...")
        
        hex_extractor = HexagonalPatchExtractor(
            self.config.image_size, 
            self.config.hex_radius
        ).to(self.config.device)
        
        square_extractor = SquarePatchExtractor(
            self.config.image_size,
            self.config.square_patch_size
        ).to(self.config.device)
        
        # Test data
        test_images = torch.randn(
            self.config.batch_size, 3, 
            self.config.image_size, self.config.image_size,
            device=self.config.device
        )
        
        # Warmup
        for _ in range(10):
            hex_extractor(test_images)
            square_extractor(test_images)
            
        # Benchmark
        hex_times = []
        square_times = []
        
        for _ in range(num_iterations):
            _, hex_stats = hex_extractor(test_images)
            hex_times.append(hex_stats['extraction_time'])
            
            _, square_stats = square_extractor(test_images)
            square_times.append(square_stats['extraction_time'])
            
        return {
            'hexagonal': {
                'mean_time': np.mean(hex_times),
                'std_time': np.std(hex_times),
                'extractions': hex_stats['valid_extractions'],
                'boundary_cases': hex_stats['boundary_cases']
            },
            'square': {
                'mean_time': np.mean(square_times),
                'std_time': np.std(square_times),
                'extractions': square_stats['valid_extractions']
            },
            'relative_speed': np.mean(hex_times) / np.mean(square_times)
        }
    
    def compare_models(self) -> Dict:
        """Compare model architectures"""
        print("Comparing model architectures...")
        
        # Create models
        hex_grid = HexagonalGrid(self.config.image_size, self.config.hex_radius)
        patch_dim = 3 * (2 * int(self.config.hex_radius)) ** 2
        
        hex_model = MinimalTransformer(
            num_patches=hex_grid.num_hexagons,
            patch_dim=patch_dim
        )
        
        square_patches = (self.config.image_size // self.config.square_patch_size) ** 2
        square_patch_dim = 3 * self.config.square_patch_size ** 2
        
        square_model = MinimalTransformer(
            num_patches=square_patches,
            patch_dim=square_patch_dim
        )
        
        # Count parameters
        hex_params = sum(p.numel() for p in hex_model.parameters())
        square_params = sum(p.numel() for p in square_model.parameters())
        
        return {
            'hexagonal': {
                'parameters': hex_params,
                'patches': hex_grid.num_hexagons,
                'patch_dim': patch_dim
            },
            'square': {
                'parameters': square_params,
                'patches': square_patches,
                'patch_dim': square_patch_dim
            },
            'parameter_ratio': hex_params / square_params
        }
    
    def run_all_experiments(self) -> Dict:
        """Run complete experimental suite"""
        print("="*60)
        print("HEXAGONAL VISION TRANSFORMER RESEARCH EXPERIMENTS")
        print("="*60)
        
        self.results['coverage_analysis'] = self.analyze_coverage()
        self.results['extraction_benchmark'] = self.benchmark_extraction()
        self.results['model_comparison'] = self.compare_models()
        
        # Key findings
        self.results['key_findings'] = {
            'coverage_improvement': self.results['coverage_analysis']['comparison']['theoretical_coverage_improvement'],
            'patch_ratio': self.results['coverage_analysis']['comparison']['hex_to_square_patch_ratio'],
            'extraction_overhead': self.results['extraction_benchmark']['relative_speed'],
            'parameter_overhead': self.results['model_comparison']['parameter_ratio']
        }
        
        return self.results
    
    def save_results(self, filepath: str = 'research_results.json'):
        """Save experimental results"""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to {filepath}")

def create_research_visualization():
    """Create visualization of hexagonal packing efficiency"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Hexagonal packing
    ax1.set_title('Hexagonal Packing (Optimal)', fontsize=14, fontweight='bold')
    hex_centers = []
    for i in range(5):
        for j in range(5):
            x = j * 1.5 * 20 + 50
            y = i * np.sqrt(3) * 20 + 50
            if j % 2 == 1:
                y += np.sqrt(3) * 10
            if x < 180 and y < 180:
                hex_centers.append((x, y))
                hexagon = RegularPolygon((x, y), 6, radius=20,
                                       orientation=np.pi/6,
                                       facecolor='lightblue',
                                       edgecolor='blue',
                                       linewidth=2)
                ax1.add_patch(hexagon)
    
    ax1.set_xlim(0, 200)
    ax1.set_ylim(0, 200)
    ax1.set_aspect('equal')
    ax1.text(100, 10, 'Density: π/(2√3) ≈ 90.69%', 
             ha='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow"))
    
    # Square packing
    ax2.set_title('Square Packing (Standard)', fontsize=14, fontweight='bold')
    for i in range(5):
        for j in range(5):
            x = j * 35 + 30
            y = i * 35 + 30
            if x < 180 and y < 180:
                square = plt.Rectangle((x-15, y-15), 30, 30,
                                     facecolor='lightcoral',
                                     edgecolor='red',
                                     linewidth=2)
                ax2.add_patch(square)
    
    ax2.set_xlim(0, 200)
    ax2.set_ylim(0, 200)
    ax2.set_aspect('equal')
    ax2.text(100, 10, 'Density: π/4 ≈ 78.54%', 
             ha='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow"))
    
    for ax in [ax1, ax2]:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    
    plt.suptitle('Mathematical Proof: Hexagonal Packing is 15.47% More Efficient', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('hexagonal_packing_proof.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Visualization saved to hexagonal_packing_proof.png")

def main():
    """Run research experiments"""
    # Configuration
    config = ResearchConfig(
        image_size=224,
        hex_radius=8.0,
        square_patch_size=16,
        batch_size=32
    )
    
    print(f"Research Configuration:")
    print(f"- Image size: {config.image_size}x{config.image_size}")
    print(f"- Hexagon radius: {config.hex_radius}")
    print(f"- Square patch size: {config.square_patch_size}")
    print(f"- Device: {config.device}")
    print(f"- Theoretical improvement: {config.theoretical_improvement:.2f}%")
    print()
    
    # Run experiments
    experiment = ResearchExperiment(config)
    results = experiment.run_all_experiments()
    experiment.save_results()
    
    # Create visualization
    print("\nCreating visualization...")
    create_research_visualization()
    
    # Print summary
    print("\n" + "="*60)
    print("RESEARCH SUMMARY")
    print("="*60)
    
    print(f"\nKey Findings:")
    print(f"- Theoretical coverage improvement: {results['key_findings']['coverage_improvement']:.2f}%")
    print(f"- Hex/Square patch ratio: {results['key_findings']['patch_ratio']:.2f}x")
    print(f"- Current extraction overhead: {results['key_findings']['extraction_overhead']:.2f}x slower")
    print(f"- Model parameter ratio: {results['key_findings']['parameter_overhead']:.2f}x")
    
    print(f"\nResearch Gaps Identified:")
    print(f"1. No existing hexagonal vision transformer implementations")
    print(f"2. No hexagonal positional encoding schemes")
    print(f"3. No optimized hexagonal operations in PyTorch/TensorFlow")
    print(f"4. No benchmark comparisons on standard datasets")
    
    print(f"\nNext Steps:")
    print(f"1. Develop proper hexagonal attention mechanisms")
    print(f"2. Design hexagonal-aware positional encodings")
    print(f"3. Implement optimized hexagonal operations")
    print(f"4. Train and evaluate on ImageNet")

if __name__ == "__main__":
    main()