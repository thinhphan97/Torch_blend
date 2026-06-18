import pytest
import torch

@pytest.fixture
def sample_tensors():
    """Provides sample tensors for testing"""
    H, W, C = 64, 64, 3
    img1 = torch.randint(0, 256, (H, W, C), dtype=torch.uint8)
    img2 = torch.randint(0, 256, (H, W, C), dtype=torch.uint8)
    mask = torch.randint(0, 256, (H, W), dtype=torch.uint8)
    return img1, img2, mask

@pytest.fixture
def edge_case_tensors():
    """Provides edge case tensors (alpha = 0 and alpha = 255)"""
    H, W, C = 16, 16, 3
    img1 = torch.full((H, W, C), 100, dtype=torch.uint8)
    img2 = torch.full((H, W, C), 200, dtype=torch.uint8)
    
    # Mask: 0 on left half, 255 on right half
    mask = torch.zeros((H, W), dtype=torch.uint8)
    mask[:, W//2:] = 255
    return img1, img2, mask