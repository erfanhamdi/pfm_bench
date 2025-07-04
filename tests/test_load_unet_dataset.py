import pytest
from src.models.UNet.utils import UNetDataset

def test_load_unet_dataset():
    dataset = UNetDataset(datadir='data/test/spect',
                            split='train',
                            train_ratio=0.8,
                            val_ratio=0.1,
                            test_ratio=0.1,
                            ds_size=10,
                            num_c=2,
                            apply_transforms=False)
    x, y = dataset[0]
    assert len(dataset) == 8
    assert x.shape == (2, 128, 128)
    assert y.shape == (2, 128, 128)
