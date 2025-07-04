import pytest
from src.models.FNO.utils import FNODataset

def test_load_fno_dataset():
    dataset = FNODataset(datadir='data/test/spect',
                            split='train',
                            initial_step=8,
                            train_ratio=0.8,
                            val_ratio=0.1,
                            test_ratio=0.1,
                            ds_size=10,
                            num_c=2,
                            return_seed=True)
    x, y, grid, seed = dataset[0]
    assert len(dataset) == 8
    assert x.shape == (128, 128, 8, 2)
    assert y.shape == (128, 128, 101, 2)
    assert grid.shape == (128, 128, 2)
    assert type(seed) == str
