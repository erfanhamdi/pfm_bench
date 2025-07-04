import pytest
import os
from src.download_data import download_dataset
import shutil

def test_download_data():
    download_dataset(case='test', decomp='spect', outdir='data/test/spect')
    assert os.path.exists('data/test/spect')