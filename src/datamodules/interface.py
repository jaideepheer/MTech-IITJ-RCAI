from typing import Tuple

import torch


class DatamoduleInterface:
    @property
    def data_shapes(self) -> Tuple[torch.Size]:
        """
        Returns a tuple containing shapes of the data tensors.
        """
        raise NotImplementedError()
    
    def setup_data(self):
        """
        Called to download/extract data.
        """
        ...
    
    def setup_splits(self):
        """
        Called to generate data splits.
        """
        ...
    
    def train_dataloader(self):
        """
        Returns training dataloader.
        """
        ...
    
    def validation_dataloader(self):
        """
        Returns validation dataloader.
        """
        ...
    
    def test_dataloader(self):
        """
        Returns test dataloader.
        """
        ...