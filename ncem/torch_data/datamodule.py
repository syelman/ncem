from argparse import ArgumentParser, Namespace
from random import choices
import pytorch_lightning as pl
from typing import Optional, Union
import squidpy as sq
import torch
from torch_geometric.loader import RandomNodeSampler
import pandas as pd
from torch_geometric.data import Data

class NCEMDataModule(pl.LightningDataModule):

    DATASET_LIST = [
        "imc", "mibitof"
    ]

    def __init__(self, pre_transform, **model_kwargs):
        super().__init__()
        self.save_hyperparameters(model_kwargs)
        # TODO: Make the DataModule agnostic of other models
        # This is hard since we won't know what x and y is and we will need dynamic resolving 
        self.pre_transform = pre_transform

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("NCEMDataModule")
        parser.add_argument("--level", type=str, default="node", choices=["graph", "node"])
        parser.add_argument("--dataset", type=str, default="imc", choices=NCEMDataModule.DATASET_LIST)
        parser.add_argument("--batch_size", type=int, default=10, help="Batch Size")
        return parent_parser

    def setup(self, stage: Optional[str] = None):
        if self.hparams["dataset"] == 'imc':
            # Load dataset
            self.adata = adata = sq.datasets.imc()

            sq.gr.spatial_neighbors(adata, coord_type="generic")
            r, c = adata.obsp['spatial_connectivities'].nonzero()
            edge_index = torch.vstack([torch.from_numpy(r).to(torch.long), torch.from_numpy(c).to(torch.long)])
            cell_type = torch.from_numpy(pd.get_dummies(adata.obs).to_numpy())
            gene_expression = torch.from_numpy(adata.X)
            pre_data = Data(
                edge_index=edge_index,
                gene_expression=gene_expression,
                cell_type=cell_type
            )
            if self.hparams["level"] == 'node':
                self.ncem_data = pre_data if self.pre_transform is None else self.pre_transform(pre_data)
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

    def train_dataloader(self):
        return RandomNodeSampler(self.ncem_data, num_parts=self.hparams.batch_size)  # TODO: implement this properly

    def val_dataloader(self):
        return RandomNodeSampler(self.ncem_data, num_parts=self.hparams.batch_size)

    def test_dataloader(self):
        return RandomNodeSampler(self.ncem_data, num_parts=self.hparams.batch_size)

    def predict_dataloader(self):
        return RandomNodeSampler(self.ncem_data, num_parts=self.hparams.batch_size)

# TODO: Handle maybe import or downloads on prepare_data()?
# TODO: More options for the confs.
# TODO: More dataset examples.
# TODO: Graph level case example.
# TODO: Handling splitting but for this we need more examples.
