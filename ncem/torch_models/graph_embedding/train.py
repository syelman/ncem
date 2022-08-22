import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch_data.datamodule import NCEMDataModule
from ncem.torch_models.graph_embedding import GraphEmbedding
from torch_geometric.data import Data
from utils.argparser import parse_args


def main():
    args, arg_groups = parse_args(GraphEmbedding)

    # Checkpoint settings
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(os.path.dirname(__file__), 'checkpoints'),
        save_top_k=1,
        verbose=False,
        monitor="val_loss",
        mode="min",
    )
    # TODO: improve check-pointing stuff.

    # Choosing the dataset
    if args.dataset == "imc":
        # A transform function just for this use case
        def _pre_transform_imc(data: Data):
            if hasattr(data, "transform_done") and data.transform_done:
                return data
            return Data(
                x=data.gene_expression,
                edge_index=data.edge_index,
                transform_done=True,
                num_nodes=data.cell_type.shape[0],
            )
        dm = NCEMDataModule(pre_transform=_pre_transform_imc)
        n_input_features = 36

    else:
        raise NotImplementedError()

    model = GraphEmbedding(
        num_features=n_input_features,
        **vars(arg_groups["GraphEmbedding"]))
    trainer: pl.Trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback])
    trainer.fit(model=model, datamodule=dm)


if __name__ == "__main__":
    main()
