import argparse
import pytorch_lightning as pl
from torch_models.graph_embedding import GraphEmbedding
from torch_models.non_linear_ncem import NonLinearNCEM
from torch_data.datamodule import NCEMDataModule
from utils import transforms

models = {
    "NonLinearNCEM": NonLinearNCEM,
    "GraphEmbedding": GraphEmbedding,
}

model2transform = {
    "NonLinearNCEM": transforms.any2non_linear_ncem,
    "GraphEmbedding": transforms.any2graph_embedding,
}


def _create_model(model_name, arg_groups, **kwargs):
    return models[model_name](**kwargs, **vars(arg_groups[model_name]))


def _default_pre_transform(model_name):
    return model2transform[model_name]


def create_pre_transform(model_name, dataset, conf):
    if conf is None:  # if default conf
        return _default_pre_transform(model_name)
    # TODO: example of a conf.


def create_model(model_name, dataset, conf, args_group):
    if model_name == "GraphEmbedding":
        return _create_model(model_name, args_group, num_features=34)
    elif model_name == "NonLinearNCEM":
        return _create_model(model_name, arg_groups, in_channels=34, out_channels=11)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trainer")
    parser = NCEMDataModule.add_model_specific_args(parser)
    parser.add_argument("--model", choices=models.keys(), help="Model to train")
    parser.add_argument("--init_model", default=None, help="initial model to load")
    parser.add_argument("--conf", default=None, help="experiment to set up")

    temp_args, _ = parser.parse_known_args()
    print(temp_args.dataset)

    parser = pl.Trainer.add_argparse_args(parser)
    parser = models[temp_args.model].add_model_specific_args(parser)
    args = parser.parse_args()
    arg_groups = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        arg_groups[group.title] = argparse.Namespace(**group_dict)

    model = create_model(args.model, args.dataset, args.conf, arg_groups)
    pre_transform = create_pre_transform(args.model, args.dataset, conf=args.conf)
    trainer: pl.Trainer = pl.Trainer.from_argparse_args(args)

    datamodule = NCEMDataModule(pre_transform=pre_transform, **vars(arg_groups["NCEMDataModule"]))
    trainer.fit(model=model, datamodule=datamodule)
