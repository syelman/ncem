from torch_geometric.data import Data


def any2graph_embedding(data: Data):
    if hasattr(data, "transform_done") and data.transform_done:
        return data
    return Data(
        x=data.gene_expression,
        edge_index=data.edge_index,
        transform_done=True,
        num_nodes=data.cell_type.shape[0],
    )


def any2non_linear_ncem(data: Data):
    if hasattr(data, "transform_done") and data.transform_done:
        return data
    return Data(
        x=data.gene_expression,
        y=data.cell_type,
        edge_index=data.edge_index,
        transform_done=True,
        num_nodes=data.cell_type.shape[0],
    )
