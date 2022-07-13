
# Shapes

## Base Estimator

from `estimators/base_estimator.py` line 1939
```python
if self.vi_model:
    kl_dummy = np.zeros((self.n_eval_nodes_per_graph,), dtype="float32")
    yield (h_1, sf, h_0, h_0_full, a, a_full, node_covar, g), (h_1, kl_dummy)
else:
    yield (h_1, sf, h_0, h_0_full, a, a_full, node_covar, g), h_1
```

### Shapes
```python
def _get_output_signature(self, resampled: bool = False):
        """Get output signatures.

        Parameters
        ----------
        resampled : bool
            Whether dataset is resampled or not.

        Returns
        -------
        output_signature
        """
        h_1 = tf.TensorSpec(
            shape=(self.n_eval_nodes_per_graph, self.n_features_1), dtype=tf.float32
        )  # input node features
        sf = tf.TensorSpec(shape=(self.n_eval_nodes_per_graph, 1), dtype=tf.float32)  # input node size factors
        h_0 = tf.TensorSpec(
            shape=(self.n_eval_nodes_per_graph, self.n_features_0), dtype=tf.float32
        )  # input node features conditional
        h_0_full = tf.TensorSpec(
            shape=(self.max_nodes, self.n_features_0), dtype=tf.float32
        )  # input node features conditional
        a = tf.SparseTensorSpec(shape=None, dtype=tf.float32)  # adjacency matrix
        a_full = tf.SparseTensorSpec(shape=None, dtype=tf.float32)  # adjacency matrix
        node_covar = tf.TensorSpec(
            shape=(self.n_eval_nodes_per_graph, self.n_node_covariates), dtype=tf.float32
        )  # node-level covariates
        domain = tf.TensorSpec(shape=(self.n_domains,), dtype=tf.int32)  # domain
        reconstruction = tf.TensorSpec(
            shape=(self.n_eval_nodes_per_graph, self.n_features_1), dtype=tf.float32
        )  # node features to reconstruct
        kl_dummy = tf.TensorSpec(shape=(self.n_eval_nodes_per_graph,), dtype=tf.float32)  # dummy for kl loss

        if self.vi_model:
            if resampled:
                output_signature = (
                    (h_1, sf, h_0, h_0_full, a, a_full, node_covar, domain),
                    (reconstruction, kl_dummy),
                    (h_1, sf, h_0, h_0_full, a, a_full, node_covar, domain),
                    (reconstruction, kl_dummy),
                )
            else:
                output_signature = ((h_1, sf, h_0, h_0_full, a, a_full, node_covar, domain), (reconstruction, kl_dummy))
        else:
            if resampled:
                output_signature = (
                    (h_1, sf, h_0, h_0_full, a, a_full, node_covar, domain),
                    reconstruction,
                    (h_1, sf, h_0, h_0_full, a, a_full, node_covar, domain),
                    reconstruction,
                )
            else:
                output_signature = ((h_1, sf, h_0, h_0_full, a, a_full, node_covar, domain), reconstruction)
        return output_signature
```

## Base Estimator Neighbors
### Shapes Known
From generator method on ```base_estimator_neighbors.py``` line 216

```python
if self.vi_model:
    kl_dummy = np.zeros((self.n_eval_nodes_per_graph,), dtype="float32")
    yield (h_targets, h_neighbors, sf, a_neighborhood, node_covar, g), (h_out, kl_dummy)
else:
    yield (h_targets, h_neighbors, sf, a_neighborhood, node_covar, g), h_out
```
where

```python
def _get_output_signature(self, resampled: bool = False):
        """Get output signatures.

        Parameters
        ----------
        resampled : bool
            Whether dataset is resampled or not.

        Returns
        -------
        output_signature
        """
        # target node features
        h_targets = tf.TensorSpec(
            shape=(self.n_eval_nodes_per_graph, self.n_features_in), dtype=tf.float32
        )
        # neighbor node features
        h_neighbors = tf.TensorSpec(
            shape=(self.n_eval_nodes_per_graph, self.n_neighbors_padded, self.n_features_in), dtype=tf.float32
        )
        sf = tf.TensorSpec(shape=(self.n_eval_nodes_per_graph, 1), dtype=tf.float32)  # input node size factors
        # node-level covariates
        node_covar = tf.TensorSpec(shape=(self.n_eval_nodes_per_graph, self.n_node_covariates), dtype=tf.float32)
        # adjacency matrix
        a = tf.TensorSpec(shape=(self.n_eval_nodes_per_graph, self.n_neighbors_padded), dtype=tf.float32)
        # domain
        domain = tf.TensorSpec(shape=(self.n_domains,), dtype=tf.int32)
        # node features to reconstruct
        reconstruction = tf.TensorSpec(shape=(self.n_eval_nodes_per_graph, self.n_features_1), dtype=tf.float32)
        # dummy for kl loss
        kl_dummy = tf.TensorSpec(shape=(self.n_eval_nodes_per_graph,), dtype=tf.float32)

        if self.vi_model:
            if resampled:
                output_signature = (
                    (h_targets, h_neighbors, sf, a, node_covar, domain),
                    (reconstruction, kl_dummy),
                    (h_targets, h_neighbors, sf, a, node_covar, domain),
                    (reconstruction, kl_dummy),
                )
            else:
                output_signature = ((h_targets, h_neighbors, sf, a, node_covar, domain),
                                    (reconstruction, kl_dummy))
        else:
            if resampled:
                output_signature = (
                    (h_targets, h_neighbors, sf, a, node_covar, domain),
                    reconstruction,
                    (h_targets, h_neighbors, sf, a, node_covar, domain),
                    reconstruction,
                )
            else:
                output_signature = ((h_targets, h_neighbors, sf, a, node_covar, domain),
                                    reconstruction)
        # print(output_signature)
        return output_signature
```

