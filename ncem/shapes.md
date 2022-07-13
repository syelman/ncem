
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

### Shapes Known
```python
g = np.zeros((self.n_domains,), dtype="int32")

a_shape = np.asarray((self.n_eval_nodes_per_graph, self.max_nodes), dtype="int64")
a = tf.SparseTensor(indices=a_ind, values=a_val, dense_shape=a_shape)

afull_shape = np.asarray((self.max_nodes, self.max_nodes), dtype="int64")
a_full = tf.SparseTensor(indices=afull_ind, values=afull_val, dense_shape=afull_shape)

node_covar = self.node_covar[key][idx_nodes]
diff = self.max_nodes - node_covar.shape[0]
zeros = np.zeros((diff, node_covar.shape[1]))
node_covar = np.asarray(np.concatenate([node_covar, zeros], axis=0), dtype="float32")
node_covar = node_covar[indices, :]

kl_dummy = np.zeros((self.n_eval_nodes_per_graph,), dtype="float32")
```

### Uncertain Shapes

```python

### h_1
h_1 = self.h_1[key][idx_nodes]
diff = self.max_nodes - h_1.shape[0]
zeros = np.zeros((diff, h_1.shape[1]), dtype="float32")
h_1 = np.asarray(np.concatenate((h_1, zeros), axis=0), dtype="float32")
h_1 = h_1[indices]


### sf
sf = np.expand_dims(self.size_factors[key][idx_nodes], axis=1)
diff = self.max_nodes - sf.shape[0]
zeros = np.zeros((diff, sf.shape[1]))
sf = np.asarray(np.concatenate([sf, zeros], axis=0), dtype="float32")
sf = sf[indices, :]

### h_0 and h_0_full
h_0 = self.h_0[key][idx_nodes]
diff = self.max_nodes - h_0.shape[0]
zeros = np.zeros((diff, h_0.shape[1]), dtype="float32")
h_0_full = np.asarray(np.concatenate((h_0, zeros), axis=0), dtype="float32")
h_0 = h_0_full[indices]
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
g = np.zeros((self.n_domains,), dtype="int32")
a_neighborhood = np.zeros((self.n_eval_nodes_per_graph, self.n_neighbors_padded), "float32")
kl_dummy = np.zeros((self.n_eval_nodes_per_graph,), dtype="float32")
```
### Uncertain shapes

``` python

### sf
sf = np.expand_dims(self.size_factors[key][idx_nodes][indices], axis=1)
## above is equivalent to 
sf = self.size_factors[key][idx_nodes][indices][:,np.newaxis]

### h_out
h_out = self.h_1[key][idx_nodes[indices], :]

### node_covar
node_covar = self.node_covar[key][idx_nodes][indices, :]

### h_neighbors
h_neighbors = []
for i, j in enumerate(idx_nodes[indices]):
    idx_neighbors = np.where(a_j > 0.)[0]
    if self.h0_in:
        h_neighbors_j = self.h_0[key][idx_neighbors, :]
    else:
        h_neighbors_j = self.h_1[key][idx_neighbors, :][:, self.idx_neighbor_features]
    h_neighbors_j = np.expand_dims(h_neighbors_j, axis=0)
    # Pad neighborhoods:
    diff = self.n_neighbors_padded - h_neighbors_j.shape[1]
    zeros = np.zeros((1, diff, h_neighbors_j.shape[2]), dtype="float32")
    h_neighbors_j = np.concatenate([h_neighbors_j, zeros], axis=1)
    h_neighbors.append(h_neighbors_j)
h_neighbors = np.concatenate(h_neighbors, axis=0)


### h_targets
if self.h0_in:
    h_targets = self.h_0[key][idx_nodes[indices], :]
else:
    h_targets = self.h_1[key][idx_nodes[indices], :][:, self.idx_target_features]
```

