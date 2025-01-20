import jax
import jax.numpy as jnp

def mrn_distance(x, y, K=16):
    x_split = jnp.array_split(x, K, axis=-1)
    y_split = jnp.array_split(y, K, axis=-1)
    dists = [mrn_distance_component(x_split[i], y_split[i]) for i in range(K)]
    return jnp.stack(dists, axis=-1).mean(axis=-1)

def mrn_distance_component(x, y):
    eps = 1e-6
    d = x.shape[-1]
    x_prefix = x[..., :d // 2]
    x_suffix = x[..., d // 2:]
    y_prefix = y[..., :d // 2]
    y_suffix = y[..., d // 2:]
    max_component = jnp.max(jax.nn.relu(x_prefix - y_prefix), axis=-1)
    l2_component = jnp.sqrt(jnp.square(x_suffix - y_suffix).sum(axis=-1) + eps)
    assert max_component.shape == l2_component.shape
    return max_component + l2_component

