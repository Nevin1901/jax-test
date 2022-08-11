import jax.numpy as jnp
from jax import grad, jit

def tanh(x):
    y = jnp.exp(-2.0 * x)
    return (1.0 - y) / (1.0 + y)

grad_tanh = grad(tanh)
print(jit(grad_tanh)(1.0))
print(jit(grad_tanh)(2.0))
print(jit(grad_tanh)(3.0))
