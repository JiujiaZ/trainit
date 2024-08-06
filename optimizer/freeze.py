import jax.numpy as jnp
from jax import custom_jvp, custom_vjp
import jax


def freeze_layer(x: jax.typing.ArrayLike,
                 clip: bool=False,
                 rms_scale: bool=True,
                 freeze: bool=False,
                 key: jax.typing.ArrayLike=None):

    @custom_vjp
    def freeze_fn(x: jax.typing.ArrayLike, key: jax.typing.ArrayLike=None):

        if clip:
          result = jnp.clip(x, -1, 1)
        else:
          result = x

        return result

    def freeze_fwd(x: jax.typing.ArrayLike, key: jax.typing.ArrayLike = None):
        return freeze_fn(x, key), x

    def freeze_bwd(x, grad_out):

        if freeze:
            new_grad = jnp.zeros_like(grad_out)
        elif clip:
            new_grad = grad_out * (jnp.sign(grad_out) * (x - jnp.clip(x, -1, 1)) >= 0)
        else:
            new_grad = grad_out
        return new_grad, None

    freeze_fn.defvjp(freeze_fwd, freeze_bwd)

    result = freeze_fn(x, key)
    if rms_scale:
        result = result * jnp.sqrt(jnp.mean(x ** 2))
    return result
