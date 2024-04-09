import jax 
import jax.numpy as jnp 


# Generate ranges 
x, y, z = (5, 5, 2)
xy = x*y 

zrange = jnp.arange(z)

def z_filter(n):
    xy=9
    return (n * xy, (n + 1) * xy - 1) 


ranges = jax.vmap(z_filter)(zrange)
print(ranges)

