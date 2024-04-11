import sys 
sys.path.append('../../../..')
import jax 
import jax.numpy as jnp 
from generator import RandomWalkGenerator 



"""

Example grid: 
    0   1   2   3   4
    5   6   7   8   9   
    10  11  12  13  14
    15  16  17  18  19
    20  21  22  23  24 

    25  26  27  28  29
    30  31  32  33  34
    35  36  37  38  39
    40  41  42  43  44
    45  46  47  48  49  
"""

class Converter:
    def __init__(self, grid_size) -> None:
        self.grid_size = grid_size
        self._init_mapping_array()
        
    def _init_mapping_array(self):
        x, y, z = self.grid_size
        self.mapping_array = jnp.stack(jnp.unravel_index(jnp.arange(x*y*z), (z,y,x)), 1) 

    def flat_to_tuple(self, flat_position):
        return jnp.take(self.mapping_array, flat_position, 0)

    def tuple_to_flat(self, position):
        return self.mapping_array[position]


# Generate ranges 
grid_size = (5, 5, 2)
gen = RandomWalkGenerator(grid_size=grid_size, num_agents=4) 

test_flat_positions = jnp.array([0,4,12,23,26,39,49]) 
test_3d_positions = jnp.array([
    [0,0,0],
    [0,0,4],
    [0,2,2],
    [0,4,3],
    [1,0,1],
    [1,2,4],
    [1,4,4],
])


def test_position_conversions(
        grid_size, 
        test_flat_positions, 
        test_3d_positions
        ):
    from generator import RandomWalkGenerator 

    gen = RandomWalkGenerator(grid_size=grid_size, num_agents=4) 

    t1 = jax.vmap(gen._convert_flat_position_to_tuple)(test_flat_positions) 
    f1 = gen._convert_tuple_to_flat_position(jax.vmap(tuple)(test_3d_positions))

    assert(t1 == test_3d_positions)
    assert(f1 == test_flat_positions) 


def test_boundaries():
    gen = RandomWalkGenerator(grid_size=grid_size, num_agents=4) 

    available_moves = jnp.array([
        [0, 0, 1],
        [0, 0,-1],
        [0, 1, 0],
        [0,-1, 0],
        [1, 0, 0],
        [-1,0, 0],
    ])  # z, -z, y, -y, x, -x 

    truth = jnp.array([False,  True, False,  True, False,  True])

    position = jnp.array([4,4,1])

    possible_locations = available_moves + position 

    # is_id_positive # is the cell value greater than or equal to 0 

    is_cell_in_grid = jax.vmap(gen._is_in_bounds)(possible_locations) 
    assert jnp.all(is_cell_in_grid == truth)



print(gen._convert_flat_position_to_tuple())