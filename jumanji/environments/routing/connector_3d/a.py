import jax 
import jax.numpy as jnp

cell = 7 

def test_cells(cell):
    x, y, z = (5,5,2)

    available_moves = jnp.full(6, cell) # [cell, cell, cell, cell, cell, cell] 
    direction_operations = jnp.array([-1*x*y, x*y, -1*y, y, -1, 1]) # -z, +z, -y, +y, -x, +x

    # import ipdb; ipdb.set_trace()

    # Create a mask to check 0 <= index < total size
    cells_to_check = available_moves + direction_operations
    is_id_in_grid = cells_to_check < x*y*z
    is_id_positive = 0 <= cells_to_check
    mask = is_id_positive & is_id_in_grid

    # Ensure adjacent cells doesn't involve going off the xy bounds.
    unflatten_available = jnp.divmod(cells_to_check, 5) 
    unflatten_current = jnp.divmod(cell, 5) 

    is_same_row = unflatten_available[0] == unflatten_current[0]
    is_same_col = unflatten_available[1] == unflatten_current[1]
    row_col_mask = is_same_row | is_same_col


    # Combine the two masks
    mask = mask & row_col_mask 
    
    output = jnp.where(mask, cells_to_check, -1) 
    jax.debug.print("cell {cell} -> output {output} \n", cell=cell, output=output) 


for item in [0,5,14,25,29,37,49]:
    test_cells(item) 