# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
from typing import Any, Tuple

import chex
import jax
from jax import numpy as jnp

from jumanji.environments.routing.connector_3d.constants import (
    DOWN,
    EMPTY,
    LEFT,
    NOOP,
    PATH,
    POSITION,
    RIGHT,
    TARGET,
    UP,
    ZUP,
    ZDOWN,
)
from jumanji.environments.routing.connector_3d.types import Agent, State
from jumanji.environments.routing.connector_3d.utils import (
    get_agent_grid,
    get_correction_mask,
    get_position,
    get_target,
    move_agent,
    move_position,
)


class Generator(abc.ABC):
    """Base class for generators for the connector environment."""

    def __init__(self, grid_size: tuple, num_agents: tuple) -> None:
        """Initialises a connector generator, used to generate grids for the Connector environment.

        Args:
            grid_size: size of the grid to generate. (x, y, z)
            num_agents: number of agents on the grid.
        """
        self._grid_size = grid_size
        self._num_agents = num_agents
        self._init_mapping_arrays()


    @property
    def grid_size(self) -> tuple:
        return self._grid_size

    @property
    def num_agents(self) -> int:
        return self._num_agents

    @abc.abstractmethod
    def __call__(self, key: chex.PRNGKey) -> State:
        """Generates a `Connector` state that contains the grid and the agents' layout.

        Returns:
            A `Connector` state.
        """

    def _init_mapping_arrays(self):
        x, y, z = self.grid_size
        self.mapping_array_1d_to_3d = jnp.stack(jnp.unravel_index(jnp.arange(x*y*z), (z,y,x)), 1) 
        self.mapping_array_3d_to_1d = jnp.arange(x*y*z).reshape((z,y,x))

    def flat_to_tuple(self, flat_position):
        return jnp.take(self.mapping_array_1d_to_3d, flat_position, 0)

    def tuple_to_flat(self, position):
        return self.mapping_array_3d_to_1d[position]


class UniformRandomGenerator(Generator):
    """Randomly generates `Connector` grids that may or may not be solvable. This generator places
    start and target positions uniformly at random on the grid.
    """

    def __init__(self, grid_size: tuple, num_agents: int) -> None:
        """Instantiates a `UniformRandomGenerator`.

        Args:
            grid_size: size of the square grid to generate.
            num_agents: number of agents/paths on the grid.
        """
        super().__init__(grid_size, num_agents)

    def __call__(self, key: chex.PRNGKey) -> State:
        """Generates a `Connector` state that contains the grid and the agents' layout.

        1. Start with a linear array of length x*y*z.
        2. Select locations on that array
        3. Convert those locations to coordinates


        Returns:
            A `Connector` state.
        """
        x, y, z = self.grid_size 

        key, pos_key = jax.random.split(key)
        starts_flat, targets_flat = jax.random.choice(
            key=pos_key,
            a=jnp.arange(x*y*z),         # a=jnp.arange(self.grid_size**2),
            shape=(2, self.num_agents),  # Start and target positions for all agents
            replace=False,               # Start and target positions cannot overlap
        )
        
        # Create 2D points from the flat arrays.
        # starts = jnp.divmod(starts_flat, self.grid_size[0]) 
        # targets = jnp.divmod(targets_flat, self.grid_size[0]) 


        # Get the agent values for starts and positions.
        agent_position_values = jax.vmap(get_position)(jnp.arange(self.num_agents))
        agent_target_values = jax.vmap(get_target)(jnp.arange(self.num_agents))

        # Create empty array.
        grid = jnp.zeros((x*y*z), dtype=jnp.int32)

        # Place the agent values at starts and targets.
        grid = grid.at[starts_flat].set(agent_position_values)
        grid = grid.at[targets_flat].set(agent_target_values)

        # Reshape the array into a n-dim grid
        grid = jnp.reshape(grid, (z, y, x))  # or (z, y, z)  

        # Get the coordinates based on the 1D indices 
        mapping_array = jnp.stack(jnp.unravel_index(jnp.arange(x*y*z), (z,y,x)), 1) 
        starts = jnp.take(mapping_array, starts_flat, 0)
        targets = jnp.take(mapping_array, targets_flat, 0)
        
        agents = jax.vmap(Agent)(
            id=jnp.arange(self.num_agents),
            start=starts,
            target=targets,
            position=starts,
        )        

        step_count = jnp.array(0, jnp.int32)

        return State(key=key, grid=grid, step_count=step_count, agents=agents)


class RandomWalkGenerator(Generator):
    """Randomly generates `Connector` grids that are guaranteed be solvable.

    This generator places start positions randomly on the grid and performs a random walk from each.
    Targets are placed at their terminuses.
    """

    def __init__(self, grid_size: int, num_agents: int) -> None:
        """Instantiates a `RandomWalkGenerator.

        Args:
            grid_size: size of the square grid to generate.
            num_agents: number of agents/paths on the grid.
        """
        super().__init__(grid_size, num_agents)

    def __call__(self, key: chex.PRNGKey) -> State:
        """Generates a `Connector` state that contains the grid and the agents' layout.

        Args:
            key: used to randomly generate the connector grid.

        Returns:
            A `Connector` state.
        """
        key, board_key = jax.random.split(key)
        solved_grid, agents, grid = self.generate_board(board_key)
        step_count = jnp.array(0, jnp.int32)

        return State(key=key, grid=grid, step_count=step_count, agents=agents)

    def generate_board(self, key: chex.PRNGKey) -> Tuple[chex.Array, Agent, chex.Array]:
        """Generates solvable board using random walk.

        Args:
            key: random key.

        Returns:
            Tuple containing solved board, the agents and an empty training board.
        """
        z, y, x = self.grid_size
        grid = jnp.zeros((z, y, x), dtype=jnp.int32)
        key, step_key = jax.random.split(key)
        grid, agents = self._initialize_agents(key, grid)

        stepping_tuple = (step_key, grid, agents)

        
        _, grid, agents = jax.lax.while_loop(
            self._continue_stepping, self._step, stepping_tuple   # condition, function, initial_values 
        )

        # Convert heads and targets to format accepted by generator
        heads = agents.start.T
        targets = agents.position.T

        solved_grid = self.update_solved_board_with_head_target_encodings(
            grid, tuple(heads), tuple(targets)
        )
        # Update agent information to include targets and positions after first step
        agents.target = agents.position
        agents.position = agents.start

        agent_position_values = get_position(jnp.arange(self.num_agents))
        agent_target_values = get_target(jnp.arange(self.num_agents))
        # Populate an empty grid with heads and targets
        grid = jnp.zeros((z, y, x), dtype=jnp.int32)
        grid = grid.at[tuple(agents.start.T)].set(agent_position_values)
        grid = grid.at[tuple(agents.target.T)].set(agent_target_values)
        return solved_grid, agents, grid

    def _step(
        self, stepping_tuple: Tuple[chex.PRNGKey, chex.Array, Agent]
    ) -> Tuple[chex.PRNGKey, chex.Array, Agent]:
        """Takes one step for all agents."""
        key, grid, agents = stepping_tuple
        key, next_key = jax.random.split(key)
        agents, grid = self._step_agents(key, grid, agents)
        return next_key, grid, agents

    def _step_agents(
        self, key: chex.PRNGKey, grid: chex.Array, agents: Agent
    ) -> Tuple[Agent, chex.Array]:
        """Steps all agents at the same time correcting for possible collisions.

        If a collision occurs we place the agent with the lower `agent_id` in its previous position.
        This method is equivalent in function to _step_agents from 'Connector' environment.

        Returns:
            Tuple of agents and grid after having applied each agents' action
        """
        agent_ids = jnp.arange(self.num_agents)
        keys = jax.random.split(key, num=self.num_agents)
        
        # Randomly select action for each agent
        actions = jax.vmap(self._select_action, in_axes=(0, None, 0))(
            keys, grid, agents
        )

        # Step all agents at the same time (separately) and return all of the grids
        new_agents, grids = jax.vmap(self._step_agent, in_axes=(0, None, 0))(
            agents, grid, actions
        )

        # Get grids with only values related to a single agent.
        # For example: remove all other agents from agent 1's grid. Do this for all agents.
        agent_grids = jax.vmap(get_agent_grid)(agent_ids, grids)
        joined_grid = jnp.max(agent_grids, 0)  # join the grids

        # Create a correction mask for possible collisions (see the docs of `get_correction_mask`)
        correction_fn = jax.vmap(get_correction_mask, in_axes=(None, None, 0))
        correction_masks, collided_agents = correction_fn(grid, joined_grid, agent_ids)
        correction_mask = jnp.sum(correction_masks, 0)

        # Correct state.agents
        # Get the correct agents, either old agents (if collision) or new agents if no collision
        agents = jax.vmap(
            lambda collided, old_agent, new_agent: jax.lax.cond(
                collided,
                lambda: old_agent,
                lambda: new_agent,
            )
        )(collided_agents, agents, new_agents)
        # Create the new grid by fixing old one with correction mask and adding the obstacles
        return agents, joined_grid + correction_mask

    def _initialize_agents(
        self, key: chex.PRNGKey, grid: chex.Array
    ) -> Tuple[chex.Array, Agent]:
        """Initializes agents using random starting point and places heads on the grid.

        Args:
            key: random key.
            grid: empty grid.

        Returns:
            Tuple of grid with populated starting points and agents initialized with
            the same starting points.
        """
        x, y, z = self.grid_size
        # Generate locations of heads and an adjacent first move for each agent.
        # Return a grid with these positions populated.
        carry, heads_and_positions = jax.lax.scan(
            self._initialize_starts_and_first_move,
            (key, grid),
            jnp.arange(self.num_agents),
        )
        
        starts, first_step = heads_and_positions
        key, grid = carry

        # Fill target with default value as targets will be assigned after random walk
        targets = jnp.full((3, self.num_agents), -1) # FIXME 

        # # Initialize agents
        agents = jax.vmap(Agent)(
            id=jnp.arange(self.num_agents),
            start=starts,
            target=targets,
            position=first_step,
        )
        return grid, agents

    def _initialize_starts_and_first_move(
        self,
        carry: Tuple[chex.PRNGKey, chex.Array],
        agent_id: int,
    ) -> Tuple[Tuple[chex.PRNGKey, chex.Array], Tuple[chex.Array, chex.Array]]:
        """Initializes the starting positions and firs move of each agent.

        Args:
            carry: contains the current state of the flattened grid and the random key.
            agent_id: id of the agent whose positions are looked for.

        Returns:
            Tuple of indices of the starting position and the first move (in flat coordinates).
        """
        x, y, z = self.grid_size

        key, grid = carry
        key, next_key = jax.random.split(key)
        grid_mask = grid == 0 
        
        start_coordinate_flat = jax.random.choice(
            key=key,
            a=jnp.arange(x*y*z),
            shape=(),
            replace=True,
            p=grid_mask.reshape(-1),
        )

        # Convert flat coordinates to 3D and get agent ID number
        start_coordinate = self._convert_flat_position_to_tuple(start_coordinate_flat) 
        grid = grid.at[tuple(start_coordinate)].set(get_target(agent_id))

        # Get a (6,3) array of cells 
        available_cells = self._available_cells( # Returns a (6x3) array 
            grid, start_coordinate
        )

        # (6,3) coordinates => (6,) boolean mask
        selection_filter = jnp.all(available_cells != jnp.array([-1, -1, -1]), axis=1)
        selection_indices = jnp.arange(available_cells.shape[0]) 
        
        first_move_coordinate_index = jax.random.choice(
            key=key,
            a=selection_indices,
            shape=(),
            replace=True,
            p=selection_filter, 
        )
        first_move_coordinate = available_cells[first_move_coordinate_index]
        grid = grid.at[tuple(first_move_coordinate)].set(get_position(agent_id))

        return (key, grid), (start_coordinate, first_move_coordinate)

    def _place_agent_heads_on_grid(self, grid: chex.Array, agent: Agent) -> chex.Array:
        """Updates grid with agent starting positions."""
        return grid.at[tuple(agent.start)].set(get_position(agent.id))

    def _continue_stepping(
        self, stepping_tuple: Tuple[chex.PRNGKey, chex.Array, Agent]
    ) -> chex.Array:
        """Determines if agents can continue taking steps."""
        _, grid, agents = stepping_tuple

        dones = jax.vmap(self._no_available_cells, in_axes=(None, 0))(grid, agents)
        return ~dones.all()

    def _no_available_cells(self, grid: chex.Array, agent: Agent) -> chex.Array:
        """Checks if there are no moves are available for the agent."""
        # cell = self._convert_tuple_to_flat_position(agent.position)
        return (self._available_cells(grid, agent.position) == jnp.array([-1, -1, -1])).all()

    def _select_action(
        self, key: chex.PRNGKey, grid: chex.Array, agent: Agent
    ) -> chex.Array:
        """Selects action for agent to take given its current position.

        Args:
            key: random key.
            grid: current state of the grid.
            agent: the agent.

        Returns:
            Integer corresponding to the action the agent will take in its next step.
            Action indices match those in connector.constants.
        """
        available_cells = self._available_cells(grid=grid, position=agent.position)

        # (6,3) coordinates => (6,) boolean mask
        selection_filter = jnp.all(available_cells != jnp.array([-1, -1, -1]), axis=1)
        selection_indices = jnp.arange(available_cells.shape[0]) 

        step_coordinate_index = jax.random.choice(
            key=key,
            a=selection_indices,
            shape=(),
            replace=True,
            p=selection_filter, 
        )
        step_coordinate = available_cells[step_coordinate_index]

        # action = self._action_from_positions(agent.position, step_coordinate)
        # return action
        return step_coordinate - agent.position 

    def _init_conversion_table(self):
        """Initializes the lookup table for flat-to-3D conversions"""
        x, y, z = self.grid_size
        self.mapping_array = jnp.stack(jnp.unravel_index(jnp.arange(x*y*z), (z,y,x)), 1) 
        # starts = jnp.take(self.mapping_array, starts_flat, 0)        

    def _convert_flat_position_to_tuple(self, position: chex.Array) -> chex.Array:
        # return jnp.array([(position // self.grid_size), (position % self.grid_size)], dtype=jnp.int32)
        return jnp.take(self.mapping_array_1d_to_3d, position, 0)

    def _convert_tuple_to_flat_position(self, position: chex.Array) -> chex.Array:
        # return jnp.array((position[0] * self.grid_size + position[1]), jnp.int32)
        # jax.vmap(tuple)(test_3d_positions)
        # gen._convert_tuple_to_flat_position(my_tuples)
        return self.mapping_array_3d_to_1d[position] 

    def _action_from_positions(
        self, position_1: chex.Array, position_2: chex.Array
    ) -> chex.Array:
        """Compares two positions and returns action id to get from one to the other."""
        action_tuple = position_2 - position_1
        return self._action_from_tuple(action_tuple)

    def _action_from_tuple(self, action_tuple: chex.Array) -> chex.Array:
        """Returns integer corresponding to taking action defined by action_tuple."""
        # FIXME not sure these are right. Actions and action multiplier might not correspond
        action_multiplier = jnp.array([UP, DOWN, LEFT, RIGHT, ZUP, ZDOWN, NOOP])
        actions = jnp.array(
            [
                (action_tuple == jnp.array([-1, 0, 0])).all(axis=0),
                (action_tuple == jnp.array([1, 0, 0])).all(axis=0),
                (action_tuple == jnp.array([0, -1, 0])).all(axis=0),
                (action_tuple == jnp.array([0, 1, 0])).all(axis=0),
                (action_tuple == jnp.array([0, 0, -1])).all(axis=0),
                (action_tuple == jnp.array([0, 0, 1])).all(axis=0),
                (action_tuple == jnp.array([0, 0, 0])).all(axis=0),
            ]
        )
        actions = jnp.sum(actions * action_multiplier, axis=0)
        return actions

    def _adjacent_cells(self, position: chex.Array) -> chex.Array:
        """Returns chex.Array of adjacent cells to the input.

        Given a cell, return a chex.Array of size 4 with the flat indices of
        adjacent cells. Padded with -1's if less than 4 adjacent cells (if on the edge of the grid).

        Args:
            cell: the flat index of the cell to find adjacent cells of.

        Returns:
            A chex.Array of size 4 with the flat indices of adjacent cells
            (padded with -1's if less than 4 adjacent cells).
        """
        
        z, y, x = self.grid_size 

        available_moves = jnp.array([
            [1, 0, 0],
            [-1,0, 0],
            [0, 1, 0],
            [0,-1, 0],
            [0, 0, 1],
            [0, 0,-1],
        ])  # z, -z, y, -y, x, -x

        possible_locations = position + available_moves
        is_cell_in_grid = jax.vmap(self._is_in_bounds)(possible_locations) 

        negative_table = jnp.broadcast_to(jnp.array([-1,-1,-1]), (6,3)) 
        
        return jnp.where(is_cell_in_grid[:,None], possible_locations, negative_table )

    def _available_cells_OLD(self, grid: chex.Array, cell: chex.Array) -> chex.Array:
        """Returns list of cells that can be stepped into from the input cell's position.

        Given a cell and the grid of the board, see which adjacent cells are available to move to
        (i.e. are currently unoccupied) to avoid stepping over existing wires.

        Args:
            grid: the current layout of the board i.e. current grid.
            cell: the flat index of the cell to find adjacent cells of.

        Returns:
            A chex.Array of size 4 with the flat indices of adjacent cells.
        """
        adjacent_cells = self._adjacent_cells(cell) # 
        # Get the wire id of the current cell
        value = grid[jnp.divmod(cell, self.grid_size)]
        wire_id = (value - 1) // 3

        available_cells_mask = jax.vmap(self._is_cell_free, in_axes=(None, 0))(
            grid, adjacent_cells
        )
        # Also want to check if the cell is touching itself more than once
        touching_cells_mask = jax.vmap(
            self._is_cell_doubling_back, in_axes=(None, None, 0)
        )(grid, wire_id, adjacent_cells)
        available_cells_mask = available_cells_mask & touching_cells_mask
        available_cells = jnp.where(available_cells_mask, adjacent_cells, -1)
        return available_cells

    def _available_cells(self, grid: chex.Array, position: chex.Array) -> chex.Array:
        # Adjacent cells is broken 
        # The position is also wrong 
        # import ipdb; ipdb.set_trace()

        adjacent_cells = self._adjacent_cells(position) 
        
        # Get the wire id of the current cell 
        value = grid[tuple(position)]
        wire_id = (value - 1) // 3
        
        
        available_cells_mask = jax.vmap(self._is_cell_free, in_axes=(None, 0))(
            grid, adjacent_cells
        ) 
        
        # Also want to check if the cell is touching itself more than once
        touching_cells_mask = jax.vmap(
            self._is_cell_doubling_back, in_axes=(None, None, 0)
        )(grid, wire_id, adjacent_cells)
        
        available_cells_mask = available_cells_mask & touching_cells_mask # Incompatible shapes for broadcasting: shapes=[(6, 3), (6,)]
        
        negative_table = jnp.broadcast_to(jnp.array([-1,-1,-1]), (6,3)) 
        available_cells = jnp.where(available_cells_mask[:,None], adjacent_cells, negative_table )
        return available_cells  

    def _is_cell_free(
        self,
        grid: chex.Array,
        coordinate: chex.Array,
    ) -> chex.Array:
        """Check if a given cell is free, i.e. has a value of 0.

        Args:
            grid: the current grid of the board.
            cell: the flat index of the cell to check.

        Returns:
            Boolean indicating whether the cell is free or not.
        """
        
        # Verify the coordinate is not OOB, and the value is 0 
        return jnp.all(coordinate != jnp.array([-1,-1,-1])) & (grid[tuple(coordinate)] == 0)

    def _is_cell_doubling_back(
        self,
        grid: chex.Array,
        wire_id: int,
        position: chex.Array,
    ) -> chex.Array:
        """Checks if moving into an adjacent position results in a wire doubling back on itself.

        Check if the cell is touching any of the wire's own cells more than once.
        This means looking for surrounding cells of value 3 * wire_id + POSITION or
        3 * wire_id + PATH.
        """
        # Get the adjacent cells of the current cell
        adjacent_cells = self._adjacent_cells(position)
        
        def is_cell_doubling_back_inner(
            grid: chex.Array, position: chex.Array
        ) -> chex.Array:
            cell_value = grid[tuple(position)]
            touching_self = (
                (cell_value == 3 * wire_id + POSITION)
                | (cell_value == 3 * wire_id + PATH)
                | (cell_value == 3 * wire_id + TARGET)
            )
            # Verify none of the coordinates are -1 
            return (jnp.all(position != jnp.array([-1, -1, -1]))) & touching_self
        
        # Count the number of adjacent cells with the same wire id
        doubling_back_mask = jax.vmap(is_cell_doubling_back_inner, in_axes=(None, 0))(
            grid, adjacent_cells
        )

        # For each position, check the grid at that point
        # doubling_back_mask = is_cell_doubling_back_inner(grid, adjacent_cells)        

        # If the cell is touching itself more than once, return False
        return jnp.sum(doubling_back_mask) <= 1

    def _step_agent(
        self,
        agent: Agent,
        grid: chex.Array,
        action: chex.Array,
    ) -> Tuple[Agent, chex.Array]:
        """Moves the agent according to the given action if it is possible.

        This method is equivalent in function to _step_agent from 'Connector' environment.

        Returns:
            Tuple of (agent, grid) after having applied the given action.
        """
        # new_pos = move_position(agent.position, action)
        new_pos = agent.position + action 

        new_agent, new_grid = jax.lax.cond(
            self._is_valid_position(grid, agent, new_pos) & jnp.all(action != jnp.array([0,0,0])),
            move_agent,                     # True
            lambda *_: (agent, grid),       # False 
            agent,
            grid,
            new_pos,
        )
        return new_agent, new_grid

    def _is_in_bounds(self, position):
        x_bound, y_bound, z_bound = self.grid_size 
        z, y, x = position  
        return (0 <= x) & (x < x_bound) & (0 <= y) & (y < y_bound) & (0 <= z) & (z < z_bound)

    def _is_valid_position(
        self,
        grid: chex.Array,
        agent: Agent,
        position: chex.Array,
    ) -> chex.Array:
        """Checks to see if the specified agent can move to `position`.

        This method is mirrors the use of to is_valid_position from the 'Connector' environment.

        Args:
            grid: the environment state's grid.
            agent: the agent.
            position: the new position for the agent in tuple format.

        Returns:
            bool: True if the agent moving to position is valid.
        """
        z_bound, y_bound, x_bound = self.grid_size 
        z, y, x = position 

        # Within the bounds of the grid
        in_bounds = (0 <= x) & (x < x_bound) & (0 <= y) & (y < y_bound) & (0 <= z) & (z < z_bound)
        # Cell is not occupied
        open_cell = (grid[tuple(position)] == EMPTY) | (grid[tuple(position)] == get_target(agent.id))
        # Agent is not connected
        not_connected = ~agent.connected

        return in_bounds & open_cell & not_connected

    def update_solved_board_with_head_target_encodings(
        self,
        solved_grid: chex.Array,
        heads: Tuple[Any, ...],
        targets: Tuple[Any, ...],
    ) -> chex.Array:
        """Updates grid array with all agent encodings."""
        agent_position_values = get_position(jnp.arange(self.num_agents))
        agent_target_values = get_target(jnp.arange(self.num_agents))
        # Transpose the agent_position_values to match the shape of the grid.
        # Place the agent values at starts and targets.
        solved_grid = solved_grid.at[heads].set(agent_position_values)
        solved_grid = solved_grid.at[targets].set(agent_target_values)
        return solved_grid


if __name__ == "__main__":
    gen = UniformRandomGenerator(grid_size=(5,5,2), num_agents=3)