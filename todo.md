## Todo items for 3d connector environment creation 
* Modify the tests. What scenarios need to be changed? 
* Add new moves (up, down), (left, right), (zup, zdown) # Z down decrements the layer count, zup increments it. Start at layer 0 


# TODO 
# FIXME 
# BUG 
# NOTE

* constants (TODO) 

* types (DONE)
    - (DONE) Agent
    - (DONE) State 
    - (DONE) Observation 

* utils
    - (TODO) get_path
    - (TODO) get_position
    - (TODO) get_target
    - (TODO) is_target
    - (TODO) is_position
    - (TODO) is_path
    - (TODO) get_agent_id
    - (TODO) move_position
    - (TODO) move_agent
    - (TODO) is_valid_position
    - (TODO) connected_or_blocked
    - (TODO) get_agent_grid
    - (TODO) get_correction_mask


* utils_test 
    - (TODO) test_get_path
    - (TODO) test_get_head
    - (TODO) test_get_target
    - (TODO) test_move_position
    - (TODO) test_move_agent
    - (TODO) test_move_agent_invalid
    - (TODO) test_is_valid_position
    - (TODO) test_connected_or_blocked
    - (TODO) test_get_agent_grid

* env CONNECTOR (TODO) 
    - (TODO) reset 
    - (TODO) step
    - (TODO) _step_agents 
    - (TODO) _step_agent 
    - (TODO) _get_action_mask 
    - (TODO) _get_extras                          
    - (TODO) render 
    - (TODO) animate
    - (TODO) observation_spec 
    - (TODO) action_spec    

* env_tests
    - (TODO) is_head_on_grid
    - (TODO) is_target_on_grid
    - (TODO) test_connector__reset
    - (TODO) test_connector__reset_jit
    - (TODO) test_connector__step_connected
    - (TODO) test_connector__step_blocked
    - (TODO) test_connector__step_horizon
    - (TODO) test_connector__step_agents_collision
    - (TODO) test_connector__step_agent_valid
    - (TODO) test_connector__step_agent_invalid
    - (TODO) test_connector__does_not_smoke
    - (TODO) test_connector__specs_does_not_smoke
    - (TODO) test_connector__get_action_mask
    - (TODO) test_connector__get_extras

* conf_tests 
    - (TODO) grid
    - (TODO) state1
    - (TODO) state2
    - (TODO) action1
    - (TODO) action2

* generator_test 
    - (TODO) valid_starting_grid
    - (TODO) valid_starting_grid_after_1_step
    - (TODO) valid_starting_grid_initialize_agents
    - (TODO) valid_solved_grid_1
    - (TODO) valid_training_grid
    - (TODO) valid_solved_grid_2
    - (TODO) grid_to_test_available_cells
    - (TODO) grids_after_1_agent_step
    - (TODO) agents_finished
    - (TODO) agents_reshaped_for_generator
    - (TODO) agents_starting
    - (TODO) agents_starting_initialise_agents
    - (TODO) agents_starting_move_after_1_step
    - (TODO) agents_starting_move_1_step_up
    - (TODO) generate_board_agents
    - (TODO) TestRandomWalkGenerator 

* generator
    - (TODO) UniformRandomGenerator 
    - (TODO) RandomWalkGenerator 

* reward 
    - (TODO) RewardFn
    - (TODO) DenseRewardFn

* reward_test 
    - (TODO) test_dense_reward 

* viewer (TODO) 