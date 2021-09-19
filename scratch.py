#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from gridworld import GridWorld

pippo = GridWorld(
    2,
    3,
    (),
    {(1, 1): 100},
    (),
    sparse_rew_map=True,
    sparse_obs_map=True,
    sparse_goals_map=True,
    do_validate=True,
    pedantic_validate=True,
    api_validate=True,
    reward_on_wall_hit=-10,
)

print(pippo.state)
pippo.take_action_string("north")
print(pippo.state)
pippo.take_action_string("north")
print(pippo.state)
pippo.take_action_string("north")
print(pippo.state)
pippo.take_action_string("north")
print(pippo.state)
pippo.take_action_string("east")
print(pippo.state)
pippo.take_action_string("east")
print(pippo.state)
pippo.take_action_string("east")
print(pippo.state)
pippo.take_action_string("east")
print(pippo.state)
pippo.take_action_string("east")
print(pippo.state)
pippo.take_action_string("east")
print(pippo.state)
pippo.take_action_string("east")
print(pippo.state)
pippo.take_action_string("east")
print(pippo.state)
