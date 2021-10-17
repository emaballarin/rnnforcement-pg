#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# gridworld.py | A minimal GirdWorld environment with robustness in mind
#

# ---- IMPORTS ----
from typing import Tuple, Union, Optional
from functools import cached_property

# ---- CUSTOM TYPES ----
realnum = Union[float, int]


# ---- AUXILIARY FUNCTIONS ----
def _sparsify_to_tuple(denserep, defaultval) -> tuple:
    temp = list()
    for idxa in range(len(denserep)):
        denseslice = denserep[idxa]
        for idxb in range(len(denseslice)):
            if denseslice[idxb] != defaultval:
                temp.append((idxa, idxb))
    return tuple(temp)


def _sparsify_to_dict(denserep, defaultval) -> dict:
    temp = dict()
    for idxa in range(len(denserep)):
        denseslice = denserep[idxa]
        for idxb in range(len(denseslice)):
            densepoint = denseslice[idxb]
            if densepoint != defaultval:
                temp = {**temp, (idxa, idxb): densepoint}
    return temp


def _densify_tuple(sparserep: tuple, nrows, ncols, defaultval, nondefaultval) -> tuple:
    temp = list()
    for _ in range(nrows):
        temp.append([defaultval] * ncols)
    for elem in sparserep:
        temp[elem[0]][elem[1]] = nondefaultval
    return tuple(tuple(inner_elem) for inner_elem in temp)


def _densify_dict(sparserep: dict, nrows, ncols, defaultval) -> tuple:
    temp = list()
    for _ in range(nrows):
        temp.append([defaultval] * ncols)
    for elem in sparserep:
        temp[elem[0]][elem[1]] = sparserep.get(elem)
    return tuple(tuple(inner_elem) for inner_elem in temp)


class GridWorld:
    def __init__(
        self,
        nrows: int,
        ncols: int,
        obstacles_map,
        rewards_map,
        goals_map,
        init_position: Tuple[int, int] = (0, 0),
        init_reward: realnum = 0.0,
        init_time: realnum = 0,
        sparse_obs_map: bool = False,
        need_sparse_obs_map: bool = None,
        sparse_rew_map: bool = False,
        default_rew_if_sparse: realnum = 0.0,
        need_sparse_rew_map: bool = None,
        sparse_goals_map: bool = False,
        need_sparse_goals_map: bool = None,
        allow_still: bool = True,
        gameover: bool = False,
        reward_on_wall_hit: realnum = 0.0,
        do_validate: bool = True,
        pedantic_validate: bool = False,
        api_validate: bool = False,
    ) -> None:

        # Validate anything user-provided (non-pedantic)
        if do_validate:
            # Size of the grid
            if nrows <= 0 or ncols <= 0:
                raise ValueError(
                    "The size of the grid must be positive in both dimensions!"
                )
            # Initial position
            if init_position[0] < 0 or init_position[1] < 0:
                raise ValueError(
                    "Initial position must be nonnegative in both coordinates!"
                )
            if init_position[0] >= ncols or init_position[1] >= nrows:
                raise ValueError(
                    "Initial position must not exceed grid size, in both coordinates!"
                )

        # Validate anything user-provided (pedantic)
        if do_validate and pedantic_validate:
            # Obstacles map
            if sparse_obs_map:
                if len(obstacles_map) > ncols * nrows:
                    raise ValueError(
                        "The number of obstacles cannot exceed the number of places in the grid!"
                    )
                for elem in obstacles_map:
                    if elem[0] < 0 or elem[1] < 0:
                        raise ValueError(
                            "Coordinates of obstacles must be nonnegative!"
                        )
                    if elem[0] >= ncols or elem[1] >= nrows:
                        raise ValueError(
                            "Coordinates of obstacles must not exceed grid size!"
                        )
            else:
                try:
                    for idxa in range(ncols):
                        obs_slice = obstacles_map[idxa]
                        for idxb in range(nrows):
                            obs_point = obs_slice[idxb]
                            if obs_point not in (True, False):
                                raise ValueError(
                                    "Obstacle values in dense representation must be booleans!"
                                )
                except IndexError:
                    raise ValueError(
                        "Obstacle map in dense representation must have a shape of exactly {} by {}".format(
                            ncols, nrows
                        )
                    )

            # Goals map
            if sparse_goals_map:
                if len(goals_map) > ncols * nrows:
                    raise ValueError(
                        "The number of goals cannot exceed the number of places in the grid!"
                    )
                for elem in goals_map:
                    if elem[0] < 0 or elem[1] < 0:
                        raise ValueError("Coordinates of goals must be nonnegative!")
                    if elem[0] >= ncols or elem[1] >= nrows:
                        raise ValueError(
                            "Coordinates of goals must not exceed grid size!"
                        )
            else:
                try:
                    for idxa in range(ncols):
                        goals_slice = goals_map[idxa]
                        for idxb in range(nrows):
                            goals_point = goals_slice[idxb]
                            if goals_point not in (True, False):
                                raise ValueError(
                                    "Goal values in dense representation must be booleans!"
                                )
                except IndexError:
                    raise ValueError(
                        "Goal map in dense representation must have a shape of exactly {} by {}".format(
                            ncols, nrows
                        )
                    )

            # Rewards map
            if sparse_rew_map:
                if len(rewards_map) > ncols * nrows:
                    raise ValueError(
                        "The number of rewards cannot exceed the number of places in the grid!"
                    )
                for elem in rewards_map:
                    if elem[0] < 0 or elem[1] < 0:
                        raise ValueError("Coordinates of rewards must be nonnegative!")
                    if elem[0] >= ncols or elem[1] >= nrows:
                        raise ValueError(
                            "Coordinates of rewards must not exceed grid size!"
                        )
            else:
                try:
                    for idxa in range(ncols):
                        rew_slice = rewards_map[idxa]
                        for idxb in range(nrows):
                            rew_point = rew_slice[idxb]
                            _ = rew_point
                except IndexError:
                    raise ValueError(
                        "Reward map in dense representation must have a shape of exactly {} by {}".format(
                            ncols, nrows
                        )
                    )

            # Must stay here (i.e. not moved outside the 'pedantic' portion
            # since it requires properly-formed representations!
            if (sparse_obs_map and init_position in obstacles_map) or (
                not sparse_obs_map and obstacles_map[init_position[0]][init_position[1]]
            ):
                raise ValueError("Initial position must not be an obstacle!")

            self._api_validate = api_validate

        # Instantiate constant properties of the GridWorld

        if allow_still:
            self._allowed_action_nr: Tuple[int, int, int, int, int] = (0, 1, 2, 3, 4)
            self._allowed_action_str: Tuple[str, str, str, str, str] = (
                "north",
                "south",
                "west",
                "east",
                "still",
            )
        else:
            self._allowed_actions_nr: Tuple[int, int, int, int] = (0, 1, 2, 3)
            self._allowed_action_str: Tuple[str, str, str, str] = (
                "north",
                "south",
                "west",
                "east",
            )

        self._nrows: int = nrows
        self._ncols: int = ncols

        # Obstacles map
        if need_sparse_obs_map is None:
            need_sparse_obs_map = sparse_obs_map
        self._sparse_obstacles = need_sparse_obs_map
        if need_sparse_obs_map and not sparse_obs_map:
            self._obstacles_map = _sparsify_to_tuple(obstacles_map, False)
        elif not need_sparse_obs_map and sparse_obs_map:
            self._obstacles_map = _densify_tuple(
                obstacles_map, nrows, ncols, False, True
            )
        else:
            self._obstacles_map = obstacles_map

        # Goals map
        if need_sparse_goals_map is None:
            need_sparse_goals_map = sparse_goals_map
        self._sparse_goals = need_sparse_goals_map
        if need_sparse_goals_map and not sparse_goals_map:
            self._goals_map = _sparsify_to_tuple(goals_map, False)
        elif not need_sparse_goals_map and sparse_goals_map:
            self._goals_map = _densify_tuple(goals_map, nrows, ncols, False, True)
        else:
            self._goals_map = goals_map

        # Rewards map
        if need_sparse_rew_map is None:
            need_sparse_rew_map = sparse_rew_map
        self._sparse_rewards = need_sparse_rew_map
        if need_sparse_rew_map and not sparse_rew_map:
            self._rewards_map = _sparsify_to_dict(rewards_map, default_rew_if_sparse)
        elif not need_sparse_rew_map and sparse_rew_map:
            self._rewards_map = _densify_dict(
                rewards_map, ncols, nrows, default_rew_if_sparse
            )
        else:
            self._rewards_map = rewards_map

        self._default_rew_if_sparse = default_rew_if_sparse
        self._reward_on_wall_hit = reward_on_wall_hit

        # Instantiate mutable properties of the Gridworld (state & related)

        self._timestep: realnum = init_time
        self._last_action: Union[None, int] = None
        self._last_position: Optional[Tuple[int, int]] = None
        self._position: Tuple[int, int] = init_position
        self._instant_reward: realnum = init_reward
        self._cumul_reward: realnum = init_reward
        self._gameover: bool = gameover

    # Getters / Properties

    @cached_property
    def allowed_action_numbers(
        self,
    ) -> Union[Tuple[int, int, int, int, int], Tuple[int, int, int, int]]:
        return self._allowed_actions_nr

    @cached_property
    def allowed_action_strings(
        self,
    ) -> Union[Tuple[str, str, str, str, str], Tuple[str, str, str, str]]:
        return self._allowed_action_str

    @cached_property
    def allowed_action_howmany(self) -> int:
        return len(self.allowed_action_numbers)

    @cached_property
    def nrows(self):
        return self._nrows

    @cached_property
    def ncols(self):
        return self._ncols

    @property
    def position(self) -> Tuple[int, int]:
        return self._position

    @property
    def instant_reward(self) -> realnum:
        return self._instant_reward

    @property
    def cumul_reward(self) -> realnum:
        return self._cumul_reward

    @property
    def timestep(self) -> int:
        return self._timestep

    @property
    def last_action_number(self) -> Optional[int]:
        return self._last_action

    @property
    def last_position(self) -> Optional[Tuple[int, int]]:
        return self._last_position

    @property
    def last_action_string(self) -> str:
        return self.allowed_action_strings[self.last_action_number]

    @property
    def gameover(self):
        return self._gameover

    # Auxiliary methods
    def _check_for_obstacles_and_update_position(self, proposal):
        self._last_position = self._position
        if self._sparse_obstacles:
            if proposal not in self._obstacles_map:
                self._position = proposal
            else:
                self._instant_reward += self._reward_on_wall_hit
        else:
            if not self._obstacles_map[proposal[0]][proposal[1]]:
                self._position = proposal
            else:
                self._instant_reward += self._reward_on_wall_hit

    def _go_north(self):
        proposal = (self.position[0], self.position[1] + 1)
        if proposal[1] < self.nrows:
            self._check_for_obstacles_and_update_position(proposal)
        else:
            self._instant_reward += self._reward_on_wall_hit

    def _go_south(self):
        proposal = (self.position[0], self.position[1] - 1)
        if proposal[1] >= 0:
            self._check_for_obstacles_and_update_position(proposal)
        else:
            self._instant_reward += self._reward_on_wall_hit

    def _go_east(self):
        proposal = (self.position[0] + 1, self.position[1])
        if proposal[0] < self.ncols:
            self._check_for_obstacles_and_update_position(proposal)
        else:
            self._instant_reward += self._reward_on_wall_hit

    def _go_west(self):
        proposal = (self.position[0] - 1, self.position[1])
        if proposal[0] >= 0:
            self._check_for_obstacles_and_update_position(proposal)
        else:
            self._instant_reward += self._reward_on_wall_hit

    def _check_for_reward_and_update_rewards(self):
        if self._sparse_rewards and self.position in self._rewards_map:
            self._instant_reward += self._rewards_map.get(self.position)
        elif self._sparse_rewards and self.position not in self._rewards_map:
            self._instant_reward += self._default_rew_if_sparse
        elif (
            not self._sparse_rewards
            and self._rewards_map[self.position[0]][self.position[1]] != 0
        ):
            self._instant_reward += self._rewards_map[self.position[0]][
                self.position[1]
            ]

        self._cumul_reward += self._instant_reward

    def _check_for_goal_and_end_game(self):
        if (self._sparse_goals and self.position in self._goals_map) or (
            not self._sparse_goals
            and self._goals_map[self.position[0]][self.position[1]]
        ):
            self._gameover = True

    # Interaction "API"

    def take_action_number(self, action_number):

        self._instant_reward = 0.0

        if action_number == 0:
            self._go_north()
        elif action_number == 1:
            self._go_south()
        elif action_number == 2:
            self._go_west()
        elif action_number == 3:
            self._go_east()
        elif action_number == 4 and self.allowed_action_howmany == 5:
            pass  # Stay still
        else:
            if self._api_validate:
                raise RuntimeError("Action selected is unbound!")

        self._check_for_reward_and_update_rewards()
        self._check_for_goal_and_end_game()
        self._last_action = action_number
        self._timestep += 1

    def take_action_string(self, action_string):
        self.take_action_number(self.allowed_action_strings.index(action_string))
