#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# tmaze.py | T-Maze RL environment as a structured GridWorld
#

# Imports
from typing import List, Tuple, Union, Dict
from functools import cached_property

from gridworld import GridWorld

# Custom types
realnum = Union[float, int]

# Auxiliary functions
def _t_place_obstacles(alley_len: int) -> Tuple[Tuple[int, int], ...]:
    temp: List[Tuple[int, int]] = []
    x: int
    for x in range(0, alley_len - 1):
        temp.append((x, 0))
        temp.append((x, 2))
    return tuple(temp)


def _t_place_goals(alley_len: int) -> Tuple[Tuple[int, int], ...]:
    return ((alley_len - 1, 0), (alley_len - 1, 2))


def _t_place_sparse_rewards(
    alley_len: int,
    reward_location: str,
    reward_value: realnum,
    punishment_value: realnum,
) -> Dict[Tuple[int, int], realnum]:

    # Check semantics
    if reward_location not in ("up", "down"):
        raise ValueError(
            "reward_location should be either {} or {}; given: {}".format(
                "up", "down", reward_location
            )
        )

    # Match coordinates and return
    if reward_location == "up":
        val_up, val_down = reward_value, punishment_value
    else:
        val_up, val_down = punishment_value, reward_value

    return {(alley_len - 1, 0): val_down, (alley_len - 1, 2): val_up}


# Class for T-Maze
class TMaze:
    def __init__(
        self,
        reward_value: realnum,
        punishment_value: realnum,
        alley_len: int = 35,
        reward_location: str = "up",
        step_penalty: int = 0,
        wall_penalty: int = 0,
        allow_still: bool = False,
    ) -> None:

        # Check semantics of reward_location
        if reward_location not in ("up", "down"):
            raise ValueError(
                "reward_location should be either {} or {}; given: {}".format(
                    "up", "down", reward_location
                )
            )

        self._binarized_rew_loc: int
        if reward_location == "up":
            self._binarized_rew_loc = 1
        else:
            self._binarized_rew_loc = -1

        self._alley_len: int = alley_len
        self._percepts_bwa: Union[
            Tuple[int, int, int, int, int], Tuple[None, None, None, None, None]
        ] = tuple((None, None, None, None, None))

        self._gw = GridWorld(
            nrows=3,
            ncols=alley_len,
            obstacles_map=_t_place_obstacles(alley_len),
            rewards_map=_t_place_sparse_rewards(
                alley_len, reward_location, reward_value, punishment_value
            ),
            goals_map=_t_place_goals(alley_len),
            init_position=(0, 1),
            init_reward=0,
            init_time=0,
            sparse_obs_map=True,
            sparse_rew_map=True,
            default_rew_if_sparse=step_penalty,
            sparse_goals_map=True,
            allow_still=allow_still,
            reward_on_wall_hit=wall_penalty,
            pedantic_validate=False,
            api_validate=False,
        )

    # AUXILIARY

    @property
    def _lookaround(self) -> Tuple[int, int, int, int]:
        if (
            self._gw.position[0] < 0
            or self._gw.position[0] >= self._alley_len
            or self._gw.position[1] < 0
            or self._gw.position[1] >= 3
            or (
                self._gw.position[1] != 1
                and self._gw.position[0] != self._alley_len - 1
            )
        ):
            raise RuntimeError("World has collapsed! Invalid position detected!")

        elif self._gw.position[0] == 0:
            if self._alley_len == 1:
                return 1, 1, 0, 0
            else:
                return 0, 0, 0, 1
        elif self._gw.position[0] == self._alley_len - 1:
            return 1, 1, 1, 0
        else:
            return 0, 0, 1, 1

    @property
    def _signal(self) -> int:
        if self._gw.timestep == 0:
            return self._binarized_rew_loc
        else:
            return 0

    # ACTIONS

    @cached_property
    def allowed_actions(
        self,
    ) -> Union[Tuple[int, int, int, int, int], Tuple[int, int, int, int]]:
        return self._gw.allowed_action_numbers

    def take_action(self, action: int) -> None:
        self._percepts_bwa = self.percepts
        self._gw.take_action_number(action)

    # STATES | PERCEPTS

    @property
    def percepts(self) -> Tuple[int, int, int, int, int]:
        return tuple(list((self._signal,)) + list(self._lookaround))

    @property
    def reward(self) -> realnum:
        return self._gw.instant_reward

    @property
    def state_readings(self) -> Tuple[int, int, int, int, int, realnum]:
        return tuple(list(self.percepts) + list((self.reward,)))

    @property
    def state_fwa(self) -> Tuple[int, int, int, int, int, int, realnum]:
        return tuple(
            list(self.percepts)
            + list((self._gw.last_action_number,))
            + list((self.reward,))
        )

    @property
    def state_bwa(self) -> Tuple[int, int, int, int, int, int, realnum]:
        return tuple(
            list(self._percepts_bwa)
            + list((self._gw.last_action_number,))
            + list((self.reward,))
        )

    @property
    def gameover(self):
        return self._gw.gameover
