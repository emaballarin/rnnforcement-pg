#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ---- IMPORTS ----
import random
from typing import List, Union
import numpy as np
import torch as th
from torch import Tensor
from torch.nn import Module, Sequential, Linear, Softmax, Identity
import torch.nn.functional as F
from ebtorch.nn import SGRUHCell
from tmaze import TMaze
from util import SimpleSumAverager


# ---- CUSTOM TYPES ----
realnum = Union[float, int]


# ---- AGENT/ENVIRONMENT INSTANTIATION ----
# Prototypical instance (used to get only state/action meta-information)
proto_tmaze: TMaze = TMaze(0, 0, 2, "up", 0, 0, False)
howmany_actions: int = len(proto_tmaze.allowed_actions)
howmany_percepts: int = len(proto_tmaze.percepts)


# ---- MODEL DEFINITION ----

# <><><> Baseline <><><>
# (state: 5, action: 4) -> [2x GRU, hidden: 4] -> [Linear 4 to 1] -> (exp. reward fx: 1)
baseline_readin: Module = Identity()
baseline_readout: Module = Linear(howmany_actions, 1)
baseline_net: Module = SGRUHCell(
    recurrent_input_size=howmany_percepts + howmany_actions,
    hidden_size=4,
    num_layers=2,
    bias=True,
    batch_first=True,
    dropout=0.05,
    bidirectional=False,
    tbptt=False,
    hx=None,
    readin_head=baseline_readin,
    readout_head=baseline_readout,
)

# <><><> Policy <><><>
# (state: 5) -> [2x GRU, hidden: 5] -> [Linear 5 to 4] -> [Softmax] -> (policy distribution: 4)
policy_readin: Module = Identity()
policy_readout: Module = Sequential(
    Linear(howmany_actions + 1, howmany_actions), Softmax(dim=2)
)
policy_net: Module = SGRUHCell(
    recurrent_input_size=howmany_percepts,
    hidden_size=howmany_actions + 1,
    num_layers=2,
    bias=True,
    batch_first=True,
    dropout=0,
    bidirectional=False,
    tbptt=False,
    hx=None,
    readin_head=policy_readin,
    readout_head=policy_readout,
)

# <><><> Utility <><><>
baseline_modules: List[Module] = [baseline_readin, baseline_readout, baseline_net]
policy_modules: List[Module] = [policy_readin, policy_readout, policy_net]
recurrent_modules: List[SGRUHCell] = [baseline_net, policy_net]
all_modules: List[Module] = baseline_modules + policy_modules


# ---- AGENT/ENVIRONMENT SETTINGS ----
REW_PLUS: realnum = 100
REW_MINUS: realnum = -100
REW_WALK: realnum = -1
REW_WALL: realnum = -5
ALLEY_LEN: int = 35
ALLOW_STILL: bool = False


# ---- TRAINING SETTINGS ----
NUM_OF_EPOCHS: int = 50
STINTS_PER_EPOCH: int = 5
GAMES_PER_STINT: int = 20
MOVES_HARD_LIMIT: int = min(10 * ALLEY_LEN, 750)


# ---- PREPARE FOR TRAINING ----
for module in all_modules:
    module.train()

# ---- TRACKERS ----
rewtracker = SimpleSumAverager()  # Gamewise reward
stintwise = SimpleSumAverager()  # Stintwise reward
epochwise = SimpleSumAverager()  # Epochwise reward

SILENT_MOVES: bool = True
SILENT_GAMES: bool = False
SILENT_STINTS: bool = False
SILENT_EPOCHS: bool = False


# ---- TRAINING LOOP ----

# Loop over "epochs"
epoch_nr: int
for epoch_nr in range(NUM_OF_EPOCHS):

    epochwise.reset()

    # Loop over stints (i.e.: updates of the policy network)
    stint_nr: int
    for stint_nr in range(STINTS_PER_EPOCH):

        stintwise.reset()

        # Loop over games
        game_nr: int
        for game_nr in range(GAMES_PER_STINT):

            if not SILENT_GAMES:
                print(
                    "<><> Epoch: {} of {} | Stint: {} of {} | Game: {} of {} <><>".format(
                        epoch_nr + 1,
                        NUM_OF_EPOCHS,
                        stint_nr + 1,
                        STINTS_PER_EPOCH,
                        game_nr + 1,
                        GAMES_PER_STINT,
                    )
                )

            # Break the computational graph in BPTT for RNNs (i.e. new-game signal)
            for module in recurrent_modules:
                module.reset_hx(None)

            # Reset the reward tracker
            rewtracker.reset()

            # Sample the "up"/"down" reward location
            if random.randint(0, 1):
                rewloc: str = "up"
            else:
                rewloc: str = "down"

            # Instantiate the new game
            train_game = TMaze(
                REW_PLUS, REW_MINUS, ALLEY_LEN, rewloc, REW_WALK, REW_WALL, ALLOW_STILL
            )
            elapsed_moves: int = 0

            # Loop over subsequent moves
            while elapsed_moves < MOVES_HARD_LIMIT and not train_game.gameover:
                elapsed_moves += 1

                # Get current state
                curr_state: Tensor = th.tensor(train_game.percepts).float().reshape(
                    1, 1, -1
                )
                curr_state.requires_grad_(True)

                # Estimate best action distribution with PolicyNet
                est_action_dist: Tensor = policy_net(curr_state.detach())[0][0]

                # Sample an action from such distribution
                est_action: int = np.random.choice(
                    np.arange(0, howmany_actions), p=est_action_dist.detach().numpy()
                ).item()

                # Convert such action to one-hot
                est_action_onehot: Tensor = (
                    F.one_hot(th.tensor(est_action), num_classes=howmany_actions)
                    .reshape(1, 1, -1)
                    .float()
                    .detach()
                )

                # Estimate reward with BaselineNet
                est_reward: Tensor = baseline_net(
                    th.cat((curr_state.detach(), est_action_onehot), dim=2)
                )[0][0]

                # Perform selected action
                train_game.take_action(est_action)

                # Get actual reward fx
                rewtracker.consider(float(train_game.reward))
                new_reward: Tensor = th.tensor([rewtracker.average]).float()
                new_reward.requires_grad_(True)

                # Compute losses
                baseline_loss = F.mse_loss(
                    est_reward, new_reward.detach(), reduction="mean"
                )
                policy_loss = -(
                    (1 / STINTS_PER_EPOCH)
                    * th.log(est_action_dist[est_action])
                    * (new_reward.detach() - est_reward.detach())
                )

                # Backwards
                baseline_loss.backward(retain_graph=True)
                policy_loss.backward(retain_graph=True)
                if not SILENT_MOVES:
                    print("                Move # {}".format(elapsed_moves))

            # TODO
            # Game is over
            if not SILENT_GAMES:
                print("            END OF GAME! Reward {}".format(rewtracker.sum))
                print("")
            stintwise.consider(rewtracker.sum)
            #
            #

        # TODO
        # The stint is over: update policy weights!
        if not SILENT_STINTS:
            print("        END OF STINT! Average reward {}".format(stintwise.average))
            if not SILENT_GAMES:
                print("")
                print("")
        epochwise.consider(stintwise.average)
        #
        #

    # TODO
    # Epoch has ended
    if not SILENT_EPOCHS:
        print("    END OF EPOCH! Average reward {}".format(epochwise.average))
        if not SILENT_STINTS:
            print("")
            print("")
    #
    #
