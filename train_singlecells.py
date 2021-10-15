#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ---- IMPORTS ----
import random
from typing import List, Union
import numpy as np
import torch as th
from torch import Tensor
from torch.nn import Module, LSTMCell
import torch.nn.functional as F
from ebtorch.nn import SGRUHCell
from ebtorch.optim import MADGRAD
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
baseline_cell = LSTMCell(howmany_percepts + howmany_actions, howmany_actions, bias=True)

# <><><> Policy <><><>
policy_cell = LSTMCell(howmany_percepts, howmany_actions, bias=True)

# <><><> Utility <><><>
baseline_modules: List[Module] = [baseline_cell]
policy_modules: List[Module] = [policy_cell]
recurrent_modules: List[SGRUHCell] = [baseline_cell, policy_cell]
all_modules: List[Module] = baseline_modules + policy_modules

# ---- AGENT/ENVIRONMENT SETTINGS ----
REW_PLUS: realnum = 4.0
REW_MINUS: realnum = -0.1
REW_WALK: realnum = -0.0025
REW_WALL: realnum = -0.0025
ALLEY_LEN: int = 8
ALLOW_STILL: bool = False


# ---- TRAINING SETTINGS ----
NUM_OF_EPOCHS: int = 100000
STINTS_PER_EPOCH: int = 1
GAMES_PER_STINT: int = 20
MOVES_HARD_LIMIT: int = ALLEY_LEN * 1000000  # min(10 * ALLEY_LEN, 1000)
DISCOUNT: float = 0.98

# ---- TRACKERS ----
gamewise = SimpleSumAverager()  # Gamewise reward
stintwise = SimpleSumAverager()  # Stintwise reward
epochwise = SimpleSumAverager()  # Epochwise reward

SILENT_MOVES: bool = True
SILENT_GAMES: bool = True
SILENT_STINTS: bool = True
SILENT_EPOCHS: bool = False


# ---- DEVICE HANDLING ----
AUTODETECT: bool = False
device = th.device("cuda" if th.cuda.is_available() and AUTODETECT else "cpu")


# ---- PREPARE FOR TRAINING ----
for module in all_modules:
    module.to(device)
    module.train()


# ---- OPTIMIZERS ----
BASELINE_OPTIM = MADGRAD(baseline_cell.parameters())
POLICY_OPTIM = MADGRAD(policy_cell.parameters())


# ---- TRAINING LOOP ----

# Loop over "epochs"
epoch_nr: int
for epoch_nr in range(NUM_OF_EPOCHS):

    epochwise.reset()

    # Loop over stints (i.e.: updates of the policy network)
    stint_nr: int
    for stint_nr in range(STINTS_PER_EPOCH):

        stintwise.reset()

        BASELINE_OPTIM.zero_grad()
        POLICY_OPTIM.zero_grad()

        # Loop over games
        game_nr: int
        for game_nr in range(GAMES_PER_STINT):

            gamewise.reset()

            # Break the computational graph in BPTT for RNNs (i.e. new-game signal)
            no_h0c0 = True
            no_h0c00 = True

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

            # ---- PLAY GAME FORWARD ----

            # Sample the "up"/"down" reward location
            if random.randint(0, 1):
                rewloc: str = "up"
            else:
                rewloc: str = "down"

            # Instantiate the new game
            train_game = TMaze(
                REW_PLUS, REW_MINUS, ALLEY_LEN, rewloc, REW_WALK, REW_WALL, ALLOW_STILL
            )
            move_idx: int = -1

            # Lists for backward pass
            rewards: List[realnum] = []
            baseline_bw: List[Tensor] = []
            policy_bw: List[Tensor] = []

            # Loop over subsequent moves
            # while move_idx < MOVES_HARD_LIMIT - 1 and not train_game.gameover:
            while not train_game.gameover:
                move_idx += 1

                # Get current state
                curr_state: Tensor = (
                    th.tensor(train_game.percepts).float().reshape(1, -1).to(device)
                )
                curr_state.requires_grad_(True)

                # Estimate best action distribution with PolicyNet
                if no_h0c0:
                    est_action_dist: Tensor
                    c0: Tensor
                    est_action_dist, c0 = policy_cell(curr_state.detach())
                    no_h0c0 = False
                else:
                    est_action_dist: Tensor
                    c0: Tensor
                    est_action_dist, c0 = policy_cell(
                        curr_state.detach(), (est_action_dist, c0)
                    )

                # Sample an action from such distribution
                legal_dist = (
                    th.softmax(est_action_dist[0], dim=0)
                    * th.tensor(train_game.percepts)[1:]
                )
                legal_dist = legal_dist / legal_dist.sum()
                est_action: int = np.random.choice(
                    np.arange(0, howmany_actions), p=legal_dist.detach().cpu().numpy()
                ).item()

                # Convert such action to one-hot
                est_action_onehot: Tensor = (
                    F.one_hot(th.tensor(est_action), num_classes=howmany_actions)
                    .reshape(1, -1)
                    .float()
                    .detach()
                ).to(device)

                # Estimate return with BaselineNet
                if no_h0c00:
                    est_return: Tensor
                    c00: Tensor
                    est_return, c00 = baseline_cell(
                        th.cat((curr_state.detach(), est_action_onehot), dim=1)
                    )
                    no_h0c00 = False
                else:
                    est_return: Tensor
                    c00: Tensor
                    est_return, c00 = baseline_cell(
                        th.cat((curr_state.detach(), est_action_onehot), dim=1),
                        (est_return, c00),
                    )

                est_return_ = 4.1 * th.sigmoid(est_return[0].sum()) - 0.1
                est_return_ = est_return_.reshape(1)

                # Perform selected action
                train_game.take_action(est_action)

                # Log reward
                gamewise.consider(train_game.reward)

                # Gather required elements from the game...
                rewards.append(train_game.reward)
                # ... and from the graph
                baseline_bw.append(est_return_)
                policy_bw.append(th.log(legal_dist[est_action]))

                if not SILENT_MOVES:
                    print("                Move # {}".format(move_idx + 1))

            # Game over! :(
            if not SILENT_GAMES:
                print(
                    "            END OF GAME {}! Reward {}".format(
                        game_nr + 1, gamewise.sum
                    )
                )
                print("")
            stintwise.consider(gamewise.sum)

            # Convert rewards to returns
            returns: List[realnum] = []
            i: int
            for i in range(len(rewards)):
                return_i: float = 0.0
                j: int
                for j in range(i, len(rewards)):
                    return_i += rewards[j] * (DISCOUNT ** j)
                returns.append(return_i)

            # ---- PLAY GAME BACKWARD ----

            assert len(rewards) == len(returns) == len(baseline_bw) == len(policy_bw)

            for move_index in range(len(rewards)):

                # Tensorize return
                new_return = th.tensor([returns[move_idx]]).float().to(device)
                new_return.requires_grad_(True)

                # Compute losses
                baseline_loss = F.l1_loss(
                    baseline_bw[move_idx], new_return.detach(), reduction="sum"
                )

                policy_loss = (
                    -(1 / STINTS_PER_EPOCH)
                    * policy_bw[move_idx]
                    * (new_return.detach() - baseline_bw[move_idx].detach())
                )

                # Backwards (i.e. gradient accumulation)
                baseline_loss.backward(retain_graph=True)
                policy_loss.backward(retain_graph=True)

            # Game over (BACKWARD)

        # The stint is over: update policy weights!
        th.nn.utils.clip_grad_norm_(baseline_cell.parameters(), 0.3)
        BASELINE_OPTIM.step()
        th.nn.utils.clip_grad_norm_(policy_cell.parameters(), 0.3)
        POLICY_OPTIM.step()
        if not SILENT_STINTS:
            print(
                "        END OF STINT {}! Average reward {}".format(
                    stint_nr + 1, stintwise.average
                )
            )
            if not SILENT_GAMES:
                print("")
                print("")
        epochwise.consider(stintwise.average)

    # Epoch has ended
    if not SILENT_EPOCHS:
        print(
            "    END OF EPOCH {}! Average reward {}".format(
                epoch_nr + 1, epochwise.average
            )
        )
        if not SILENT_STINTS:
            print("")
            print("")
        if epochwise.average > 3.95:
            raise RuntimeError("You won!")
