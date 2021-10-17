#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ---- IMPORTS ----
import random

from typing import List, Union

import numpy as np
import matplotlib.pyplot as plt

import torch as th
from torch import Tensor
from torch.nn import Module, LSTMCell
import torch.nn.functional as F

from ebtorch.optim import MADGRAD  # Facebook's MADGRAD with Nestor Demeure's tweaks

from tmaze import TMaze
from util import SimpleSumAverager


# ---- CUSTOM TYPES ----
realnum = Union[float, int]


# ---- AGENT/ENVIRONMENT INSTANTIATION ----
# Prototypical instance (used to get only state/action meta-information)
proto_tmaze: TMaze = TMaze(0, 0, 2, "up", 0, 0, False)


# ---- AGENT/ENVIRONMENT SETTINGS ----
REW_PLUS: realnum = 4.0  # As in original paper
REW_MINUS: realnum = -0.1  # As in original paper
REW_WALK: realnum = (
    -0.0025
)  # Improvement: reduce length of solutions / faster convergence
REW_WALL: realnum = (
    -0.0025
)  # Improvement: reduce length of solutions / faster convergence (not used in the "only legal moves allowed" scenario)
MIN_ALLEY_LEN: int = 1
MAX_ALLEY_LEN: int = 8
ALLOW_STILL: bool = False


# ---- TRAINING SETTINGS ----
NUM_OF_EPOCHS: int = (
    100000  # Approximation of infinite-epochs regime: stop only on successful training
)
STINTS_PER_EPOCH: int = (
    1  # stint == epoch unless we want to collect different statistics, or schedule LR
)
GAMES_PER_STINT: int = 20  # As in original paper
# MOVES_HARD_LIMIT: int = CURRENT_ALLEY_LEN * 1000000  # min(10 * CURRENT_ALLEY_LEN, 1000)
DISCOUNT: float = 0.98  # As in original paper
REPLICAS = 5

# ---- TRACKERS ----
gamewise = SimpleSumAverager()  # For internal bookkeeping
stintwise = SimpleSumAverager()  # For internal bookkeeping
epochwise = SimpleSumAverager()  # For internal bookkeeping
avgmoves = SimpleSumAverager()  # For graphing
avgrew = SimpleSumAverager()  # For graphing

SILENT_MOVES: bool = True
SILENT_GAMES: bool = True
SILENT_STINTS: bool = True
SILENT_EPOCHS: bool = True


# ---- DEVICE HANDLING ----
AUTODETECT: bool = False  # No great benefit in running on GPU (TODO: transition to CUDNN sequence-wise RNN implementation for very long games)
device = th.device("cuda" if th.cuda.is_available() and AUTODETECT else "cpu")


# ---- TRAINING LOOP(S) ----

# "Open-loop" approach preferred for easier hackability
# TODO: functionalize training
# TODO: Build a proper dataset/loader if going for (ev. partially) off-policy learning


# Loop over alley lengths (for automated graphing)
for alley_len_iter in range(MIN_ALLEY_LEN, MAX_ALLEY_LEN + 1):

    all_replicas_rew = []  # To-be time series to plot
    all_replicas_mov = []  # To-be time series to plot

    # Loop over replicas (for automated graphing)
    replica_nr: int
    for replica_nr in range(REPLICAS):

        # Not strictly needed, since some trackers are reset afterwards; here for extra caution
        gamewise.reset()
        stintwise.reset()
        epochwise.reset()
        avgmoves.reset()
        avgrew.reset()

        single_replica_rew = []  # To-be time series to plot (single replica)
        single_replica_mov = []  # To-be time series to plot (single replica)

        # Properly instantiate the replica

        # ---- MODEL DEFINITION ----

        # <><><> Baseline <><><>
        # Here in their simplest form: 1-layer-deep (depth-wise), sized for dense representation learning (since we assume the "only legal moves scenario")
        baseline_cell = LSTMCell(
            len(proto_tmaze.percepts) + len(proto_tmaze.allowed_actions),
            len(proto_tmaze.allowed_actions),
            bias=True,
        )

        # <><><> Policy <><><>
        # Here in their simplest form: 1-layer-deep (depth-wise), sized for dense representation learning (since we assume the "only legal moves scenario")
        policy_cell = LSTMCell(
            len(proto_tmaze.percepts), len(proto_tmaze.allowed_actions), bias=True
        )

        # <><><> Shorthands <><><>
        baseline_modules: List[Module] = [baseline_cell]
        policy_modules: List[Module] = [policy_cell]
        recurrent_modules: List[Module] = [baseline_cell, policy_cell]
        all_modules: List[Module] = baseline_modules + policy_modules

        # ---- PREPARE FOR TRAINING ----
        for module in all_modules:
            module.to(device)
            module.train()

        # ---- OPTIMIZERS ----
        BASELINE_OPTIM = MADGRAD(baseline_cell.parameters())
        POLICY_OPTIM = MADGRAD(policy_cell.parameters())

        # <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <

        # Loop over "epochs"
        epoch_nr: int
        for epoch_nr in range(NUM_OF_EPOCHS):

            epochwise.reset()

            # Loop over stints (i.e.: updates of the policy network)
            stint_nr: int
            for stint_nr in range(STINTS_PER_EPOCH):

                stintwise.reset()

                avgmoves.reset()
                avgrew.reset()

                BASELINE_OPTIM.zero_grad()  # Accumulate gradient from now on...
                POLICY_OPTIM.zero_grad()  # Accumulate gradient from now on...

                # Loop over games
                game_nr: int
                for game_nr in range(GAMES_PER_STINT):

                    gamewise.reset()

                    # Break the computational graph in BPTT for RNNs (i.e. new-game signal)
                    no_h0c0 = True  # i.e. initialize hidden states to zero
                    no_h0c00 = True  # i.e. initialize hidden states to zero

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
                        REW_PLUS,
                        REW_MINUS,
                        alley_len_iter,
                        rewloc,
                        REW_WALK,
                        REW_WALL,
                        ALLOW_STILL,
                    )
                    move_idx: int = -1

                    # Lists for backward pass
                    rewards: List[realnum] = []
                    baseline_bw: List[Tensor] = []
                    policy_bw: List[Tensor] = []

                    # Loop over subsequent moves
                    while not train_game.gameover:
                        move_idx += 1

                        # Get current state
                        curr_state: Tensor = (
                            th.tensor(train_game.percepts)
                            .float()
                            .reshape(1, -1)
                            .to(device)
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
                            np.arange(0, len(proto_tmaze.allowed_actions)),
                            p=legal_dist.detach().cpu().numpy(),
                        ).item()

                        # Convert such action to one-hot
                        est_action_onehot: Tensor = (
                            F.one_hot(
                                th.tensor(est_action),
                                num_classes=len(proto_tmaze.allowed_actions),
                            )
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

                        # Improvement: since we know return range, enforce that!
                        est_return_ = 4.1 * th.sigmoid(est_return[0].sum()) - 0.1
                        est_return_ = est_return_.reshape(1)

                        # Perform selected action
                        train_game.take_action(est_action)

                        # Log reward
                        gamewise.consider(train_game.reward)

                        # Gather required elements from the game...
                        rewards.append(train_game.reward)
                        # ... and from the computational graph of the RNNs
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
                    avgrew.consider(gamewise.sum)
                    avgmoves.consider(train_game.timestep)

                    # Convert rewards to returns (as in the paper)
                    returns: List[realnum] = []
                    i: int
                    for i in range(len(rewards)):
                        return_i: float = 0.0
                        j: int
                        for j in range(i, len(rewards)):
                            return_i += rewards[j] * (DISCOUNT ** j)
                        returns.append(return_i)

                    # ---- PLAY GAME BACKWARDS ----
                    # Necessary (w.r.t. to out-of-loop) to ensure random sampling is not re-done

                    # A sanity check, run with python -O to suppress
                    assert (
                        len(rewards)
                        == len(returns)
                        == len(baseline_bw)
                        == len(policy_bw)
                    )

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

                        # Backward pass (i.e. gradient accumulation)
                        baseline_loss.backward(
                            retain_graph=True
                        )  # Graph retention necessary
                        policy_loss.backward(
                            retain_graph=True
                        )  # Graph retention necessary

                # The stint is over: update policy weights!

                single_replica_rew.append(avgrew.average)
                single_replica_mov.append(avgmoves.average)

                th.nn.utils.clip_grad_norm_(
                    baseline_cell.parameters(), 0.3
                )  # Also valid: clip at 3rd quartile (e.g. with https://github.com/emaballarin/ebtorch/blob/main/ebtorch/nn/utils/autoclip.py)
                BASELINE_OPTIM.step()
                th.nn.utils.clip_grad_norm_(
                    policy_cell.parameters(), 0.3
                )  # Also valid: clip at 3rd quartile (e.g. with https://github.com/emaballarin/ebtorch/blob/main/ebtorch/nn/utils/autoclip.py)
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
                if not SILENT_EPOCHS:
                    print("Training successful; stopping...")
                break

        # Single replica has finished training
        all_replicas_rew.append(single_replica_rew)
        all_replicas_mov.append(single_replica_mov)

    # Fixed alley len; save graphs!

    # Rewards
    nparray_len_rew = max(map(len, all_replicas_rew))
    plot_array_rew = np.array(
        [sub + [None] * (nparray_len_rew - len(sub)) for sub in all_replicas_rew]
    )

    fig, ax = plt.subplots()
    ax.plot(np.arange(1, nparray_len_rew + 1), plot_array_rew.transpose())
    ax.set(
        xlabel="# of stint",
        ylabel="Avg. reward per game (replicas)",
        title="T-Maze (alley len: {})".format(alley_len_iter),
    )
    ax.grid()
    plt.savefig("plots/tmaze_{}_improved_rew.png".format(alley_len_iter))

    # Number of moves
    nparray_len_mov = max(map(len, all_replicas_mov))
    plot_array_mov = np.array(
        [sub + [None] * (nparray_len_mov - len(sub)) for sub in all_replicas_mov]
    )

    fig, ax = plt.subplots()
    ax.plot(np.arange(1, nparray_len_mov + 1), plot_array_mov.transpose())
    ax.set(
        xlabel="# of stint",
        ylabel="Avg. # of moves per game (replicas)",
        title="T-Maze (alley len: {})".format(alley_len_iter),
    )
    ax.grid()
    plt.savefig("plots/tmaze_{}_improved_mov.png".format(alley_len_iter))
