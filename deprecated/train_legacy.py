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
# (state: 5, action: 4) -> [2x GRU, hidden: 4] -> [Linear 4 to 1] -> (exp. reward fx: 1)
baseline_readin: Module = Identity()
baseline_readout: Module = Linear(howmany_actions, 1)
baseline_net: Module = SGRUHCell(
    recurrent_input_size=1 + howmany_actions,
    hidden_size=4,
    num_layers=1,
    bias=True,
    batch_first=True,
    dropout=0,
    bidirectional=False,
    tbptt=False,
    hx=None,
    readin_head=baseline_readin,
    readout_head=baseline_readout,
)

# <><><> Policy <><><>
# (state: 5) -> [2x GRU, hidden: 5] -> [Linear 5 to 4] -> [Softmax] -> (policy distribution: 4)
policy_readin: Module = Identity()
policy_readout: Module = Sequential(Softmax(dim=2))
policy_net: Module = SGRUHCell(
    recurrent_input_size=1,
    hidden_size=howmany_actions,
    num_layers=1,
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
REW_PLUS: realnum = 4.0
REW_MINUS: realnum = -0.1
REW_WALK: realnum = 0.0
REW_WALL: realnum = 0.0
ALLEY_LEN: int = 5
ALLOW_STILL: bool = False


# ---- TRAINING SETTINGS ----
NUM_OF_EPOCHS: int = 100000
STINTS_PER_EPOCH: int = 1
GAMES_PER_STINT: int = 20
MOVES_HARD_LIMIT: int = min(10 * ALLEY_LEN, 1000)
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
BASELINE_OPTIM = MADGRAD(
    list(baseline_readin.parameters())
    + list(baseline_net.parameters())
    + list(baseline_readout.parameters())
)
POLICY_OPTIM = MADGRAD(
    list(policy_readin.parameters())
    + list(policy_net.parameters())
    + list(policy_readout.parameters())
)


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
            for module in recurrent_modules:
                module.reset_hx(None)

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
            while move_idx < MOVES_HARD_LIMIT - 1 and not train_game.gameover:
                move_idx += 1

                # Get current state
                curr_state: Tensor = (
                    th.tensor(train_game.percepts)[0]
                    .float()
                    .reshape(1, 1, -1)
                    .to(device)
                )
                curr_state.requires_grad_(True)

                # Estimate best action distribution with PolicyNet
                est_action_dist: Tensor = policy_net(curr_state.detach())[0][0]

                # Sample an action from such distribution
                est_action: int = np.random.choice(
                    np.arange(0, howmany_actions),
                    p=est_action_dist.detach().cpu().numpy(),
                ).item()

                # Convert such action to one-hot
                est_action_onehot: Tensor = (
                    F.one_hot(th.tensor(est_action), num_classes=howmany_actions)
                    .reshape(1, 1, -1)
                    .float()
                    .detach()
                ).to(device)

                # Estimate return with BaselineNet
                est_return: Tensor = baseline_net(
                    th.cat((curr_state.detach(), est_action_onehot), dim=2)
                )[0][0]

                # Perform selected action
                train_game.take_action(est_action)

                # Log reward
                gamewise.consider(train_game.reward)

                # Gather required elements from the game...
                rewards.append(train_game.reward)
                # ... and from the graph
                baseline_bw.append(est_return)
                policy_bw.append(th.log(est_action_dist[est_action]))

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
        BASELINE_OPTIM.step()
        # th.nn.utils.clip_grad.clip_grad_norm_(policy_net.parameters(), 0.3)
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
