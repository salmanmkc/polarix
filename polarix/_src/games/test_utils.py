# Copyright 2025 The polarix Authors.
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

"""Helper functions for testing solvers."""

import numpy as np
from polarix._src.games import base

RRPS = np.asarray([
    [0.0, 0.0, -1.0, 1.0],
    [0.0, 0.0, -1.0, 1.0],  # duplicate "rock" action
    [1.0, 1.0, 0.0, -1.0],
    [-1.0, -1.0, 1.0, 0.0],
])
RRPS_ACTIONS = np.asarray(['rock1', 'rock2', 'paper', 'scissors'])


def make_dominated() -> base.Game:
  return base.Game(
      payoffs=np.asarray([
          [
              [13.0, 1.0, 7.0],
              [4.0, 3.0, 6.0],
              [-1.0, 2.0, 8.0],
          ],
          [
              [3.0, 4.0, 3.0],
              [1.0, 3.0, 2.0],
              [9.0, 8.0, -1.0],
          ],
      ]),
      actions=(
          np.asarray(['up', 'middle', 'down']),
          np.asarray(['left', 'center', 'right']),
      ),
      players=('row', 'col'),
  )


def make_rrps() -> base.Game:
  return base.Game(
      payoffs=np.asarray([RRPS, -RRPS]),
      sample=lambda _: np.asarray([RRPS, -RRPS]),
      actions=(RRPS_ACTIONS, RRPS_ACTIONS),
      players=('p1', 'p2'),
      symmetry_groups=(0, 0),
  )


def make_chicken(duplicate: bool = True) -> base.Game:
  """Returns a version of the game of Chicken.

  The game has two players and three actions: Swerve, Straight, and a duplicate
  Straight. The payoffs are symmetric.

  The max-entropy NE is to go straight 1/12 of the time and swerve the rest.

  Args:
    duplicate: If True, includes a duplicate "Straight" action. Otherwise, only
      two actions ("Swerve", "Straight") are used.
  """
  payoffs = np.asarray([
      [0.0, -1.0, -1.0],
      [1.0, -12.0, -12.0],
      [1.0, -12.0, -12.0],  # duplicate "straight" action
  ])
  actions = np.asarray(['Swerve', 'Straight', 'Straight'])
  if not duplicate:
    actions = actions[:-1]
    payoffs = payoffs[:-1]
    payoffs = payoffs[:, :-1]
  return base.Game(
      payoffs=np.asarray([payoffs, payoffs.T]),
      sample=lambda _: np.asarray([payoffs, payoffs.T]),
      actions=(actions, actions),
      players=np.arange(2),
      symmetry_groups=(0, 0),
  )


def make_el_farol(n=2, c=0.5, b=0, s=1, g=2) -> base.Game:
  """Returns a game with the El Farol Stage game payoffs.

  See Section 3.1, The El Farol Stage Game in
  http://www.econ.ed.ac.uk/papers/id186_esedps.pdf

  action 0: go to bar
  action 1: avoid bar

  Args:
    n: int, number of players
    c: float, threshold for `crowded' as a fraction of number of players
    b: float, payoff for going to a crowded bar
    s: float, payoff for staying at home
    g: float, payoff for going to an uncrowded bar
  """
  payoffs = np.zeros((n,) + (2,) * n)
  for idx in np.ndindex(payoffs.shape):
    p = idx[0]
    a = idx[1:]
    a_i = a[p]
    go_to_bar = a_i < 1
    crowded = (n - 1 - sum(a) + a_i) >= (c * n)
    if go_to_bar and not crowded:
      payoffs[idx] = g
    elif go_to_bar and crowded:
      payoffs[idx] = b
    else:
      payoffs[idx] = s
  actions = np.asarray(['Attend Bar', 'Stay Home'])
  return base.Game(
      payoffs=payoffs,
      sample=lambda _: payoffs,
      actions=(actions,) * n,
      players=np.arange(n),
      symmetry_groups=(0,) * n,
  )
