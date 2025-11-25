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

"""Tabulates data into payoff tensor of different games."""

import chex
import jax.numpy as jnp
from polarix._src.games import base
from polarix._src.games import normalize
from polarix._src.games import sxs


def agent_vs_task_game(
    *,
    agents: chex.Array,
    tasks: chex.Array,
    agent_vs_task: chex.Array,
    agent_vs_task_stddev: chex.Array | None = None,
    agent_player: str = "agent",
    task_player: str = "task",
    normalizer: str = "ptp",
) -> base.Game:
  """Returns an evaluation game with item-level data.

  Args:
    agents: A list of agents action labels.
    tasks: A list of tasks action labels.
    agent_vs_task: An (agents, task) score matrix for each agent on each task.
    agent_vs_task_stddev: An (agents, tasks) stddev matrix for each agent on
      each task. If not provided, it is assumed to be 0.
    agent_player: The agent player name.
    task_player: The task player name.
    normalizer: Normalization strategy, one of "ptp", "uvzm", "rank", "winrate",
      or "none". If "none", no normalization is applied. If not provided, the
      score matrix is normalized by peak-to-peak across agents for each task.

  Returns:
    A Game instance.
  """
  if agent_vs_task_stddev is None:
    agent_vs_task_stddev = jnp.zeros_like(agent_vs_task)

  chex.assert_tree_all_finite((agent_vs_task, agent_vs_task_stddev))
  chex.assert_shape(
      (agent_vs_task, agent_vs_task_stddev), (len(agents), len(tasks))
  )

  if jnp.any(agent_vs_task_stddev < 0):
    raise ValueError("Agent vs task stddev must be non-negative.")

  if normalizer == "winrate":
    return sxs.winrate_game(
        loc=agent_vs_task.T,
        scale=agent_vs_task_stddev.T,
        actions=(tasks, agents),
        players=(task_player, agent_player),
        utilities=[lambda wr: jnp.abs(2.0 * wr - 1.0)],
        min_stddev=1e-6,
    )

  # Choose normalization method.
  match normalizer:
    case "none":
      normalizer = lambda x: x
    case "ptp":
      normalizer = normalize.ptp
    case "uvzm":
      normalizer = normalize.uvzm
    case "rank":
      normalizer = normalize.rank
    case _:
      raise ValueError(f"Unsupported normalizer: {normalizer}")

  return sxs.diff_game(
      loc=agent_vs_task.T,
      scale=agent_vs_task_stddev.T,
      actions=(tasks, agents),
      players=(task_player, agent_player),
      utilities=[jnp.abs],  # absolute score difference for the task player.
      normalizer=normalizer,
  )
