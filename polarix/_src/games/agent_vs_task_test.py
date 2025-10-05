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

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from polarix._src.games import agent_vs_task


AGENT_VS_TASK_MEAN = np.asarray([
    [0.1, 0.7, 0.1, 0.9],
    [0.6, 0.7, 0.2, 0.5],
    [0.1, 0.9, 0.9, 0.2],
])

AGENT_VS_TASK_STDDEV = np.asarray([
    [0.1, 0.9, 0.1, 0.0],
    [0.6, 1.2, 0.1, 0.0],
    [0.1, 0.9, 0.1, 0.0],
])

EXPECTED_WINRATE_PAYOFFS = np.asarray(
    [
        [
            [[0.0, 0.59, 0.0], [0.59, 0.0, 0.59], [0.0, 0.59, 0.0]],
            [[0.0, 0.0, 0.12], [0.0, 0.0, 0.11], [0.12, 0.11, 0.0]],
            [[0.0, 0.52, 1.0], [0.52, 0.0, 1.0], [1.0, 1.0, 0.0]],
            [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
        ],
        [
            [[0.5, 0.21, 0.5], [0.79, 0.5, 0.79], [0.5, 0.21, 0.5]],
            [[0.5, 0.5, 0.44], [0.5, 0.5, 0.45], [0.56, 0.55, 0.5]],
            [[0.5, 0.24, 0.0], [0.76, 0.5, 0.0], [1.0, 1.0, 0.5]],
            [[0.5, 1.0, 1.0], [0.0, 0.5, 1.0], [0.0, 0.0, 0.5]],
        ],
        [
            [[0.5, 0.79, 0.5], [0.21, 0.5, 0.21], [0.5, 0.79, 0.5]],
            [[0.5, 0.5, 0.56], [0.5, 0.5, 0.55], [0.44, 0.45, 0.5]],
            [[0.5, 0.76, 1.0], [0.24, 0.5, 1.0], [0.0, 0.0, 0.5]],
            [[0.5, 0.0, 0.0], [1.0, 0.5, 0.0], [1.0, 1.0, 0.5]],
        ],
    ],
)


class GamesTest(parameterized.TestCase):

  def test_agent_vs_task_game(self):
    na, nt = 4, 8

    avt = np.arange(na * nt).reshape((na, nt)).astype(np.float32)
    avt_stddev = np.random.uniform(size=(na, nt), low=0.0, high=1.0).astype(
        np.float32
    )

    agents = np.arange(na)
    tasks = np.arange(nt)

    with self.subTest("raise_on_negative_stddev"):
      with self.assertRaisesRegex(ValueError, ".*non-negative.*"):
        agent_vs_task.agent_vs_task_game(
            agents=agents,
            tasks=tasks,
            agent_vs_task=avt,
            agent_vs_task_stddev=avt_stddev * -1.0,
        )

    game = agent_vs_task.agent_vs_task_game(
        agents=agents,
        tasks=tasks,
        agent_vs_task=avt,
        agent_vs_task_stddev=avt_stddev,
    )

    self.assertEqual(game.payoffs.shape, (3, nt, na, na))
    self.assertEqual((game.payoffs[0].min(), game.payoffs[0].max()), (0, 1))
    self.assertEqual((game.payoffs[1].min(), game.payoffs[1].max()), (-1, 1))
    self.assertEqual((game.payoffs[2].min(), game.payoffs[2].max()), (-1, 1))

  def test_winrate_agent_vs_task_game(self):
    na, nt = AGENT_VS_TASK_MEAN.shape
    agents = np.arange(na)
    tasks = np.arange(nt)

    with self.subTest("raise_on_negative_stddev"):
      with self.assertRaisesRegex(ValueError, ".*non-negative.*"):
        agent_vs_task.agent_vs_task_game(
            agents=agents,
            tasks=tasks,
            agent_vs_task=AGENT_VS_TASK_MEAN,
            agent_vs_task_stddev=AGENT_VS_TASK_STDDEV * -1.0,
            normalizer="winrate",
        )

    game = agent_vs_task.agent_vs_task_game(
        agents=agents,
        tasks=tasks,
        agent_vs_task=AGENT_VS_TASK_MEAN,
        agent_vs_task_stddev=AGENT_VS_TASK_STDDEV,
        normalizer="winrate",
    )

    np.testing.assert_allclose(
        game.payoffs, EXPECTED_WINRATE_PAYOFFS, atol=1e-2
    )


if __name__ == "__main__":
  absltest.main()
