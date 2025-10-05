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

import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
from polarix._src.games import normalize


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


class NormalizeTest(parameterized.TestCase):

  def test_normalize_rank(self):
    avt = np.asarray([0, 1, 2, 3, 4])
    expected_rank = np.asarray([0, 1, 2, 3, 4]) / 4.0
    rank = normalize.rank(avt)
    np.testing.assert_allclose(rank, expected_rank)

    scores = np.asarray([0, 1, 1, 1, 4])
    expected_rank = np.asarray([0, 2, 2, 2, 4]) / 4.0
    rank = normalize.rank(scores)
    np.testing.assert_allclose(rank, expected_rank)

    scores = np.asarray([0, 1, 1, 3])
    expected_rank = np.asarray([0, 1.5, 1.5, 3]) / 3.0
    rank = normalize.rank(scores)
    np.testing.assert_allclose(rank, expected_rank)

  def test_normalize_uvzm(self):
    avt = np.asarray([0.0, 1, 2, 3, 4])
    avt_ = normalize.uvzm(avt)
    np.testing.assert_allclose(np.var(avt_), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.max(avt_), 0.0, atol=1e-6)

    avt = np.asarray([0.0, 1, 1, 1, 2, 2, 4])
    avt_ = normalize.uvzm(avt)
    np.testing.assert_allclose(np.var(avt_), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.max(avt_), 0.0, atol=1e-6)

    avt = np.asarray([0.0, 4, 5, 6, 9, 2.5])
    avt_ = normalize.uvzm(avt)
    np.testing.assert_allclose(np.var(avt_), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.max(avt_), 0.0, atol=1e-6)

  def test_normalize_ptp(self):
    agent_vs_task_score = jax.vmap(normalize.ptp, in_axes=1, out_axes=1)(
        AGENT_VS_TASK_MEAN
    )

    np.testing.assert_allclose(
        jax.vmap(normalize.rank, in_axes=1, out_axes=1)(AGENT_VS_TASK_MEAN),
        jax.vmap(normalize.rank, in_axes=1, out_axes=1)(agent_vs_task_score),
    )
    self.assertSequenceEqual(
        agent_vs_task_score.shape,
        AGENT_VS_TASK_MEAN.shape,
    )

  def test_normalize_mc_expected_rank(self):
    expected_rank = jax.vmap(
        functools.partial(normalize.expected_rank, key=jax.random.PRNGKey(0)),
        in_axes=1,
        out_axes=1,
    )(AGENT_VS_TASK_MEAN, np.zeros_like(AGENT_VS_TASK_STDDEV))
    rank = jax.vmap(normalize.rank, in_axes=1, out_axes=1)(AGENT_VS_TASK_MEAN)
    np.testing.assert_allclose(expected_rank, rank)

  def test_normalize_win(self):
    scores = np.asarray(
        [1.0, 0.5, 0.5, 0.1],
    )
    payoff = normalize.win(scores)
    self.assertSequenceEqual(payoff.shape, (4, 4))
    np.testing.assert_allclose(payoff + payoff.T, 1.0)
    np.testing.assert_allclose(np.diag(payoff), 0.5)

  def test_normalize_winrate(self):
    scores = np.asarray(
        [1.0, 0.5, 0.5, 0.1],
    )
    scores_stddev = np.asarray(
        [1.0, 1.0, 1.0, 2.0],
    )
    payoff = normalize.winrate(scores, scores_stddev)
    self.assertSequenceEqual(payoff.shape, (4, 4))
    np.testing.assert_allclose(payoff + payoff.T, 1.0)
    np.testing.assert_allclose(np.diag(payoff), 0.5)

  def test_normalize_beta(self):
    # 3 agents, 4 score samples per agent.
    score_samples = np.asarray([
        [1.0, 0.5, 0.5, 0.1],
        [0.0, 0.6, 0.3, 0.2],
        [0.0, 0.0, 0.0, 0.0],
    ])
    alpha, beta = normalize.beta(score_samples)
    self.assertSequenceEqual(alpha.shape, (3, 3))
    self.assertSequenceEqual(beta.shape, (3, 3))


if __name__ == "__main__":
  absltest.main()
