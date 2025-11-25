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
import chex
import jax
import jax.numpy as jnp
import numpy as np
from polarix._src.games import base


class GamesTest(parameterized.TestCase):

  def test_payoffs_ndim(self):
    with self.assertRaisesRegex(ValueError, "one more dimension"):
      base.Game(
          payoffs=np.reshape(np.arange(6), (2, 3)),
          actions=(np.arange(2), np.arange(3)),
          players=("p1", "p2"),
      )

  def test_payoffs_shape(self):
    with self.assertRaisesRegex(
        ValueError,
        "Player 0 has 1 actions, but payoff shape implies 2 actions.",
    ):
      base.Game(
          payoffs=np.reshape(np.arange(12), (2, 2, 3)),
          actions=(np.arange(1), np.arange(3)),
          players=("p1", "p2"),
      )

  def test_payoffs_isna(self):
    with self.assertRaisesRegex(ValueError, "finite"):
      base.Game(
          payoffs=np.full((2, 2, 3), np.nan),
          actions=(np.arange(2), np.arange(3)),
          players=("p1", "p2"),
      )

  def test_actions_len(self):
    with self.assertRaisesRegex(
        ValueError,
        "Player 1 has 4 actions, but payoff shape implies 3 actions.",
    ):
      base.Game(
          payoffs=np.ones((2, 2, 3)),
          actions=(np.arange(2), np.arange(4)),
          players=("p1", "p2"),
      )

  def test_symmetry_group_len(self):
    with self.assertRaisesRegex(ValueError, "same length as actions"):
      base.Game(
          payoffs=np.ones((2, 2, 3)),
          actions=(np.arange(2), np.arange(3)),
          symmetry_groups=(0, 0, 0),
          players=("p1", "p2"),
      )

  def test_symmetry_group_same_num_actions(self):
    with self.assertRaisesRegex(
        ValueError, "Symmetry groups must have the same length as actions"
    ):
      base.Game(
          payoffs=np.reshape(np.arange(12), (2, 2, 3)),
          actions=(np.arange(2), np.arange(3)),
          symmetry_groups=(0, 0, 0),
          players=("p1", "p2"),
      )

  def test_symmetry_group_asymmetric_payoffs(self):
    with self.assertRaisesRegex(ValueError, "Payoffs are not symmetric"):
      base.Game(
          payoffs=np.reshape(np.arange(54), (3, 2, 3, 3)),
          actions=(np.arange(2), np.arange(3), np.arange(3)),
          symmetry_groups=(0, 1, 1),
          players=("p1", "p2", "p3"),
      )


class MarginalsJointTest(parameterized.TestCase):

  def test_marginals_from_joint(self):
    marginals = [
        np.random.dirichlet(np.ones(4, dtype=np.float32)),
        np.random.dirichlet(np.ones(6, dtype=np.float32)),
    ]

    joint = base.joint_from_marginals(marginals)
    np.testing.assert_almost_equal(joint.sum(), 1.0, decimal=5)

    np.testing.assert_allclose(joint, np.outer(*marginals), atol=1e-6)

    marginals_ = base.marginals_from_joint(joint)
    for actual, expected in zip(marginals_, marginals):
      np.testing.assert_allclose(actual, expected, atol=1e-6)


@functools.partial(jax.jit, static_argnums=(2, 3))
def joint_payoffs_contribution(
    payoffs: chex.Array,
    joint: chex.Array,
    rating_player: int,
    contrib_player: int,
) -> chex.Array:
  """Returns the contribution of one player to another player under the joint."""
  # Compute deviation gains for the rating player.
  # Shape: [    1,|S_1|,...,|S_p-1|,|S_p|,|S_p+1|,...,|S_N|]
  # The swap is changing shapes:
  # From:  [    1,|S_1|,...,|S_p-1|,|S_p|,|S_p+1|,...,|S_N|]
  # To:    [|S_p|,|S_1|,...,|S_p-1|,    1,|S_p+1|,...,|S_N|]
  rating_action_payoffs = jnp.swapaxes(
      jnp.expand_dims(payoffs[rating_player], 0), 0, rating_player + 1
  )
  not_contrib = tuple(p for p in range(payoffs.shape[0]) if p != contrib_player)

  def _contrib(action_payoffs, payoffs):
    # [|S_1|,...,|S_p-1|,|S_p|,|S_p+1|,...,|S_N|]
    expected_gains = (action_payoffs - payoffs) * joint
    return expected_gains.sum(not_contrib)

  return jax.vmap(_contrib, in_axes=(0, None))(
      rating_action_payoffs, payoffs[rating_player]
  )


class ContributionTest(absltest.TestCase):

  def test_rating_contribution_memory_usage(self):
    payoffs = np.random.uniform(size=(3, 1500, 150, 150))
    joint = np.random.uniform(size=(1500, 150, 150))
    rating_contrib = base.joint_payoffs_contribution(
        payoffs, joint, rating_player=1, contrib_player=0
    )
    self.assertEqual(rating_contrib.shape, (150, 1500))

  def test_rating_contribution_efficient(self):
    payoffs = np.random.uniform(size=(3, 20, 15, 15))
    joint = np.random.uniform(size=(20, 15, 15))
    joint /= np.sum(joint)
    actual = base.joint_payoffs_contribution(
        payoffs, joint, rating_player=1, contrib_player=0
    )
    expected = joint_payoffs_contribution(
        payoffs, joint, rating_player=1, contrib_player=0
    )
    np.testing.assert_allclose(actual, expected, 0.0, atol=1e-6)


if __name__ == "__main__":
  absltest.main()
