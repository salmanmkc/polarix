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
import jax.numpy as jnp
import numpy as np
from polarix._src.games import normalize
from polarix._src.games import sxs


class SxSTest(absltest.TestCase):

  def test_diff_game(self):
    game = sxs.diff_game(
        loc=np.reshape(np.arange(24, dtype=np.float32), (2, 3, 4)),
        scale=np.reshape(np.arange(24, dtype=np.float32), (2, 3, 4)),
        actions=(jnp.arange(2), jnp.arange(3), jnp.arange(4)),
        players=("a", "b", "c"),
        utilities=(jnp.ones_like, jnp.zeros_like),
        normalizer=normalize.uvzm,
    )
    self.assertSequenceEqual(game.players, ("a", "b", "c", "c"))
    self.assertTrue(np.all(game.payoffs[0] == 1.0))
    self.assertTrue(np.all(game.payoffs[1] == 0.0))
    np.testing.assert_allclose(
        game.payoffs[2, 0, 0],
        np.asarray([
            [0.0, -0.894427, -1.788854, -2.683281],
            [0.894427, 0.0, -0.894427, -1.788854],
            [1.788854, 0.894427, 0.0, -0.894427],
            [2.683281, 1.788854, 0.894427, 0.0],
        ]),
        atol=1e-6,
    )

  def test_normal_game(self):
    game = sxs.winrate_game(
        loc=np.reshape(np.arange(24, dtype=np.float32), (2, 3, 4)),
        scale=np.reshape(np.arange(24, dtype=np.float32), (2, 3, 4)),
        actions=(jnp.arange(2), jnp.arange(3), jnp.arange(4)),
        players=("a", "b", "c"),
        utilities=(jnp.ones_like, jnp.zeros_like),
    )
    self.assertSequenceEqual(game.players, ("a", "b", "c", "c"))
    self.assertTrue(np.all(game.payoffs[0] == 1.0))
    self.assertTrue(np.all(game.payoffs[1] == 0.0))
    np.testing.assert_allclose(game.payoffs[2], 1 - game.payoffs[3], rtol=1e-6)
    np.testing.assert_allclose(
        game.payoffs[2], jnp.swapaxes(game.payoffs[3], -2, -1), rtol=1e-6
    )
    np.testing.assert_allclose(
        game.payoffs[2, 0, 0],
        np.asarray([
            [0.5, 0.15865527, 0.15865527, 0.15865527],
            [0.8413447, 0.5, 0.32736042, 0.26354462],
            [0.8413447, 0.6726396, 0.5, 0.39075565],
            [0.8413447, 0.7364554, 0.60924435, 0.5],
        ]),
        atol=1e-6,
    )


if __name__ == "__main__":
  absltest.main()
