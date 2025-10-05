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
from polarix._src.games import base
from polarix._src.solvers import average


class AverageTest(parameterized.TestCase):

  @parameterized.parameters([
      dict(na=(2, 3, 5, 6)),
      dict(na=(4, 3, 2, 1)),
  ])
  def test_random_pt(self, na):
    npl = len(na)
    payoffs = np.random.rand(npl, *na)
    game = base.Game(
        payoffs=payoffs,
        actions=tuple([np.arange(nai) for nai in na]),
        players=("a", "b", "c", "d"),
    )
    res = average.average(game)
    self.assertLen(res.ratings, npl)
    self.assertLen(res.marginals, npl)
    for p in range(npl):
      self.assertTrue(np.all(np.isfinite(res.ratings[p])))
      self.assertTrue(np.all(np.isfinite(res.marginals[p])))


if __name__ == "__main__":
  absltest.main()
