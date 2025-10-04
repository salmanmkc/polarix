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
import numpy as np
import pandas as pd
from polarix._src.games import test_utils as test_utils_games
from polarix._src.viz import equilibrium
from polarix._src.viz import test_utils


METADATA = pd.DataFrame.from_dict({
    "p1": pd.Series(
        ["rock1", "rock2", "paper", "scissors"], index=np.arange(4)
    ),
    "action_type": pd.Series(
        ["rock", "rock", "paper", "scissors"], index=np.arange(4)
    ),
})


_EXPECTED_RATING_AND_MARGINAL = "rating_and_marginal.json"


class EquilibriumTest(test_utils.JsonAlmostEqualTestCase):

  def test_plot_ratings_and_marginals(self):
    rrps_game = test_utils_games.make_rrps()
    na = len(rrps_game.actions[0])
    ratings = [np.arange(na, dtype=np.float32)] * 2
    marginals = [np.ones(na, dtype=np.float32) / na] * 2
    chart = equilibrium.rating_and_marginal(
        rrps_game,
        ratings=ratings,
        marginals=marginals,
        metadata={"p1": METADATA},
    )
    self.assertChartEqual(chart, _EXPECTED_RATING_AND_MARGINAL)


if __name__ == "__main__":
  absltest.main()
