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
from polarix._src.games import base
from polarix._src.games import test_utils as test_utils_games
from polarix._src.viz import marginal_contrib
from polarix._src.viz import test_utils


EXPECTED_RRPS = "rating_contribution.json"
EXPECTED_RRPS_BOTTOM_K = "rating_contribution_bottom_k.json"
EXPECTED_RRPS_CATEGORISED = "rating_contribution_categorised.json"
EXPECTED_RRPS_JOINT_CATEGORISED = "rating_contribution_joint_categorised.json"
EXPECTED_RRPS_HREF = "rating_contribution_href.json"

# METADATA associated with each contributor action.
METADATA = pd.DataFrame.from_dict({
    "p1": pd.Series(
        ["rock1", "rock2", "paper", "scissors"], index=np.arange(4)
    ),
    "p2": pd.Series(
        ["rock1", "rock2", "paper", "scissors"], index=np.arange(4)
    ),
    "action_type": pd.Series(
        ["rock", "rock", "paper", "scissors"], index=np.arange(4)
    ),
    "rating_color": pd.Series(
        ["orange", "orange", "purple", "green"], index=np.arange(4)
    ),
})

BAD_METADATA = pd.DataFrame.from_dict({
    "p1": pd.Series(
        ["rock1", "rock2", "paper"], index=np.arange(3)
    ),
    "p2": pd.Series(
        ["rock1", "rock2", "paper"], index=np.arange(3)
    ),
})

# METADATA associated with each (rating, contributor) action pair.
JOINT_METADATA = pd.DataFrame.from_dict({
    "p1": pd.Series(
        [
            "rock1",
            "rock1",
            "rock1",
            "rock1",
            "rock2",
            "rock2",
            "rock2",
            "rock2",
            "paper",
            "paper",
            "paper",
            "paper",
            "scissors",
            "scissors",
            "scissors",
            "scissors",
        ],
        index=np.arange(16),
    ),
    "p2": pd.Series(
        [
            "rock1",
            "rock2",
            "paper",
            "scissors",
            "rock1",
            "rock2",
            "paper",
            "scissors",
            "rock1",
            "rock2",
            "paper",
            "scissors",
            "rock1",
            "rock2",
            "paper",
            "scissors",
        ],
        index=np.arange(16),
    ),
    "joint_action_type": pd.Series(
        [
            "rock_rock",
            "rock_rock",
            "rock_paper",
            "rock_scissors",
            "rock_rock",
            "rock_rock",
            "rock_paper",
            "rock_scissors",
            "paper_rock",
            "paper_rock",
            "paper_paper",
            "paper_scissors",
            "scissors_rock",
            "scissors_rock",
            "scissors_paper",
            "scissors_scissors",
        ],
        index=np.arange(16),
    ),
})


class MarginalContributionTest(test_utils.JsonAlmostEqualTestCase):

  def test_rrps(self):
    rrps_game = test_utils_games.make_rrps()
    na = len(rrps_game.actions[0])
    marginals = [np.ones(na, np.float32) / na, np.ones(na, np.float32) / na]
    joint = base.joint_from_marginals(marginals)

    chart = marginal_contrib.rating_contribution(
        rrps_game, joint, rating_player=0, contrib_player=1
    )
    self.assertChartEqual(chart, EXPECTED_RRPS)

  def test_bottom_k(self):
    rrps_game = test_utils_games.make_rrps()
    na = len(rrps_game.actions[0])
    marginals = [np.ones(na, np.float32) / na, np.ones(na, np.float32) / na]
    joint = base.joint_from_marginals(marginals)

    chart = marginal_contrib.rating_contribution(
        rrps_game, joint, rating_player=0, contrib_player=1, top_k=1, bottom_k=1
    )
    self.assertChartEqual(chart, EXPECTED_RRPS_BOTTOM_K)

  def test_missing_rating_metadata_raises_warning(self):
    rrps_game = test_utils_games.make_rrps()
    na = len(rrps_game.actions[0])
    marginals = [np.ones(na, np.float32) / na, np.ones(na, np.float32) / na]
    joint = base.joint_from_marginals(marginals)

    with self.subTest("missing_rating_metadata"):
      with self.assertLogs(level="WARNING") as cm:
        marginal_contrib.rating_contribution(
            rrps_game,
            joint,
            rating_player=0,
            contrib_player=1,
            rating_metadata=BAD_METADATA,
            contrib_metadata=METADATA,
        )
        self.assertRegex(cm.output[0], "should contain game.actions")

  def test_missing_contrib_metadata_raises_error(self):
    rrps_game = test_utils_games.make_rrps()
    na = len(rrps_game.actions[0])
    marginals = [np.ones(na, np.float32) / na, np.ones(na, np.float32) / na]
    joint = base.joint_from_marginals(marginals)

    with self.subTest("missing_contrib_metadata"):
      with self.assertRaisesRegex(ValueError, "must contain game.actions"):
        marginal_contrib.rating_contribution(
            rrps_game,
            joint,
            rating_player=0,
            contrib_player=1,
            rating_metadata=METADATA,
            contrib_metadata=BAD_METADATA,
        )

  def test_contrib_categories(self):
    rrps_game = test_utils_games.make_rrps()
    na = len(rrps_game.actions[0])
    marginals = [np.ones(na, np.float32) / na, np.ones(na, np.float32) / na]
    joint = base.joint_from_marginals(marginals)

    drop_columns = ["p1", "rating_color"]

    with self.subTest("missing_category"):
      with self.assertRaisesRegex(ValueError, "must contain column"):
        marginal_contrib.rating_contribution(
            rrps_game,
            joint,
            rating_player=0,
            contrib_player=1,
            contrib_metadata=METADATA.drop(drop_columns, axis=1),
            contrib_categories=("missing_category",),
        )

    chart = marginal_contrib.rating_contribution(
        rrps_game,
        joint,
        rating_player=0,
        contrib_player=1,
        contrib_metadata=METADATA.drop(drop_columns, axis=1),
        contrib_categories=("action_type",),
    )
    self.assertChartEqual(chart, EXPECTED_RRPS_CATEGORISED)

  def test_joint_contrib_categories(self):
    rrps_game = test_utils_games.make_rrps()
    na = len(rrps_game.actions[0])
    marginals = [np.ones(na, np.float32) / na, np.ones(na, np.float32) / na]
    joint = base.joint_from_marginals(marginals)

    chart = marginal_contrib.rating_contribution(
        rrps_game,
        joint,
        rating_player=0,
        contrib_player=1,
        contrib_metadata=JOINT_METADATA,
        contrib_categories=("joint_action_type",),
    )
    self.assertChartEqual(chart, EXPECTED_RRPS_JOINT_CATEGORISED)

  def test_custom_encodings_and_styles(self):
    rrps_game = test_utils_games.make_rrps()
    na = len(rrps_game.actions[0])
    marginals = [np.ones(na, np.float32) / na, np.ones(na, np.float32) / na]
    joint = base.joint_from_marginals(marginals)

    chart = marginal_contrib.rating_contribution(
        rrps_game,
        joint,
        rating_player=1,
        contrib_player=0,
        rating_metadata=METADATA.drop(columns=["p1"]),
        rating_href="action_type",
        rating_color="rating_color",
        rating_styles={"stroke": "black", "filled": True, "size": 80},
        contrib_metadata=JOINT_METADATA,
        contrib_categories=("joint_action_type",),
        contrib_href="joint_action_type",
    )
    self.assertChartEqual(chart, EXPECTED_RRPS_HREF)


if __name__ == "__main__":
  absltest.main()
