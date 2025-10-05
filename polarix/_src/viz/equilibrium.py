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

"""Visualize an equilibrium solution for each player."""

from collections.abc import Mapping, Sequence

import altair as alt
import chex
import numpy as np
import pandas as pd
from polarix._src.games import base


_CHART_HEIGHT_PER_ACTION = 10
_CHART_WIDTH_PER_PLAYER = 30


def rating_and_marginal(
    game: base.Game,
    ratings: Sequence[chex.Array],
    marginals: Sequence[chex.Array],
    top_k: int | None = None,
    ratings_samples: chex.Array | None = None,
    metadata: Mapping[str, pd.DataFrame] | None = None,
    height: int | None = None,
    width: int | None = None,
) -> alt.Chart:
  """Plots ratings and marginals for each player.

  Args:
    game: The game object.
    ratings: The ratings of the actions of the players.
    marginals: The marginals of the actions of the players.
    top_k: The number of actions to show for each player.
    ratings_samples: The samples of the ratings of the players.
    metadata: If specified, a mapping from player name to metadata dataframe for
      that player.
    height: The height of the chart.
    width: The width of the chart.

  Returns:
    The chart of ratings
  """
  if len(game.actions) != len(ratings) != len(marginals):
    raise ValueError(
        "The lengths of game.actions, ratings, and marginals must be equal"
        " to the number of players."
    )
  if ratings_samples is not None and len(ratings_samples) != len(ratings):
    raise ValueError(
        "The lengths of ratings and ratings_samples must be equal."
    )

  if metadata is None:
    metadata = {}

  for k, data in metadata.items():
    if k not in game.players:
      raise ValueError(
          f"The metadata has been specified for non-existent player {k}. "
          f"Available players are {game.players}."
      )
    if len(data) != len(data[k].unique()):
      raise ValueError(f"Column {k} of metadata[{k}] is not all unique.")

  if top_k is None:
    top_k = max([len(a_p) for a_p in game.actions])

  if height is None:
    num_actions = min(max([len(a_p) for a_p in game.actions]), top_k)
    height = _CHART_HEIGHT_PER_ACTION * num_actions
  if width is None:
    width = _CHART_WIDTH_PER_PLAYER * len(game.players)

  charts = []
  for p, (a_p, r_p, m_p) in enumerate(zip(game.actions, ratings, marginals)):
    chex.assert_equal_shape((a_p, r_p, m_p))
    player = game.players[p]
    pd_dict = {player: a_p, "ratings": r_p, "marginals": m_p}
    if ratings_samples is not None:
      pd_dict["ratings_stddev"] = np.std(ratings_samples[p], axis=0)
    data = pd.DataFrame(pd_dict)

    if player in metadata:
      data = pd.merge(
          data,
          metadata[player],
          on=player,
          how="left",
          suffixes=("", "_metadata"),
          validate="many_to_one",
      )

    top_k_actions_by_ratings = a_p[np.argsort(-r_p)[:top_k]]
    num_actions = len(top_k_actions_by_ratings)
    ratings_chart = (
        alt.Chart(data[data[player].isin(top_k_actions_by_ratings)])
        .mark_point(color="red", shape="diamond")
        .encode(
            y=alt.Y(
                f"{player}:N",
                # Setting op is required for the sort to work. See CL/707524538.
                sort=alt.EncodingSortField(
                    field="ratings", op="min", order="descending"
                ),
            ),
            x=alt.X("ratings:Q", title="Equilibrium ratings"),
            tooltip=alt.Tooltip(sorted(data.columns.to_list())),
        )
        .properties(height=height, width=width)
    )
    if ratings_samples is not None:
      ratings_ci_chart = (
          ratings_chart.mark_errorbar(
              color="darkblue", ticks=True, size=0.3 * height // num_actions
          )
          .encode(
              x=alt.X("ratings_lo:Q"),
              x2=alt.X2("ratings_hi:Q"),
              strokeWidth=alt.value(1),
          )
          .transform_calculate(
              ratings_lo="datum.ratings - datum.ratings_stddev",
              ratings_hi="datum.ratings + datum.ratings_stddev",
          )
          .properties(height=height, width=width)
      )
      ratings_chart += ratings_ci_chart

    top_k_actions_by_marginals = a_p[np.argsort(-m_p)[:top_k]]
    marginals_chart = (
        alt.Chart(data[data[player].isin(top_k_actions_by_marginals)])
        .mark_bar()
        .encode(
            y=alt.Y(f"{player}:N", sort="-x"),
            x=alt.X("marginals:Q", title="Marginal support"),
            tooltip=alt.Tooltip(sorted(data.columns.to_list())),
        )
        .properties(height=height, width=width)
    )

    charts.append(
        alt.vconcat(ratings_chart, marginals_chart)
        .resolve_scale(x="independent")
        .properties(
            title=alt.TitleParams(
                f"{game.players[p].capitalize()} player", anchor="middle"
            )
        )
    )
  return alt.hconcat(*charts)
