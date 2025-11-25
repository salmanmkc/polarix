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

"""Returns marginal contribution analysis for a given game and ranking."""

from typing import Any, Mapping, Sequence

from absl import logging
import altair as alt
import chex
import frozendict
import jax.numpy as jnp
import numpy as np
import pandas as pd
from polarix._src.games import base


_RATING_CHART_WIDTH = 100
_RATING_CHART_PER_ACTION_HEIGHT = 20

_CATEGORY_PANEL_WIDTH = 400

MAX_NUM_ROWS = 5_000
LONG_TAIL = "[LONG-TAIL CONTRIBUTORS]"


_COMPAT_V4 = alt.__version__.startswith("4")


def _rating_contribution_dataframe(
    game: base.Game,
    joint: chex.Array,
    rating_player: int,
    contrib_player: int,
):
  """Returns a dataframe of the rating contribution by contributor actions.

  Args:
    game: The game object.
    joint: The joint distribution of the game.
    rating_player: The player whose ratings are being plotted.
    contrib_player: The player whose contribution to the ratings of the rating
      player is being plotted.

  Returns:
    A dataframe of the rating contribution of a player to another player's
    ratings.
  """
  if _COMPAT_V4:
    logging.info("Enabled altair v4 compatibility mode.")

  contrib_to_ratings = base.joint_payoffs_contribution(
      payoffs=game.payoffs,
      joint=joint,
      rating_player=rating_player,
      contrib_player=contrib_player,
  )  # [num_rating_actions, num_contrib_actions]
  joint_support = jnp.sum(
      jnp.moveaxis(joint, (rating_player, contrib_player), (0, 1)),
      axis=tuple(range(2, len(joint.shape))),
  )  # [num_rating_actions, num_contrib_actions]
  rating_actions, contrib_actions = np.meshgrid(
      jnp.arange(len(game.actions[rating_player])),
      jnp.arange(len(game.actions[contrib_player])),
      indexing="ij",
  )
  return pd.DataFrame.from_dict({
      game.players[rating_player]: rating_actions.ravel(),
      game.players[contrib_player]: contrib_actions.ravel(),
      "contrib": contrib_to_ratings.ravel(),
      "support": joint_support.ravel(),
  })


_DEFAULT_RATING_STYLE = frozendict.frozendict(
    {"shape": "diamond", "color": "red", "filled": False}
)
_DEFAULT_CONTRIB_STYLE = frozendict.frozendict({"stroke": "black"})


def rating_contribution(
    game: base.Game,
    joint: chex.Array,
    *,
    rating_player: int,
    contrib_player: int,
    rating_metadata: pd.DataFrame | None = None,
    contrib_metadata: pd.DataFrame | None = None,
    contrib_categories: Sequence[str] = (),
    rating_href: str | None = None,
    rating_color: str | None = None,
    rating_styles: Mapping[str, Any] | None = _DEFAULT_RATING_STYLE,
    rating_ascending: bool = False,
    contrib_href: str | None = None,
    contrib_styles: Mapping[str, Any] | None = _DEFAULT_CONTRIB_STYLE,
    contrib_tooltip: Sequence[str] | None = None,
    top_k: int = 100,
    bottom_k: int = 0,
    use_categorical_contrib: bool = False,
    include_rating_plot: bool = True,
    rating_chart_width: int = _RATING_CHART_WIDTH,
    category_panel_width: int = _CATEGORY_PANEL_WIDTH,
    max_num_rows: int = MAX_NUM_ROWS,
) -> alt.Chart:
  """Plots the rating contribution of a player to another player's ratings.

  Args:
    game: The game object.
    joint: The joint distribution of the game.
    rating_player: The player whose action ratings are being plotted.
    contrib_player: The player whose action contribution to the ratings of the
      rating player actions is being plotted.
    rating_metadata: A dataframe with metadata for rating actions.
    contrib_metadata: A dataframe with metadata for contributor actions or for
      each (rating, contributor) action pairs. Metadata will be shown as the
      tooltips and may contain action categories information.
    contrib_categories: A sequence of action categories to be used to group
      action contributions. Action categories must correspond to columns of
      `contrib_metadata`. Categories must form a top-to-bottom tree structure,
      with each action belonging to a single `contrib_categories[-1]`, and each
      `contrib_category[l]` belonging to a single `contrib_category[l-1]`.
    rating_href: if set, the href to open upon clicking each rating action. Must
      be a column of `rating_metadata` if set.
    rating_color: if set, the literal color to use for rating diamonds. Must be
      a column of `rating_metadata` if set.
    rating_styles: A dictionary of altair `mark_point` style parameters.
    rating_ascending: If True, the rating actions are sorted in ascending order.
      Otherwise, they are sorted in descending order.
    contrib_href: if set, the href to open on clicking each contributor action.
      Must be a column of `contrib_metadata` if set.
    contrib_styles: A dictionary of altair `mark_bar` style parameters.
    contrib_tooltip: A sequence of columns to show as tooltip on hover. The
      columns options consist of the merge of game_metadata, rating_metadata,
      contrib_metadata, and special columns `contrib`, `support` and `rating`.
      If they are not included already, rating_name and contrib_name from the
      game players will be prepended.
    top_k: The number of top rating actions to show.
    bottom_k: The number of bottom rating actions to show.
    use_categorical_contrib: When True, the contrib is labelled categorically in
      a legend. When False, the contrib is colored quantitatively with a
      colorbar.
    include_rating_plot: When True, a rating plot is included.
    rating_chart_width: The width of the rating chart.
    category_panel_width: The width of each category panel.
    max_num_rows: The maximum number of elements in the graph to show. This is
      to avoid overloading the browser renderer.

  Returns:
    An Altair chart object.
  """
  if game.players is None:
    raise ValueError("Game must have player names explicitly defined.")

  if max_num_rows > MAX_NUM_ROWS:
    alt.data_transformers.disable_max_rows()

  if not isinstance(contrib_categories, (tuple, list)):
    raise ValueError(
        "`contrib_categories` must be a `Sequence` but is"
        f" {type(contrib_categories)}."
    )

  if contrib_categories and contrib_metadata is None:
    raise ValueError(
        "`contrib_metadata` must be provided if `contrib_categories` is"
        " provided."
    )

  if rating_href is not None:
    if rating_metadata is None:
      raise ValueError(
          "`rating_metadata` must be provided if `rating_href` is set."
      )
    if rating_href not in rating_metadata.columns:
      raise ValueError(
          f"`rating_href` ({rating_href}) must be a column of `rating_metadata`"
          f" ({rating_metadata.columns}) if `rating_href` is set."
      )

  if rating_color is not None:
    if rating_metadata is None:
      raise ValueError(
          "`rating_metadata` must be provided if `rating_color` is set."
      )
    if rating_color not in rating_metadata.columns:
      raise ValueError(
          f"`rating_color` ({rating_color}) must be a column of"
          f" `rating_metadata` ({rating_metadata.columns}) if `rating_color` is"
          " set."
      )

  if contrib_href is not None:
    if contrib_metadata is None:
      raise ValueError(
          "`contrib_metadata` must be provided if `contrib_href` is set."
      )
    if contrib_href not in contrib_metadata.columns:
      raise ValueError(
          f"`contrib_href` ({contrib_href}) must be a column of"
          f" `contrib_metadata` ({contrib_metadata.columns}) if `contrib_href`"
          " is set."
      )

  for category in contrib_categories:
    if category not in contrib_metadata.columns:
      raise ValueError(
          f"`contrib_metadata` must contain column `{category}` (declared in"
          f" {contrib_categories}) but columns are {contrib_metadata.columns}."
      )

  rating_name = game.players[rating_player]
  contrib_name = game.players[contrib_player]
  rating_actions = game.actions[rating_player]
  contrib_actions = game.actions[contrib_player]

  if rating_metadata is not None:
    nunique_by_rating_name = rating_metadata.groupby(rating_name).nunique(False)
    if np.any(nunique_by_rating_name != 1):
      raise ValueError(
          "Rating metadata must be unique per rating action, but is not"
          f" ({nunique_by_rating_name.reset_index()})."
      )

    rating_actions_set = set(game.actions[rating_player].tolist())
    missing_actions = rating_actions_set - set(rating_metadata[rating_name])
    if missing_actions:
      logging.warning(
          "rating_metadata should contain game.actions[rating_player]. Missing"
          " actions: %s", missing_actions
      )

  if contrib_metadata is not None:
    contrib_actions_set = set(game.actions[contrib_player].tolist())
    missing_actions = contrib_actions_set - set(contrib_metadata[contrib_name])
    if missing_actions:
      raise ValueError(
          "contrib_metadata must contain game.actions[contrib_player]. Missing"
          f" actions: {missing_actions}"
      )

  if contrib_tooltip is not None:
    contrib_tooltip = list(contrib_tooltip)
    if contrib_name not in contrib_tooltip:
      contrib_tooltip.insert(0, contrib_name)
    if rating_name not in contrib_tooltip:
      contrib_tooltip.insert(0, rating_name)

  data = _rating_contribution_dataframe(
      game=game,
      joint=joint,
      rating_player=rating_player,
      contrib_player=contrib_player,
  )

  # Computes rating player's action ratings to order rows by.
  sorted_rating_actions = (
      (data[[rating_name, "contrib"]].groupby(rating_name).sum().reset_index())
      .sort_values(by="contrib", ascending=rating_ascending)[rating_name]
      .values
  )

  # Computes the height of the rating chart.
  num_actions = len(rating_actions)
  if (top_k + bottom_k) > num_actions:
    # If the top_k + bottom_k is larger than the number of actions, then show
    # all actions from top to bottom.
    top_k, bottom_k = num_actions, 0
  num_actions_to_display = min(num_actions, top_k + bottom_k)
  height = num_actions_to_display * _RATING_CHART_PER_ACTION_HEIGHT

  sorted_rating_actions = np.concatenate(
      [
          sorted_rating_actions[:top_k],
          sorted_rating_actions[num_actions - bottom_k :],
      ],
      axis=0,
  )
  data = data[data[rating_name].isin(sorted_rating_actions)]
  data[rating_name] = data[rating_name].apply(lambda a: rating_actions[a])
  data[contrib_name] = data[contrib_name].apply(lambda a: contrib_actions[a])
  sorted_rating_actions = [rating_actions[a] for a in sorted_rating_actions]

  nrows = 0
  charts = []

  # Interval for selecting a subset of rating actions.
  ratings_data = (
      data[[rating_name, "contrib", "support"]]
      .groupby(rating_name)
      .sum()
      .reset_index()
  )
  if rating_metadata is not None:
    ratings_data = pd.merge(
        ratings_data,
        rating_metadata,
        on=rating_name,
        how="left",
        suffixes=(None, "_metadata"),
        validate="many_to_one",
    )
  ratings_data = ratings_data.rename(columns={"contrib": "rating"})
  ratings_chart = alt.Chart(ratings_data)
  nrows += len(ratings_chart.data)

  # Additional encoding channels for the rating points.
  points_encodings = {}
  if rating_href is not None:
    points_encodings["href"] = alt.Href(rating_href)
  if rating_color is not None:
    points_encodings["color"] = alt.Color(f"{rating_color}:N", scale=None)
  points_encodings["tooltip"] = alt.Tooltip(ratings_data.columns.to_list())

  sidebar = []

  points = ratings_chart.mark_point(**rating_styles).encode(
      y=alt.Y(
          f"{rating_name}:N",
          sort=sorted_rating_actions,
          title=None,
          axis=alt.Axis(labels=True, grid=True),
      ),
      x=alt.X(
          "sum(rating):Q",
          title=(
              f"{rating_name} ratings" if include_rating_plot else alt.Undefined
          ),
          axis=alt.Axis(grid=True, orient="top"),
      ),
      **points_encodings,
  )
  sidebar.append(points)

  top_bottom_text = None
  if top_k > 0 and bottom_k > 0:
    top_bottom_text = (
        alt.Chart(
            pd.DataFrame({
                f"{rating_name}": sorted_rating_actions[top_k - 1 : top_k + 1],
                "annotation": [f"⬆ top-{top_k}", f"⬇ bottom-{bottom_k}"],
            })
        )
        .mark_text(align="left", baseline="middle", dx=5)
        .encode(
            x=alt.value(0),
            y=alt.Y(f"{rating_name}:N", sort=sorted_rating_actions),
            text="annotation",
            color=alt.value("#000000"),
        )
    )
    sidebar.append(top_bottom_text)

  interval = None
  if include_rating_plot:
    interval = alt.selection_interval(encodings=["y"])

    line = ratings_chart.mark_line(strokeDash=[2, 2]).encode(
        y=alt.Y(
            f"{rating_name}:N",
            sort=sorted_rating_actions,
            title=None,
            axis=alt.Axis(labels=True, grid=True),
        ),
        x=alt.X(
            "support:Q",
            title=f"{rating_name} support",
            axis=alt.Axis(grid=True, orient="bottom"),
            scale=alt.Scale(zero=False),
        ),
    )
    sidebar.append(line)

    if top_bottom_text is not None:
      top_bottom_text = top_bottom_text.transform_filter(interval)

    sidebar = (
        alt.layer(*sidebar)
        .resolve_scale(x="independent", y="shared")
        .properties(width=rating_chart_width, height=height)
    )
    if _COMPAT_V4:
      sidebar = sidebar.add_selection(interval)
    else:
      sidebar = sidebar.add_params(interval)
    charts.append(sidebar)

  category_selectors = []
  categories = tuple(contrib_categories) + (contrib_name,)
  for i, category in enumerate(categories):
    category_data = data

    if contrib_metadata is not None:
      merge_on = contrib_name
      if rating_name in contrib_metadata.columns:
        merge_on = (rating_name, contrib_name)
      category_data = pd.merge(
          category_data,
          contrib_metadata,
          on=merge_on,
          how="left",
          suffixes=(None, "_metadata"),
          validate="many_to_one",
      )

    # Group by categories to aggregate over, and retain top-k elements. The
    # remaining long-tail elements are consolidated into one with non-category
    # columns overwritten.
    if i < len(categories) - 1 or use_categorical_contrib:
      # Aggregating over **categories** of contributor actions.
      color = alt.Color(
          f"{category}:N",
          scale=alt.Scale(scheme="spectral"),
          legend=alt.Legend(
              title=category.capitalize(),
              columns=1,
              # Make smaller than width to fit in the panel.
              labelLimit=int(category_panel_width * 0.9),
          ),
      )
      y_labels = False
      grouping = [rating_name, *categories[: i + 1]]

      if _COMPAT_V4:
        selector = alt.selection_multi(encodings=["color"])
      else:
        selector = alt.selection_point(encodings=["color"])

      # Highlight bars with matching contribution category.
      hoverlight_fields = list(grouping)
      hoverlight_fields.remove(rating_name)
    else:
      # Aggregating over **individual** contributor actions.
      color = alt.Color(
          "contrib:Q",
          scale=alt.Scale(scheme="redyellowgreen", domainMid=0.0),
          legend=alt.Legend(title="Contribution"),
      )
      y_labels = True
      grouping = list(
          filter(
              lambda c: c not in ["contrib", "support"], category_data.columns
          )
      )
      selector = None

      # Highlight bars with matching contribution action.
      hoverlight_fields = [contrib_name]

    if _COMPAT_V4:
      highlight = alt.selection_multi(
          fields=hoverlight_fields, on="mouseover", empty="none"
      )
    else:
      highlight = alt.selection_point(
          fields=hoverlight_fields, on="mouseover", empty=False
      )

    # Adjust bars strokeWidth with actively hovered-over contribution category.
    stroke_width = alt.condition(highlight, alt.value(2), alt.value(0))

    x = alt.X(
        "sum(contrib):Q",
        title=f"Contribution by {category} to {rating_name} ratings",
        axis=alt.Axis(grid=False, orient="top"),
    )
    y = alt.Y(
        f"{rating_name}:N",
        sort=sorted_rating_actions,
        title=None,
        scale=alt.Scale(domain=alt.Undefined if interval is None else interval),
        axis=alt.Axis(
            grid=True,
            labels=y_labels,
            labelLimit=300,
            orient="right",
        ),
    )
    category_data = (
        category_data[[*grouping, "contrib"]]
        .groupby(grouping)
        .sum()
        .reset_index()
    )
    absolute_contrib = category_data.contrib.abs()
    top_k = max_num_rows - nrows
    if top_k < len(absolute_contrib):
      # If there are more data rows than the maximum number of rows allowed,
      # then group the remaining data into a single "long-tail" row.
      cutoff = absolute_contrib.nlargest(top_k).iloc[-1].item()
      overwrite_columns = [
          c
          for c in grouping
          if c not in contrib_categories and c != rating_name
      ]
      # Ensure that the columns are strings and can be overwritten by LONG_TAIL.
      category_data[overwrite_columns] = category_data[
          overwrite_columns
      ].astype(str)
      category_data.loc[
          (absolute_contrib <= cutoff) | (absolute_contrib == 0.0),
          overwrite_columns,
      ] = LONG_TAIL

    category_data = (
        category_data[[*grouping, "contrib"]]
        .groupby(grouping)
        .sum()
        .reset_index()
    )

    nrows += len(category_data)

    category_chart = alt.Chart(category_data)
    if contrib_tooltip is None:
      tooltip = category_data.columns.to_list()
    else:
      tooltip = contrib_tooltip
    bars = category_chart.mark_bar(**contrib_styles).encode(
        x=x,
        y=y,
        color=color,
        opacity=(
            alt.condition(selector, alt.value(1.0), alt.value(0.25))
            if selector is not None
            else alt.value(1.0)
        ),
        # Stroke width is not strictly incl. in the width of the bars; when
        # stacking bars, stroke will overlap with neighbouring bars.
        strokeWidth=stroke_width,
        tooltip=alt.Tooltip(tooltip),
        href=(
            contrib_href
            if selector is None and contrib_href is not None
            else alt.Href()
        ),
    )
    if interval is not None:
      bars = bars.transform_filter(interval)

    # Order pos and neg contributions separately to address vega/altair bug.
    bars = alt.layer(
        bars.transform_filter(alt.datum.contrib < 0).encode(
            order=alt.Order("contrib:Q", sort="descending"),
        ),
        bars.transform_filter(alt.datum.contrib >= 0).encode(
            order=alt.Order("contrib:Q", sort="ascending"),
        ),
    ).resolve_scale(color="shared")
    rule = (  # Add a vertical rule at the 0-contribution point.
        alt.Chart(pd.DataFrame({"x": [1e-4]}))
        .mark_rule(opacity=0.5, size=1, strokeDash=[2, 2])
        .encode(x="x:Q")
    )
    if selector is not None:
      if _COMPAT_V4:
        bars = bars.add_selection(selector)
      else:
        bars = bars.add_params(selector)

    if _COMPAT_V4:
      bars = bars.add_selection(highlight)
    else:
      bars = bars.add_params(highlight)

    # For each contribution category panel, we overlay the rating points which
    # sums over all contributors, filtered by prior category selections. At the
    # same time, we wish to show rating action details (rating, <metadata>)
    # as tooltips.
    #
    # NOTE: We cannot reuse ratings side bar points here because `ratings_data`
    # does not contain the `category` columns which are needed for filtering by
    # selected category.
    #
    # NOTE: We reuse tooltip columns from `points_encodings` because
    # `overlay_ratings_data` contains contributor specific columns whereas
    # `overlay_rating_points` tooltips should be specific to each `rating_name`.
    overlay_ratings_data = pd.merge(
        ratings_data,
        category_data[[rating_name, *categories[: i + 1], "contrib"]],
        on=rating_name,
        how="left",
        suffixes=(None, "_category_data"),
        validate="one_to_many",
    )
    overlay_points = (
        alt.Chart(overlay_ratings_data)
        .mark_point(**rating_styles)
        .encode(
            y=alt.Y(
                f"{rating_name}:N",
                sort=sorted_rating_actions,
                title=None,
                axis=alt.Axis(labels=True, grid=True),
            ),
            x=alt.X(
                "sum(contrib):Q",
                title=f"{rating_name} ratings",
                axis=alt.Axis(grid=True, orient="top"),
            ),
            **points_encodings,
        )
    )

    if interval is not None:
      overlay_points = overlay_points.transform_filter(interval)

    for category_selector in category_selectors:
      bars = bars.transform_filter(category_selector)
      overlay_points = overlay_points.transform_filter(category_selector)

    if selector is not None:
      category_selectors.append(selector)

    category_panel = [bars, rule, overlay_points]
    if top_bottom_text is not None:
      category_panel.append(top_bottom_text)

    charts.append(
        alt.layer(*category_panel)
        .resolve_scale(x="shared", y="shared", color="independent")
        .properties(width=category_panel_width, height=height)
    )

  chart = alt.hconcat(*charts).resolve_scale(color="independent")

  # Always open links in a new tab.
  chart["usermeta"] = {"embedOptions": {"loader": {"target": "_blank"}}}

  return chart
