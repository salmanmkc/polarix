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

"""Utility functions for testing Altair charts."""

import os
import pathlib
import re

from absl import logging
from absl.testing import parameterized
import altair as alt

from etils import epath


def to_stable_chart_json(chart: alt.Chart) -> str:
  """Converts an Altair chart to JSON, making the output more stable.

  This function disables altair's dataset consolidation, which can cause
  the JSON output to be unstable across runs. It also replaces selector
  names with a placeholder and removes the schema URL, making the output
  more suitable for use in tests with golden files.

  Args:
    chart: The Altair chart to convert.

  Returns:
    The JSON representation of the chart, with stability improvements.
  """
  # Disabling "consolidate_datasets" for datasets hashing.
  with alt.data_transformers.enable(consolidate_datasets=False):
    json_data = chart.to_json()

  # See https://github.com/vega/altair/issues/3416 for context.
  # Replace selector names. (for Altair v4)
  json_data = re.sub(r"selector\d+", "selectorXXX", json_data)
  # Replace param names. (for Altair v5)
  json_data = re.sub(r"param_\d+", "paramXXX", json_data)
  # Replace param names. (for Altair v5)
  json_data = re.sub(r"view_\d+", "viewXXX", json_data)
  # Remove the schema URL.
  json_data = re.sub(r'"\$schema": ".*"', '"$schema": "<removed>"', json_data)
  return json_data


_MISMATCH_TEMPLATE = """
ACTUAL:

{chart_json}

EXPECTED:

{expected_json}

Consider update {filename} if you expect these changes.
Inspect the ACTUAL chart at {actual_path}.
"""


class JsonAlmostEqualTestCase(parameterized.TestCase):
  """TestCase that check for JSON near equality against golden files."""

  def read_testdata(self, filename: str) -> str | bytes:
    return open(filename, "r").read()

  def assertEqual(self, first, second, msg=None):
    # Override assertEqual to use assertAlmostEqual to account for floats in
    # case of floats.
    # NOTE: `assertEqual` is used in `assertJsonEqual` (absl/.../absltest.py).
    if isinstance(first, float) and isinstance(second, float):
      return self.assertAlmostEqual(first, second, msg=msg)
    else:
      return super().assertEqual(first, second, msg=msg)

  def assertChartEqual(self, chart: alt.Chart, filename: str):  # pylint: disable=invalid-name
    """Asserts that the chart is equal to the golden file."""
    if alt.__version__.startswith("4"):
      logging.info(
          "Skipping chart comparison for altair v4 compatibility tests."
      )
      return

    testdata = epath.resource_path("polarix") / "_src" / "viz" / "testdata"
    filename = testdata / filename
    actual_path = testdata / f"ACTUAL_{pathlib.Path(filename).stem}.html"
    actual_path = actual_path.as_posix()

    # Export the chart to a file for visual inspection.
    chart.save(actual_path, format="html")
    chart_json = to_stable_chart_json(chart)
    expected_json = (self.read_testdata(filename),)
    self.assertJsonEqual(
        chart_json,
        self.read_testdata(filename),
        msg=(
            _MISMATCH_TEMPLATE.format(
                chart_json=chart_json,
                expected_json=expected_json,
                filename=filename,
                actual_path=actual_path,
            )
        ),
    )
