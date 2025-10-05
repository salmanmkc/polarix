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
from polarix._src.solvers import schedule as _schedule


class ScheduleTest(parameterized.TestCase):

  @parameterized.parameters([
      dict(
          values=[1.0, 0.5, 0.25, 0.125, 0.5, 0.25],
          threshold=0.25,
          min_steps_before_increment=1,
          expected=[0, 0, 1, 1, 1, 2],
      ),
  ])
  def test_threshold_anneal(
      self, values, threshold, min_steps_before_increment, expected
  ):
    schedule = _schedule.fixed_threshold(
        schedule=lambda x: x,
        threshold=threshold,
        min_steps_before_increment=min_steps_before_increment,
    )
    state = schedule.init()
    for value, expected_value in zip(values, expected):
      state = schedule.update(value, state)
      self.assertEqual(schedule.apply(state), expected_value)

  @parameterized.parameters([
      dict(
          values=[1.0, 1.0, 2.0, 1.0, 1.0, 1.0],
          min_steps_before_increment=1,
          expected=[0, 0, 1, 1, 1, 2],
      ),
      dict(
          values=[1.0, 1.0, 2.0, 1.0, 1.0, 1.0],
          min_steps_before_increment=0,
          expected=[0, 1, 1, 1, 2, 2],
      ),
      dict(
          values=[1.0, 0.5, 0.5, 0.5, 0.25, 0.25],
          min_steps_before_increment=1,
          expected=[0, 0, 0, 1, 1, 1],
      ),
      dict(
          values=[1.0, 0.5, 0.5, 0.5, 0.25, 0.25],
          min_steps_before_increment=2,
          expected=[0, 0, 0, 0, 0, 0],
      ),
  ])
  def test_adaptive_threshold(
      self, values, min_steps_before_increment, expected
  ):
    schedule = _schedule.adaptive_threshold(
        schedule=lambda x: x,
        min_steps_before_increment=min_steps_before_increment,
    )
    state = schedule.init()
    actuals = []
    for value in values:
      state = schedule.update(value, state)
      actuals.append(schedule.apply(state))
    self.assertSequenceEqual(actuals, expected)


if __name__ == '__main__':
  absltest.main()
