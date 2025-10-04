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

"""Implements schedules based on different thresholding criteria."""

from collections.abc import Callable
from typing import Any, NamedTuple

import chex
import jax
import jax.numpy as jnp
import optax


State = Any


def repeats(state: State, size: int) -> State:
  return jax.tree.map(
      lambda x: jnp.repeat(jnp.asarray(x)[jnp.newaxis], size, axis=0),
      state,
  )


class Schedule(NamedTuple):
  """A triplet of pure functions implementing a scheduler.

  Attributes:
    init: A pure function which returns a pytree containing the initial value
      for the optimizer state.
    update: A pure function which takes as input the previous optimizer state
      (which may have been initialized using the init function) alongside
      current statistics, and returns a new schedule state.
    apply: A pure function which takes as input the current schedule state and
      returns the current value of the schedule.
  """

  init: Callable[..., State]
  update: Callable[[chex.ArrayTree, State], State]
  apply: Callable[[State], chex.Numeric]


class FixedThresholdState(NamedTuple):
  """A state tuple that keeps track of the fixed threshold schedule.

  Fields:
    increment_update_count: a scalar counter that keeps track of the last time
      the parameter was incremented.
    schedule_count: the count for retrieving the current value of the
      parameter according to its `optax.schedule`.
  """

  increment_update_count: chex.Array
  schedule_count: chex.Array


def fixed_threshold(
    schedule: optax.Schedule,
    threshold: float,
    min_steps_before_increment: int,
) -> Schedule:
  """Returns a schedule that increments when value dips below a threshold."""

  def init() -> FixedThresholdState:
    return FixedThresholdState(
        increment_update_count=jnp.zeros(()), schedule_count=jnp.zeros(())
    )

  def update(
      value: chex.ArrayTree, state: FixedThresholdState
  ) -> FixedThresholdState:
    incrementing = jnp.logical_and(
        value <= threshold,
        state.increment_update_count > min_steps_before_increment,
    )
    increment_update_count = jnp.where(
        incrementing, 0.0, state.increment_update_count + 1
    )
    schedule_count = jnp.where(
        incrementing, state.schedule_count + 1, state.schedule_count
    )
    return FixedThresholdState(
        increment_update_count=increment_update_count,
        schedule_count=schedule_count,
    )

  def apply(state: FixedThresholdState) -> chex.Numeric:
    return schedule(state.schedule_count)

  return Schedule(init=init, update=update, apply=apply)


class AdaptiveThresholdState(NamedTuple):
  """A state tuple that keeps track of the adaptive threshold schedule.

  Fields:
    threshold: the lowest observed value at the current `schedule_count`.
    threshold_updated_at: a scalar counter that keeps track of the last time
      the threshold value has been updated.
    schedule_count: the count for retrieving the current value of the
      parameter according to its `optax.schedule`.
    max_threshold_annealed: a scalar that keeps track of the maximum threshold
    value at each the schedule has incremented.
  """

  threshold: chex.Array
  threshold_count: chex.Array
  schedule_count: chex.Array
  max_threshold_annealed: chex.Array


def adaptive_threshold(
    schedule: optax.Schedule,
    min_steps_before_increment: int,
) -> Schedule:
  """Returns an adaptive schedule based on observing values decreasing.

  Args:
    schedule: the underlying schedule that returns a value based on the
      `schedule_count`.
    min_steps_before_increment: the minimum number of updates required without
      observing a lower value than the current `threshold` before incrementing
      the `schedule_count`.
  """

  def init() -> AdaptiveThresholdState:
    return AdaptiveThresholdState(
        threshold=jnp.full((), jnp.inf),
        threshold_count=jnp.zeros(()),
        schedule_count=jnp.zeros(()),
        max_threshold_annealed=jnp.full((), -jnp.inf),
    )

  def update(value: chex.ArrayTree, state: AdaptiveThresholdState):
    update = value < state.threshold
    increment = ~update & (state.threshold_count >= min_steps_before_increment)
    return AdaptiveThresholdState(
        threshold=jnp.where(
            increment, jnp.inf, jnp.minimum(value, state.threshold)
        ),
        threshold_count=jnp.where(
            update | increment, 0.0, state.threshold_count + 1
        ),
        schedule_count=jnp.where(
            increment, state.schedule_count + 1, state.schedule_count
        ),
        max_threshold_annealed=jnp.where(
            increment,
            jnp.maximum(state.threshold, state.max_threshold_annealed),
            state.max_threshold_annealed,
        ),
    )

  def apply(state: AdaptiveThresholdState) -> chex.Numeric:
    return schedule(state.schedule_count)

  return Schedule(init=init, update=update, apply=apply)


def early_stopping(
    early_stopping_if_no_progress: int,
) -> Schedule:
  """Returns a schedule that returns an indicator for early stopping."""

  return adaptive_threshold(
      schedule=lambda x: x > 0,
      min_steps_before_increment=early_stopping_if_no_progress,
  )
