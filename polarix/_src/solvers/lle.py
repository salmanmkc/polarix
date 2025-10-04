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

"""Limited logit equilibrium solver."""

from collections.abc import Callable, Generator, Sequence
import functools
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import optax
from polarix._src.games import base
from polarix._src.solvers import run
from polarix._src.solvers import schedule


def _qre_exploitability_logits(
    logits: base.Logits,
    temperature: chex.Array,
    target_logits: Sequence[chex.Array],
    marginal_payoffs: Callable[..., chex.Array],
) -> tuple[chex.Array, tuple[chex.Array, Sequence[chex.Array]]]:
  """Calculates QRE exploitability from approx eq (logits).

  As seen in Gemp et al, 2021:
  "Sample-based Approximation of Nash in Large Many-Player Games via Gradient
  Descent". https://arxiv.org/abs/2106.01285

  Args:
    logits: list of jnp arrays with each array containing na_i logits
    temperature: float >= 0
    target_logits: list of jnp arrays with each array being a na_i vector of
      target logits, player utilities will be penalized with kl divergence from
      these target_logits.
    marginal_payoffs: function to calculate marginal payoffs given player index,
      logits, and masks.

  Returns:
    NashConv w/ KL divergence regularization and nash_conv in the original game.
  """
  qre_exp_sum = jnp.zeros(())
  exp = jnp.zeros(())
  ratings = []
  for i, (logit_i, target_i) in enumerate(
      zip(logits, target_logits)
  ):
    grad_i = marginal_payoffs(i, logits=logits)

    log_dist_i = jax.nn.log_softmax(logit_i)
    dist_i = jnp.exp(log_dist_i)
    log_dist_i = jnp.where(dist_i == 0, 0.0, log_dist_i)
    dist_i_kl = jnp.sum(dist_i * (log_dist_i - target_i), axis=-1)

    br_i_logit = grad_i / temperature + target_i
    log_br_i = jax.nn.log_softmax(br_i_logit)
    br_i = jnp.exp(log_br_i)
    log_br_i = jnp.where(br_i == 0, 0.0, log_br_i)
    br_i_kl = jnp.sum(br_i * (log_br_i - target_i), axis=-1)

    # Evaluate value gap in the original game.
    u_i_dist_ = jnp.dot(grad_i, dist_i)
    u_i_br_ = jnp.max(grad_i, initial=-jnp.inf)
    exp = jnp.maximum(exp, u_i_br_ - u_i_dist_)
    ratings_i = grad_i - u_i_dist_
    ratings.append(ratings_i)

    # Evaluate value gap in the annealed game.
    u_i_dist = u_i_dist_ - temperature * dist_i_kl
    u_i_br = jnp.dot(grad_i, br_i) - temperature * br_i_kl
    qre_exp_sum += u_i_br - u_i_dist

  return qre_exp_sum, (exp, ratings)


class LLEOutputs(NamedTuple):
  """LLE solver outputs."""

  terminal: chex.Array
  loss: chex.Array
  trigger: chex.Array
  exp: chex.Array
  temperature: chex.Array
  ratings: Sequence[chex.Array]
  marginals: Sequence[chex.Array]


def _lle_update(
    opt: optax.GradientTransformation,
    game: base.Game,
    temperature_schedule: schedule.Schedule,
    epsilon: float = 1e-8,
):
  """Returns an update function for LLE solving.

  Args:
    opt: the optimiser used.
    game: the game to solve.
    temperature_schedule: the temperature annealing schedule.
    epsilon: the target level of approximation in terms of exploitability for
      determining if the resulting ratings are terminal.

  Returns:
    An update function that can be recursively applied to approximate an NE.
  """

  @jax.jit
  def update(
      logits: base.Logits,
      opt_state: optax.OptState,
      anneal_state: schedule.FixedThresholdState,
      target_logits: Sequence[chex.Array],
  ) -> tuple[
      base.Logits,
      optax.OptState,
      schedule.FixedThresholdState,
      LLEOutputs,
  ]:
    loss_fn = _qre_exploitability_logits

    marginal_payoffs = functools.partial(
        base.marginal_payoffs, payoffs=game.payoffs
    )

    temp = temperature_schedule.apply(anneal_state)
    loss_fn = functools.partial(loss_fn, marginal_payoffs=marginal_payoffs)

    (loss, (exp, ratings)), grad = jax.value_and_grad(loss_fn, has_aux=True)(
        logits, temp, target_logits,
    )
    trigger = loss

    updates, opt_state = opt.update(grad, opt_state, logits)

    logits = optax.apply_updates(logits, updates)
    marginals = [jax.nn.softmax(l, -1) for l in logits]

    anneal_state = temperature_schedule.update(trigger, anneal_state)
    outputs = LLEOutputs(
        terminal=jnp.less_equal(exp, epsilon),
        loss=loss,
        trigger=trigger,
        exp=exp,
        temperature=temp,
        ratings=ratings,
        marginals=marginals,
    )
    return logits, opt_state, anneal_state, outputs

  return update


def lle(
    game: base.Game,
    optim: optax.GradientTransformation = optax.adam(1e-2),
    epsilon: float = 1e-8,
    init_temperature: float = 1.0,
    min_temperature: float = 1e-5,
    anneal_rate: float = 0.95,
    min_anneal_iters: int = 250,
    target_logits: Sequence[chex.Array] | None = None,
) -> Generator[run.MarginalRatings, None, None]:
  """Yields an approximate NE for payoffs.

  Args:
    game: the game to solve.
    optim: the optimiser used.
    epsilon: the target level of approximation in terms of exploitability for
      determining if the resulting ratings are terminal.
    init_temperature: the initial temperature.
    min_temperature: the minimum level of temperature.
    anneal_rate: the rate at which the temperature is annealed (multiplied by).
    min_anneal_iters: the minimum number of iterations before annealing.
    target_logits: the target logits to use for KL divergence regularization.

  Yields:
    a `run.MarginalRatings` at each iteration.
  """
  na = list(map(len, game.actions))

  if target_logits is None:
    target_logits = [jnp.zeros(nai) for nai in na]

  logits = [jnp.array(t) for t in target_logits]

  # Ensure all target_logits have the same rank (1).
  chex.assert_equal_rank(list(target_logits))
  chex.assert_rank(target_logits, {1})

  temperature_schedule = schedule.adaptive_threshold(
      schedule=optax.exponential_decay(
          init_temperature,
          transition_steps=1,
          decay_rate=anneal_rate,
          end_value=min_temperature,
      ),
      min_steps_before_increment=min_anneal_iters,
  )
  anneal_state = temperature_schedule.init()

  update_fn = _lle_update(
      opt=optim,
      game=game,
      temperature_schedule=temperature_schedule,
      epsilon=epsilon,
  )
  opt_init_fn = optim.init

  opt_state = opt_init_fn(logits)

  # Place on devices to avoid cross-device-transfer of static inputs.
  target_logits = tuple(jax.device_put(t) for t in target_logits)

  while True:
    logits, opt_state, anneal_state, outputs = update_fn(
        logits, opt_state, anneal_state, target_logits,
    )
    yield run.MarginalRatings(
        ratings=outputs.ratings,
        marginals=outputs.marginals,
        terminal=outputs.terminal,
        extra=dict(
            logits=logits, **outputs._asdict(), **anneal_state._asdict()
        ),
    )


def _nash_conv_from_logits(
    logits: base.Logits, payoffs: chex.Array
) -> tuple[chex.Array, Sequence[chex.Array]]:
  """Calculates exploitability (with arbitrary noise) from approx eq (logits).

  Args:
    logits: list of jnp arrays with each array containing na_i logits
    payoffs: npl x na_1 x ... na_npl payoff tensor

  Returns:
    NashConv of current approx eq profile.
  """
  exp = jnp.zeros(())
  ratings = []
  for i, logit_i in enumerate(logits):
    grad_i = base.marginal_payoffs(i, payoffs, logits=logits)

    log_dist_i = jax.nn.log_softmax(logit_i)
    dist_i = jnp.exp(log_dist_i)

    # Evaluate value gap in the original game.
    u_i_dist_ = jnp.dot(grad_i, dist_i)
    u_i_br_ = jnp.max(grad_i, initial=-jnp.inf)
    exp = jnp.maximum(exp, u_i_br_ - u_i_dist_)

    ratings_i = grad_i - u_i_dist_
    ratings.append(ratings_i)

  return exp, ratings
