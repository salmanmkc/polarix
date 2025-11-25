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

"""Utils and loss for computing affinity-based entropy."""

from collections.abc import Callable, Generator, Iterator, Sequence

import chex
import jax
import jax.numpy as jnp
import optax
from polarix._src.games import base
from polarix._src.solvers import run
from polarix._src.solvers import schedule

Kernels = Sequence[chex.Array]
KernelFn = Callable[[chex.Array], Kernels]
RatingFn = Callable[
    [
        chex.Array,
        base.Logits,
        base.Masks,
        schedule.AdaptiveThresholdState,
        chex.ArrayTree,
    ],
    dict[str, chex.ArrayTree | chex.Numeric],
]
UpdateFn = Callable[
    [
        base.Logits,
        optax.OptState,
        schedule.AdaptiveThresholdState,
        Sequence[chex.Array],
        base.Masks,
    ],
    tuple[
        base.Logits,
        optax.OptState,
        schedule.AdaptiveThresholdState,
        chex.ArrayTree,
    ],
]


def affinity_entropy(
    probs: chex.Array,
    kernel: chex.Array,
    mask: chex.Array,
    p: float = 0.0,
) -> chex.Array:
  """Affinity entropy of a distribution w.r.t. a kernel.

  Args:
    probs: nai jnp.array, the distribution to compute affinity entropy for.
    kernel: nai x nai jnp.array, the affinity kernel to use.
    mask: jnp.array, the action mask.
    p: float, the Tsallis entropy parameter (p in [0, 1]). p=0 corresponds to
      Shannon entropy (limit of Tsallis entropy as p -> 0). This definition is
      concave in probabilities.

  Returns:
    float, the affinity entropy of the distribution w.r.t. the kernel.
  """
  column_sum = jnp.sum(kernel ** (p + 1.0) * mask, axis=0) ** (1.0 / (p + 1.0))
  safe = column_sum > 0.0
  lambda_ = jnp.where(safe, 1 / (jnp.where(safe, column_sum, 1.0)), 0.0)
  lambda_probs = lambda_ * probs * mask
  kernel_lambda_probs = jnp.dot(kernel, lambda_probs)

  if p > 0.0:
    entr = 1.0 / p * (1.0 - jnp.sum(kernel_lambda_probs ** (p + 1.0)))
  else:
    log_column_sum = jnp.where(
        safe, jnp.log(jnp.where(safe, column_sum, 1.0)), 0.0
    )

    safe = kernel_lambda_probs > 0.0
    kernel_lambda_log_probs = jnp.where(
        safe, jnp.log(jnp.where(safe, kernel_lambda_probs, 1.0)), 0.0
    )
    entr = -jnp.sum(kernel_lambda_probs * kernel_lambda_log_probs)

    kernel_lambda = kernel * lambda_
    safe = kernel > 0.0
    log_kernel = jnp.where(safe, jnp.log(jnp.where(safe, kernel, 1.0)), 0.0)
    correction = log_column_sum - jnp.sum(kernel_lambda * log_kernel, axis=0)
    entr -= jnp.sum(correction * probs)

  return entr


def _sample_estimate_kernel(
    payoffs: chex.Array, profiles: Sequence[chex.Array]
) -> list[chex.Array]:
  """Estimates the kernel for a given player from a set of profiles."""
  npl = len(profiles)
  kernels = []
  for p in range(npl):
    joint = jnp.ones(payoffs.shape[1:])
    for q in range(npl):
      not_q = tuple(nq for nq in range(npl) if nq != q)
      if q != p:
        joint *= jnp.expand_dims(profiles[q], axis=not_q)
    not_p = tuple(q for q in range(npl) if q != p)
    grad = jnp.sum(payoffs[p] * joint, axis=not_p)
    diff = jnp.expand_dims(grad, 1) - jnp.expand_dims(grad, 0)
    kernels.append(jnp.square(diff))
  return kernels


def _make_update_fn(
    opt: optax.GradientTransformation,
    early_stopping: schedule.Schedule,
    p: float = 0.0,
    entropy_cost: float = 0.0,
) -> UpdateFn:
  """Returns an update function for affinity-entropy solving.

  Args:
    opt: the optimiser used.
    early_stopping: the early stopping schedule.
    p: float, the Tsallis entropy parameter (p in [0, 1]). See
      `affinity_entropy`.
    entropy_cost: float >= 0, coefficient for regularizing towards uniform dist.

  Returns:
    An update function that can be recursively applied to approximate an NE.
  """

  def loss_fn(
      logits: base.Logits, kernels: Kernels, masks: base.Masks
  ) -> tuple[chex.Array, chex.Array]:
    affinity_entropies = []
    for logit_i, kernel_i, mask_i in zip(logits, kernels, masks):
      probs_i = jax.nn.softmax(logit_i, where=mask_i)
      aff_entropy = affinity_entropy(probs_i, kernel_i, mask_i, p)
      log_probs_i = jnp.where(
          probs_i == 0.0, 0.0, jax.nn.log_softmax(logit_i, where=mask_i)
      )
      entropy = -jnp.sum(probs_i * log_probs_i, axis=-1)
      affinity_entropies.append(aff_entropy + entropy_cost * entropy)
    affinity_entropies = jnp.asarray(affinity_entropies)
    loss = -jnp.sum(affinity_entropies)
    return loss, affinity_entropies

  @jax.jit
  def update(logits, opt_state, early_stopping_state, kernels, masks):
    (loss, affinity_entropies), grad = jax.value_and_grad(
        loss_fn, has_aux=True
    )(logits, kernels, masks)
    updates, opt_state = opt.update(grad, opt_state, logits)
    early_stopping_state = early_stopping.update(loss, early_stopping_state)
    logits = optax.apply_updates(logits, updates)
    extra = dict(
        affinity_entropies=affinity_entropies,
        loss=loss,
        **early_stopping_state._asdict(),
    )
    return logits, opt_state, early_stopping_state, extra

  return update


def _make_marginal_rating_fn(
    early_stopping: schedule.Schedule,
) -> RatingFn:
  """Returns a function that computes ratings from logits.

  Args:
    early_stopping: the early stopping schedule.

  Returns:
    A function that can be used to compute ratings from logits.
  """

  @jax.jit
  def _to_marginal_ratings(
      payoffs: chex.Array,
      logits: base.Logits,
      masks: base.Masks,
      early_stopping_state: schedule.AdaptiveThresholdState,
      extra: chex.ArrayTree,
  ) -> dict[str, chex.ArrayTree | chex.Numeric]:
    marginals = [jax.nn.softmax(l, -1, m) for l, m in zip(logits, masks)]

    ratings = []
    for p in range(len(logits)):
      ratings.append(base.marginal_payoffs(p, payoffs, logits, masks))

    terminal = early_stopping.apply(early_stopping_state)
    return dict(
        ratings=ratings, terminal=terminal, marginals=marginals, extra=extra
    )

  return _to_marginal_ratings


def _make_joint_rating_fn(
    early_stopping: schedule.Schedule,
) -> RatingFn:
  """Returns a function that computes ratings from logits.

  Args:
    early_stopping: the early stopping schedule.

  Returns:
    A function that can be used to compute ratings from logits.
  """

  @jax.jit
  def _to_joint_ratings(
      payoffs: chex.Array,
      logits: base.Logits,
      masks: base.Masks,
      early_stopping_state: schedule.AdaptiveThresholdState,
      extra: chex.ArrayTree,
  ) -> dict[str, chex.ArrayTree | chex.Numeric]:
    ratings = _make_marginal_rating_fn(early_stopping)(
        payoffs, logits, masks, early_stopping_state, extra
    )
    joint = base.joint_from_marginals(ratings.pop("marginals"))
    return dict(joint=joint, **ratings)

  return _to_joint_ratings


def _solve_maxent_marginals(
    kernels: Sequence[chex.Array],
    masks: base.Masks,
    optim: optax.GradientTransformation,
    early_stopping: schedule.Schedule,
    p: float = 1.0,
    entropy_cost: float = 0.0,
) -> Iterator[
    tuple[base.Logits, schedule.AdaptiveThresholdState, chex.ArrayTree]
]:
  """Yields an approximate marginals for payoffs that is max-affinity-entropy.

  Args:
    kernels: list of chex.Array with each array being a na_i x na_i similarity
      kernel (symmetric with entries in [0, 1])
    masks: the action masks for each player.
    optim: the optimiser used.
    early_stopping: the early stopping schedule.
    p: float, the Tsallis entropy parameter (p in [0, 1]). See
      `affinity_entropy`.
    entropy_cost: float >= 0, coefficient for regularizing towards uniform dist.

  Yields:
    logits: the max-affinity-entropy strategy profile ([|A_1|], ..., [|A_N|]).
    loss: current loss.
    affinity_entropies: the affinity entropies for each player.
  """
  assert len(kernels) == len(masks)
  chex.assert_equal_rank(masks)
  chex.assert_rank(masks, {1, 2})

  logits = [jnp.zeros_like(mask, dtype=jnp.float32) for mask in masks]
  early_stopping_state = early_stopping.init()

  update_fn = _make_update_fn(optim, early_stopping, p, entropy_cost)
  opt_state = optim.init(logits)
  while True:
    logits, opt_state, early_stopping_state, extra = update_fn(
        logits, opt_state, early_stopping_state, kernels, masks
    )
    yield logits, early_stopping_state, extra


def affinity_kernel(
    key: chex.Array,
    sample_size: int = 512,
    kernel_variance: float = 1e-6,
) -> KernelFn:
  """Returns a function that estimates each player's kernel from samples."""

  @jax.jit
  def kernel_fn(payoffs: chex.Array) -> list[chex.Array]:
    _, *na = payoffs.shape
    keys = jax.random.split(key, len(na))
    profiles = []
    for na_p, key_p in zip(na, keys):
      profiles.append(
          jax.random.dirichlet(key_p, jnp.ones(na_p), (sample_size,))
      )
    kernels = jax.vmap(_sample_estimate_kernel, in_axes=(None, 0))(
        payoffs, profiles
    )
    return [jnp.exp(-jnp.mean(k, 0) / (2.0 * kernel_variance)) for k in kernels]

  return kernel_fn


def identity_kernel() -> KernelFn:
  """Returns a function that returns identity kernels for each player."""

  @jax.jit
  def kernel_fn(payoffs: chex.Array) -> list[chex.Array]:
    return [jnp.eye(nai) for nai in payoffs.shape[1:]]

  return kernel_fn


def _max_affinity_entropy(
    game: base.Game,
    kernel_fn: KernelFn,
    rating_fn: RatingFn,
    early_stopping: schedule.Schedule,
    masks: Sequence[chex.Array] | None = None,
    optim: optax.GradientTransformation = optax.adam(1e-2),
    p: float = 1.0,
    entropy_cost: float = 0.0,
) -> Iterator[dict[str, chex.ArrayTree]]:
  """Yields an approximate uniform affinity marginal profile.

  Args:
    game: the game to solve.
    kernel_fn: the function that estimates the kernels for each player.
    rating_fn: the function that computes ratings from logits.
    early_stopping: if the loss has not decreased after this number of updates,
      stop early.
    masks: the action masks for each player.
    optim: the optimiser used.
    p: float, the Tsallis entropy parameter (p in [0, 1]). See
      `affinity_entropy`.
    entropy_cost: float >= 0, coefficient for regularizing towards uniform dist.

  Yields:
    a `run.MarginalRatings` at each iteration.
  """
  kernels = kernel_fn(game.payoffs)

  if masks is None:
    masks = [jnp.ones(nai) for nai in game.payoffs.shape[1:]]

  masks = [jnp.asarray(m, dtype=jnp.bool_) for m in masks]
  rating_fn = jax.jit(rating_fn)

  for logits, early_stopping_state, extra in _solve_maxent_marginals(
      kernels, masks, optim, early_stopping, p, entropy_cost
  ):
    yield rating_fn(game.payoffs, logits, masks, early_stopping_state, extra)


def max_affinity_entropy_marginals(
    game: base.Game,
    kernel_fn: KernelFn,
    masks: Sequence[chex.Array] | None = None,
    optim: optax.GradientTransformation = optax.adam(1e-2),
    early_stopping: int = 10_000,
    p: float = 1.0,
    entropy_cost: float = 0.0,
) -> Generator[run.MarginalRatings, None, None]:
  """Yields an approximate uniform affinity marginal profile.

  Args:
    game: the game to solve.
    kernel_fn: the function that estimates the kernels for each player.
    masks: the action masks for each player.
    optim: the optimiser used.
    early_stopping: if the loss has not decreased after this number of updates,
      stop early.
    p: float, the Tsallis entropy parameter (p in [0, 1]). See
      `affinity_entropy`.
    entropy_cost: float >= 0, coefficient for regularizing towards uniform dist.

  Yields:
    a `run.MarginalRatings` at each iteration.
  """
  early_stopping = schedule.adaptive_threshold(
      lambda x: x > 0, min_steps_before_increment=early_stopping
  )
  rating_fn = _make_marginal_rating_fn(early_stopping)
  for ratings in _max_affinity_entropy(
      game=game,
      kernel_fn=kernel_fn,
      rating_fn=rating_fn,
      masks=masks,
      optim=optim,
      early_stopping=early_stopping,
      p=p,
      entropy_cost=entropy_cost,
  ):
    yield run.MarginalRatings(**ratings)


def max_affinity_entropy_joint(
    game: base.Game,
    kernel_fn: KernelFn,
    masks: Sequence[chex.Array] | None = None,
    optim: optax.GradientTransformation = optax.adam(1e-2),
    early_stopping: int = 10_000,
    p: float = 1.0,
    entropy_cost: float = 0.0,
) -> Generator[run.JointRatings, None, None]:
  """Yields an approximate uniform affinity marginal profile.

  Args:
    game: the game to solve.
    kernel_fn: the function that estimates the kernels for each player.
    masks: the action masks for each player.
    optim: the optimiser used.
    early_stopping: if the loss has not decreased after this number of updates,
      stop early.
    p: float, the Tsallis entropy parameter (p in [0, 1]). See
      `affinity_entropy`.
    entropy_cost: float >= 0, coefficient for regularizing towards uniform dist.

  Yields:
    a `run.JointRatings` at each iteration.
  """
  early_stopping = schedule.early_stopping(early_stopping)
  rating_fn = _make_joint_rating_fn(early_stopping)
  for ratings in _max_affinity_entropy(
      game=game,
      kernel_fn=kernel_fn,
      rating_fn=rating_fn,
      masks=masks,
      optim=optim,
      early_stopping=early_stopping,
      p=p,
      entropy_cost=entropy_cost,
  ):
    yield run.JointRatings(**ratings)
