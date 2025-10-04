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

"""The polarix public API."""

# pylint: disable=g-importing-member,useless-import-alias
from polarix._src.games.agent_vs_task import agent_vs_task_game
from polarix._src.games.base import Game
from polarix._src.games.base import joint_from_marginals
from polarix._src.games.base import joint_payoffs_contribution
from polarix._src.games.base import marginal_payoffs
from polarix._src.games.base import marginals_from_joint
from polarix._src.games.sxs import diff_game as sxs_diff_game
from polarix._src.games.sxs import winrate_game as sxs_winrate_game
from polarix._src.solvers.affinity_entropy import affinity_kernel
from polarix._src.solvers.affinity_entropy import identity_kernel
from polarix._src.solvers.affinity_entropy import max_affinity_entropy_joint
from polarix._src.solvers.affinity_entropy import max_affinity_entropy_marginals
from polarix._src.solvers.average import average
from polarix._src.solvers.ce_maxent import ce_maxent
from polarix._src.solvers.lle import lle
from polarix._src.solvers.run import JointRatings
from polarix._src.solvers.run import MarginalRatings
from polarix._src.solvers.run import Ratings
from polarix._src.solvers.run import solve
from polarix._src.viz.equilibrium import rating_and_marginal as plot_rating_and_marginal
from polarix._src.viz.marginal_contrib import rating_contribution as plot_rating_contribution
# pylint: enable=g-importing-member,useless-import-alias


# A new PyPI release will be pushed every time `__version__` is increased.
# When changing this, also update the CHANGELOG.md.
__version__ = "0.1.0"


__all__ = (
    "Game",
    "JointRatings",
    "MarginalRatings",
    "Ratings",
    "affinity_kernel",
    "agent_vs_task_game",
    "average",
    "ce_maxent",
    "identity_kernel",
    "joint_from_marginals",
    "joint_payoffs_contribution",
    "lle",
    "marginal_payoffs",
    "marginals_from_joint",
    "max_affinity_entropy_joint",
    "max_affinity_entropy_marginals",
    "plot_rating_and_marginal",
    "plot_rating_contribution",
    "solve",
    "sxs_diff_game",
    "sxs_winrate_game",
)

#  _________________________________________
# / Please don't use symbols in `_src` they \
# \ are not part of the polarix public API.    /
#  -----------------------------------------
#         \   ^__^
#          \  (oo)\_______
#             (__)\       )\/\
#                 ||----w |
#                 ||     ||
#
try:
  del _src  # pylint: disable=undefined-variable
except NameError:
  pass
