from .q_learner import QLearner
from .dcg_learner import DCGLearner
from .gacg_learner import GroupQLearner
from .vast_learner import VastQLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["dcg_learner"] = DCGLearner

REGISTRY["gacg_learner"] = GroupQLearner
REGISTRY["vast_learner"] = VastQLearner
