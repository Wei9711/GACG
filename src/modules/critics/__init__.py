from .coma import COMACritic
from .centralV import CentralVCritic
REGISTRY = {}

REGISTRY["coma_critic"] = COMACritic
REGISTRY["cv_critic"] = CentralVCritic