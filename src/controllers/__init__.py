REGISTRY = {}

from .basic_controller import BasicMAC
from .dcg_controller import DeepCoordinationGraphMAC
from .dicg_controller import DICGraphMAC
from .gacg_controller import GroupMessageMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["dcg_mac"] = DeepCoordinationGraphMAC
REGISTRY["dicg_mac"] = DICGraphMAC
REGISTRY["gacg_mac"] = GroupMessageMAC