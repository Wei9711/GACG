REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .cg_episode_runner import CGEpisodeRunner
REGISTRY["cg_episode"] = CGEpisodeRunner