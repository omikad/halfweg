from collections import deque
from typing import Optional

import numpy as np
from environments import env_base
from hw_impl import env_torch_wrapper, model_mgmt



class ExperienceReplayEpisode:
    def __init__(self, initial_env: env_base.BaseEnv):
        self.initial_env = initial_env.copy()
        self.actions = []
        self.terminal_reward = 0

    def on_action(self, action, reward: int, done: bool):
        if done:
            self.terminal_reward = reward
        self.actions.append(action)

    def yield_training_tuples(self):
        env = self.initial_env.copy()
        terminal_reward = self.terminal_reward
        yield -1, env, terminal_reward

        for action in self.actions:
            env.step(action)
            if not env.done:
                yield action, env, terminal_reward

    def yield_steps_tuples(self):
        env = self.initial_env.copy()
        yield -1, env, 0.0

        for action in self.actions:
            reward, done = env.step(action)
            if not done:
                yield action, env, reward


class ExperienceReplay:
    """
    Store played games history
    """
    def __init__(self, max_episodes):
        self.episodes : deque[ExperienceReplayEpisode] = deque()
        self.max_episodes = max_episodes

    def split(self, chunks: int):
        ers = [ExperienceReplay(self.max_episodes) for _ in range(chunks)]
        for i, episode in enumerate(self.episodes):
            ers[i % chunks].append_replay_episode(episode)
        return ers

    def append_replay_episode(self, replay_episode: ExperienceReplayEpisode):
        self.episodes.append(replay_episode)
        while len(self.episodes) > self.max_episodes:
            self.episodes.popleft()

    def extend(self, other: "ExperienceReplay"):
        for episode in other.episodes:
            self.append_replay_episode(episode)

    def clear(self):
        self.episodes.clear()

    def yield_training_tuples(self):
        for episode in self.episodes:
            yield from episode.yield_training_tuples()


class MemoryEnvsSampler:
    source_memory: ExperienceReplay = None

    def __init__(self, model_keeper: Optional[model_mgmt.ModelKeeper] = None, memory: Optional[ExperienceReplay] = None):
        assert model_keeper is not None
        self.model_keeper = model_keeper
        self.source_memory = memory
        self.episodes = []
        self.computed_envs_cached = []
        self.episode_probs = np.zeros(0)
        self.reset_cache()

    def reset_cache(self):
        mem = self.get_memory()
        if mem is not None:
            self.episodes = list(mem.episodes)
            self.computed_envs_cached = [[ep.initial_env] for ep in mem.episodes]
            self.episode_probs = np.array([len(ep.actions) + 1 for ep in mem.episodes], dtype=np.float32)
            self.episode_probs /= np.sum(self.episode_probs)
        else:            
            self.episodes.clear()
            self.computed_envs_cached.clear()
            self.episode_probs = np.zeros(0)

    def get_memory(self) -> ExperienceReplay:
        if self.model_keeper is not None:
            return self.model_keeper.long_memory
        return self.source_memory

    def _ensure_episode(self, gi: int, si: int):
        episodes = self.episodes
        assert 0 <= si <= len(episodes[gi].actions)
        computed_envs = self.computed_envs_cached[gi]
        while si >= len(computed_envs):
            i = len(computed_envs) - 1
            env_copy = computed_envs[i].copy()
            action = episodes[gi].actions[i]
            env_copy.step(action)
            computed_envs.append(env_copy)

    def get_random_env(self, also_sample_final: bool) -> env_base.BaseEnv:
        episodes = self.episodes

        gi = np.random.choice(np.arange(len(episodes)), p=self.episode_probs)
        si = np.random.randint(0, len(episodes[gi].actions) + (1 if also_sample_final else 0))

        self._ensure_episode(gi, si)

        picked_env = self.computed_envs_cached[gi][si]

        ri = np.random.randint(0, picked_env.get_symmetrical_envs_count())
        return picked_env.get_symmetrical_env(ri)

    def get_random_envs(self, cnt: int, also_sample_final: bool) -> env_torch_wrapper.EnvsTensorList:
        return env_torch_wrapper.EnvsTensorList(envs=[self.get_random_env(also_sample_final) for _ in range(cnt)])
    
    def get_random_envs_same_episode(self, cnt: int):
        episodes = self.episodes
        computed_envs_cached = self.computed_envs_cached
        gi = np.random.choice(np.arange(len(episodes)), p=self.episode_probs)
        si = len(episodes[gi].actions)
        self._ensure_episode(gi, si)

        ri = np.random.randint(0, computed_envs_cached[gi][0].get_symmetrical_envs_count())

        I = np.random.randint(0, si + 1, size=cnt)
        for i in I:
            yield computed_envs_cached[gi][i].get_symmetrical_env(ri)