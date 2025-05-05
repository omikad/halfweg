from typing import Optional

import numpy as np
import torch

from environments import env_base


class EnvsTensorList:
    """
    Provides methods to work with environments, their numpy and torch representations in batch manner
    """
    def __init__(self, envs: Optional[list[env_base.BaseEnv]] = None, states_np: Optional[np.ndarray] = None, states_t: Optional[torch.Tensor] = None):
        self.envs = envs
        self.states_np = states_np
        self.states_t = states_t
        self.device = torch.device('cpu')

    def __len__(self) -> int:
        if self.envs is not None: return len(self.envs)
        if self.states_np is not None: return self.states_np.shape[0]
        if self.states_t is not None: return self.states_t.shape[0]

    def get_envs(self) -> list[env_base.BaseEnv]:
        assert self.envs is not None
        return self.envs

    def get_states_np(self) -> list[np.ndarray]:
        if self.states_np is None:
            if self.states_t is not None:
                self.states_np = self.states_t.cpu().numpy()
            else:
                envs = self.envs
                states_np = None
                for i, env in enumerate(envs):
                    state = env.get_model_input_s()
                    if states_np is None:
                        states_np = np.zeros((len(envs), *state.shape), dtype=np.float32)
                    states_np[i, ...] = state
                self.states_np = states_np
        return self.states_np
    
    def get_states_t(self) -> torch.Tensor:
        if self.states_t is None:
            self.states_t = torch.as_tensor(self.get_states_np(), device=self.device)
        return self.states_t

    def to(self, device):
        self.device = device
        if self.states_t is not None:
            self.states_t = self.states_t.to(device)

    def trace_env(self, env_idx: int, plan: np.ndarray):
        assert len(plan.shape) == 1 and plan.dtype == np.int64

        env = self.get_envs()[env_idx].copy()
        AS = len(env.get_valid_actions_mask())
        
        yield 0, env
        for ai, action in enumerate(plan):
            if action == AS:
                break
            if not env.done and env.get_valid_actions_mask()[action] > 0:
                env.step(action)
            yield ai + 1, env

    def get_states_on_trajectory(self, env_idx: int, plan: np.ndarray, sorted_state_indices: list[int]) -> list[np.ndarray]:
        assert len(plan.shape) == 1 and plan.dtype == np.int64
        ssi = 0
        states = []
        for si, env in self.trace_env(env_idx, plan):
            while ssi < len(sorted_state_indices) and si == sorted_state_indices[ssi]:
                states.append(env.get_model_input_s().copy())
                ssi += 1
            if ssi >= len(sorted_state_indices):
                break
        return states

    def apply_actions__batch(self, actions_np: Optional[np.ndarray] = None, actions_t: Optional[torch.Tensor] = None) -> "EnvsTensorList":
        if actions_np is None:
            actions_np = actions_t.cpu().numpy()
        assert len(actions_np.shape) == 2
        B = actions_np.shape[0]

        envs = self.get_envs()
        assert len(envs) == B

        AS = len(envs[0].get_valid_actions_mask())

        result_envs = envs[0].copy_and_apply_actions_batch(envs, actions_np, AS)
        return EnvsTensorList(envs=result_envs)

    def tile(self, cnt: int) -> "EnvsTensorList":
        assert len(self) == 1

        s0_env = self.get_envs()[0]
        tiled_s0_envs = [s0_env] * cnt

        s0_enc = self.get_states_t()
        tiled_s0_enc = torch.tile(s0_enc, (cnt, 1, 1, 1))

        tiled_envs = EnvsTensorList(envs=tiled_s0_envs, states_np=None, states_t=tiled_s0_enc)
        tiled_envs.device = self.device
        return tiled_envs
    
    def extend_envs(self, envs: list[env_base.BaseEnv]):
        self.envs.extend(envs)
        self.states_np = None
        self.states_t = None