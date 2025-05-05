import itertools
from typing import Optional
import numpy as np
import torch

from hw_impl import env_torch_wrapper, hw_common, hw_experience_replay, model_mgmt


class BasePolicy:
    def get_behavior_actions(self, s0: env_torch_wrapper.EnvsTensorList, branch_factor: int) -> torch.Tensor:
        raise Exception(f"Override this method in child")

    def get_plan_envs_to_envs(self, s0: env_torch_wrapper.EnvsTensorList, target: env_torch_wrapper.EnvsTensorList, b: torch.Tensor) -> torch.Tensor:
        raise Exception(f"Override this method in child")

    def get_expand_items_for_plhw(self, s0_env: env_torch_wrapper.EnvsTensorList, branch_factor: int, device) -> tuple[Optional[env_torch_wrapper.EnvsTensorList], torch.Tensor]:
        raise Exception(f"Override this method in child")

    def get_plan_last_enc(self, s0_env: Optional[env_torch_wrapper.EnvsTensorList], s0_enc: torch.Tensor, target: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        raise Exception(f"Override this method in child")

    def eval(self):
        pass

    def __str__(self) -> str:
        return self.__class__.__name__


class FixedFullScanPolicy(BasePolicy):
    def __init__(self, plta):
        self.AS = plta.AS
        self.actions_len = plta.PREDICT_STEPS
        self.all_paths_np = None
        self.all_paths_t = None

    def _ensure_cached_all_paths(self):
        AS = self.AS
        actions_len = self.actions_len

        if self.all_paths_t is None:
            res = []
            for path_len in range(1, actions_len + 1):
                for i, path in enumerate(itertools.product(list(range(AS)), repeat=path_len)):
                    res.append(list(path) + [AS] * (actions_len - path_len))
            self.all_paths_np = np.array(res, dtype=np.int64)
            self.all_paths_t = torch.as_tensor(self.all_paths_np)
        else:
            assert self.all_paths_t.shape == self.all_paths_np.shape

    def get_all_paths(self, need_np: bool, need_t: bool) -> tuple[np.ndarray, torch.Tensor]:
        self._ensure_cached_all_paths()
        res_np = None
        if need_np:
            res_np = self.all_paths_np.copy()
        res_t = None
        if need_t:
            res_t = self.all_paths_t.clone()
        return res_np, res_t

    def get_behavior_actions(self, s0: env_torch_wrapper.EnvsTensorList, branch_factor: int) -> torch.Tensor:
        assert len(s0) == 1
        self._ensure_cached_all_paths()
        return self.all_paths_t

    def get_plan_envs_to_envs(self, s0: env_torch_wrapper.EnvsTensorList, target: env_torch_wrapper.EnvsTensorList, b: torch.Tensor) -> torch.Tensor:
        B = len(s0)
        assert len(target) == B
        assert b.shape == (B, 1) and b.dtype == torch.int64

        self._ensure_cached_all_paths()

        all_paths_np = self.all_paths_np

        target_enc = target.get_states_np()

        b = b.cpu().numpy()

        plans_np = np.zeros((B, all_paths_np.shape[1]), dtype=np.int64)

        for i in range(B):
            s0_i = env_torch_wrapper.EnvsTensorList(envs=[s0.get_envs()[i]])
            ns_envs = s0_i.apply_actions(actions_np=all_paths_np)
            ns_enc = ns_envs.get_states_np()

            tsi = target_enc[i:i+1, ...]

            distances = np.sum(np.square(tsi - ns_enc), axis=(1,2,3))
            assert distances.shape == (len(ns_enc),)

            j = np.argmin(distances) if b[i, 0] == 0 else np.argmax(distances)

            plans_np[i, :] = all_paths_np[j, :]

        plans_t = torch.as_tensor(plans_np)

        return plans_t


class PolicyZeroLevel(BasePolicy):
    def __init__(self, plta, layers: list[BasePolicy]):
        self.plta = plta
        self.layers = layers

    def _call_plta(self, s0: env_torch_wrapper.EnvsTensorList, targets: torch.Tensor, b: torch.Tensor):
        B = len(targets)
        assert len(s0) == B
        assert b.shape == (B, 1) and b.dtype == torch.int64

        plta = self.plta
        plta.eval()

        device = b.device
        s0_enc = plta.forward_model_board_normalize(s0.get_states_t().to(device))

        plans = plta.forward_model_target_actions__enc(
            ss=s0_enc,
            ts=targets,
            min_max_01=b)   # (B, PREDICT_STEPS, AS + 1) torch.float32
        plans = torch.argmax(plans, axis=2)
        return plans

    def get_expand_items_for_plhw(self, s0_env: env_torch_wrapper.EnvsTensorList, branch_factor: int, device) -> tuple[Optional[env_torch_wrapper.EnvsTensorList], torch.Tensor]:
        assert len(s0_env) == 1

        _, plans = self.layers[0].get_all_paths(need_np=False, need_t=True)

        ns_env = s0_env.apply_actions(actions_t=plans)
        ns_env.to(device)
        ns_enc = self.plta.forward_model_board_normalize(ns_env.get_states_t())

        return ns_env, ns_enc

    def get_plan_last_enc(self, s0_env: Optional[env_torch_wrapper.EnvsTensorList], s0_enc: torch.Tensor, target: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        assert s0_env is not None
        B = len(s0_env)
        assert len(target) == B
        assert len(b) == B and b.dtype == torch.int64

        plans = self._call_plta(s0=s0_env, targets=target, b=b)
        ns_env = s0_env.apply_actions__batch(actions_t=plans)
        ns_env.to(b.device)
        ns_enc = self.plta.forward_model_board_normalize(ns_env.get_states_t())

        return ns_enc

    def get_plan_envs_to_envs(self, s0: env_torch_wrapper.EnvsTensorList, target: env_torch_wrapper.EnvsTensorList, b: torch.Tensor) -> torch.Tensor:
        B = len(s0)
        assert len(target) == B
        assert b.shape == (B, 1) and b.dtype == torch.int64

        env_obs_shape = s0.get_envs()[0].get_model_input_s().shape
        ts_enc = target.get_states_t().to(b.device)
        if ts_enc.shape[1:] == env_obs_shape:
            ts_enc = self.plta.forward_model_board_normalize(ts_enc)

        plans = self._call_plta(s0=s0, targets=ts_enc, b=b)
        return plans

    def get_behavior_actions(self, s0: env_torch_wrapper.EnvsTensorList, branch_factor: int) -> torch.Tensor:
        return self.layers[0].get_behavior_actions(s0, branch_factor)

    def eval(self):
        self.plta.eval()

    def __str__(self) -> str:
        return f"Policy0 (layer_idx=1)"


class PolicyHighLevel(BasePolicy):
    def __init__(self, model_keeper: model_mgmt.ModelKeeper, long_memory_envs_sampler: hw_experience_replay.MemoryEnvsSampler, layers: list[BasePolicy], layer_i: int):
        self.model_keeper = model_keeper
        self.plta = self.model_keeper.models["PLTA"]
        self.plhw = self.model_keeper.models["PLHW"]
        self.long_memory_envs_sampler = long_memory_envs_sampler
        self.layers = layers
        self.layer_i = layer_i

    def _call_plhw(self, s0_env: env_torch_wrapper.EnvsTensorList, s0_enc: torch.Tensor, targets: torch.Tensor, b: torch.Tensor):
        B = len(targets)
        assert s0_env is None or len(s0_env) == B
        assert s0_enc is None or len(s0_enc) == B
        assert b.shape == (B, 1) and b.dtype == torch.int64

        plta = self.plta
        plhw = self.plhw
        plta.eval()
        plhw.eval()

        if s0_enc is None:
            s0_enc = plta.forward_model_board_normalize(s0_env.get_states_t())

        layer_idx_t = torch.ones(B, dtype=torch.long, device=b.device) * (self.layer_i - 2)

        hw_enc = plhw.forward_model_hw(
            ss=s0_enc,
            ts=targets,
            min_max_01=b,
            layer_idx=layer_idx_t
        )

        ns_enc = plhw.forward_model_hw(
            ss=hw_enc,
            ts=targets,
            min_max_01=b,
            layer_idx=layer_idx_t
        )

        return hw_enc, ns_enc

    def get_expand_items_for_plhw(self, s0_env: env_torch_wrapper.EnvsTensorList, branch_factor: int, device) -> tuple[Optional[env_torch_wrapper.EnvsTensorList], torch.Tensor]:
        long_memory_envs_sampler = self.long_memory_envs_sampler
        assert len(long_memory_envs_sampler.episodes) > 0

        s0_enc = s0_env.get_states_t()

        target_envs = long_memory_envs_sampler.get_random_envs(cnt=branch_factor, also_sample_final=True)
        target_envs.to(s0_enc.device)
        targets_t = target_envs.get_states_t()

        tiled_s0 = s0_env.tile(cnt=len(target_envs))

        b_array = hw_common.get_semi_random_b_array(cnt=branch_factor, device=s0_enc.device)

        targets_t = self.plta.forward_model_board_normalize(targets_t)

        hw_enc, ns_enc = self._call_plhw(s0_env=tiled_s0, s0_enc=None, targets=targets_t, b=b_array)
        return None, ns_enc

    def get_plan_last_enc(self, s0_env: Optional[env_torch_wrapper.EnvsTensorList], s0_enc: torch.Tensor, target: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        B = len(s0_enc)
        assert len(target) == B
        assert len(b) == B and b.dtype == torch.int64

        hw_enc, ns_enc = self._call_plhw(s0_env=None, s0_enc=s0_enc, targets=target, b=b)
        return ns_enc

    def get_plan_envs_to_envs(self, s0: env_torch_wrapper.EnvsTensorList, target: env_torch_wrapper.EnvsTensorList, b: torch.Tensor) -> torch.Tensor:
        B = len(s0)
        assert len(target) == B
        assert b.shape == (B, 1) and b.dtype == torch.int64

        AS = self.plta.AS
        device = b.device
        b_to = hw_common.get_b_array_from_towards_or_away(True, cnt=B, device=device)
        layer_idx_t = torch.ones(B, dtype=torch.long).to(device)
        target.to(device)

        def __get_plan_recursive(li, curr_s0_env, curr_targets_enc, b):
            layer = self.layers[li]
            s0_enc = self.plta.forward_model_board_normalize(curr_s0_env.get_states_t().to(device))

            if isinstance(layer, PolicyHighLevel):
                hw_enc = layer.plhw.forward_model_hw(
                    ss=s0_enc,
                    ts=curr_targets_enc,
                    min_max_01=b,
                    layer_idx=layer_idx_t * (li - 2)
                )
                plan1, hw_envs = __get_plan_recursive(li - 1, curr_s0_env, hw_enc, b=b_to)
                plan2, ns_envs = __get_plan_recursive(li - 1, hw_envs, curr_targets_enc, b=b)
                plan1.extend(plan2)
                return plan1, ns_envs
            
            elif isinstance(layer, PolicyZeroLevel):
                plans = layer.plta.forward_model_target_actions__enc(
                    ss=s0_enc,
                    ts=curr_targets_enc,
                    min_max_01=b)
                plans = torch.argmax(plans, axis=2)
                ns_envs = curr_s0_env.apply_actions__batch(actions_t=plans)
                return [plans], ns_envs

            else:
                raise Exception("Invalid layers structure")

        env_obs_shape = s0.get_envs()[0].get_model_input_s().shape
        ts_enc = target.get_states_t()
        if ts_enc.shape[1:] == env_obs_shape:
            ts_enc = self.plta.forward_model_board_normalize(ts_enc)

        plans_list, _ = __get_plan_recursive(self.layer_i, s0, ts_enc, b)
        plans = hw_common.hstack_plans__torch(plans_list, AS)
        return plans

    def get_behavior_actions(self, s0: env_torch_wrapper.EnvsTensorList, branch_factor: int) -> torch.Tensor:
        assert len(s0) == 1

        long_memory_envs_sampler = self.long_memory_envs_sampler
        assert len(long_memory_envs_sampler.episodes) > 0

        target_envs = long_memory_envs_sampler.get_random_envs(cnt=branch_factor, also_sample_final=True)

        b = hw_common.get_semi_random_b_array(cnt=len(target_envs), device=s0.get_states_t().device)

        if len(target_envs) > 1:
            s0 = s0.tile(len(target_envs))

        plans = self.get_plan_envs_to_envs(s0, target_envs, b)
        return plans

    def eval(self):
        self.plta.eval()
        self.plhw.eval()

    def __str__(self) -> str:
        return f"Policy{self.layer_i - 1} (layer_idx={self.layer_i})"