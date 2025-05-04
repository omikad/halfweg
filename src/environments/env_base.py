import numpy as np


class BaseEnv:
    def step(self, action: int) -> tuple[int, bool]:
        raise Exception("Need to be implemented in child, return tuple (reward, done)")

    def copy(self) -> "BaseEnv":
        raise Exception("Need to be implemented in child")

    def get_valid_actions_mask(self) -> list:
        raise Exception("Need to be implemented in child, return 1d array of 0 or 1")

    def get_model_input_s(self) -> np.ndarray:
        raise Exception("Need to be implemented in child, return encoded state")

    def get_target_states(self) -> np.ndarray:
        """
        Return array of possible solution states (where agent needs navigate to)
        Output shape should be `(batch, *observation_shape)`
        where `observation_shape` is a shape of what is returned by method `get_model_input_s`
        """
        raise Exception("Need to be implemented in child")

    def render_ascii(self):
        raise Exception("Need to be implemented in child")

    def get_symmetrical_envs_count(self) -> int:
        """
        Return number of symmetries returned by method `get_symmetrical_envs`
        """
        return 1

    def get_symmetrical_env(self, sym_idx: int) -> "BaseEnv":
        """
        Return reflection of environment
        """
        return self

    def get_random_action(self) -> int:
        I = np.where(np.array(self.get_valid_actions_mask()) == 1)[0]
        action = np.random.choice(I)
        return action

    def play_plan_1d(self, plan_1d: np.ndarray, AS: int) -> tuple[float, bool]:
        assert len(plan_1d.shape) == 1

        reward, done = 0, self.done
        for action in plan_1d:
            if done:
                break
            if action == AS:
                break
            reward, done = self.step(action)
        return reward, done

    def copy_and_apply_actions_batch(self, envs: list["BaseEnv"], plan_2d: np.ndarray, AS: int) -> list["BaseEnv"]:
        assert len(plan_2d.shape) == 2
        assert len(plan_2d) == len(envs)

        result = [env.copy() for env in envs]
        for i in range(len(envs)):
            result[i].play_plan_1d(plan_2d[i, :], AS)

        return result


class BaseEnvsManager:
    def create_env_with_key(self) -> tuple[str, BaseEnv]:
        raise Exception("Need to be implemented in child")

    def create_env(self) -> BaseEnv:
        raise Exception("Need to be implemented in child")