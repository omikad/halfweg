import numpy as np
import torch
from numba import jit


def get_semi_random_b_array(cnt: int, device) -> torch.Tensor:
    arr = torch.ones((cnt, 1), dtype=torch.int64, device=device)
    arr[cnt // 2:, :] = 0
    if np.random.randint(2) == 1:
        arr = 1 - arr
    return arr


def get_b_array_from_towards_or_away(towards_or_away: bool, cnt: int = 1, device=None) -> torch.Tensor:
    return torch.ones((cnt, 1), dtype=torch.int64, device=device) * (0 if towards_or_away else 1)


@jit(nopython=True)
def trim_plan(plan: np.ndarray, AS: int) -> np.ndarray:
    assert len(plan.shape) == 1
    if len(plan) == 0:
        return plan
    i = np.argmax(plan == AS)
    if plan[i] == AS:
        return plan[:i]
    return plan


@jit(nopython=True)
def trim_plans_2d(plans: np.ndarray, AS: int) -> list[np.ndarray]:
    assert len(plans.shape) == 2
    return [trim_plan(row, AS=AS) for row in plans]


def hstack_plans__numpy(plans_list: list[np.ndarray], AS: int) -> np.ndarray:
    for plans in plans_list:
        assert len(plans.shape) == 2
        assert plans.shape[0] == plans_list[0].shape[0]

    B = plans_list[0].shape[0]

    plan_list2d = [trim_plans_2d(plans, AS) for plans in plans_list]

    max_width = max((sum(len(plan[i]) for plan in plan_list2d) for i in range(B)))

    result = np.ones((B, max_width), dtype=np.int64) * AS

    for i in range(B):
        start = 0
        for plans in plan_list2d:
            plan = plans[i]
            result[i, start : start + len(plan)] = plan
            start += len(plan)

    return result


def hstack_plans__torch(plans_list: list[torch.Tensor], AS: int) -> np.ndarray:
    result = hstack_plans__numpy([plans.cpu().numpy() for plans in plans_list], AS)
    return torch.as_tensor(result)


def vstack_plans__numpy(plans_list: list[np.ndarray], AS: int) -> np.ndarray:
    for plans in plans_list:
        assert len(plans.shape) == 2

    total_len = sum(plans.shape[0] for plans in plans_list)
    total_width = max(plans.shape[1] for plans in plans_list)

    result = np.ones((total_len, total_width), dtype=np.int64) * AS
    start = 0
    for plans in plans_list:
        result[start : start + plans.shape[0], : plans.shape[1]] = plans
        start += plans.shape[0]

    return result