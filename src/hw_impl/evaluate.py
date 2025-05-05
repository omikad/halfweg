import pprint

import numpy as np
import torch
import environments
from environments import env_base
import helpers
from hw_impl import env_torch_wrapper, hw_common, hw_experience_replay, hw_policies, model_mgmt


def get_policy(policy_name: str, model_keeper: model_mgmt.ModelKeeper, long_memory_envs_sampler: hw_experience_replay.MemoryEnvsSampler) -> hw_policies.BasePolicy:
    plta = model_keeper.models['PLTA']

    layers = []
    layers.append(hw_policies.FixedFullScanPolicy(plta))
    if policy_name == 'fixed_full_scan':
        return layers[-1]

    layers.append(hw_policies.PolicyZeroLevel(plta, layers))
    if policy_name == 'PL0':
        return layers[-1]

    if policy_name.startswith('PL'):
        pl_i = int(policy_name.replace('PL', ''))
        for i in range(1, pl_i + 1):
            planner_layer = hw_policies.PolicyHighLevel(model_keeper, long_memory_envs_sampler, layers, layer_i=i + 1)
            layers.append(planner_layer)
        return layers[-1]

    else:
        raise Exception(f"Planner '{policy_name}' not found")


def get_policies(model_keeper: model_mgmt.ModelKeeper, config_policy_name: str) -> list[str]:
    if config_policy_name is None or config_policy_name == 'all':
        model_states = model_keeper.models['PLHW']
        result = ['PL0']
        for i in range(model_states.PLHW_LAYERS):
            result.append(f"PL{i+1}")

    elif config_policy_name == 'last':
        model_states = model_keeper.models['PLHW']
        result = [f"PL{model_states.PLHW_LAYERS}"]

    else:
        result = [config_policy_name]

    return result


def _solve__one_shot(
        device,
        start_envs: env_torch_wrapper.EnvsTensorList,
        targets: list[torch.Tensor],
        policy: hw_policies.BasePolicy,
        towards_or_away: bool) -> list[list[np.ndarray]]:
    assert len(targets) == len(start_envs)

    plans = []

    for game_i in range(len(targets)):
        curr_targets = targets[game_i]

        target_env = env_torch_wrapper.EnvsTensorList(states_t=curr_targets)

        b = hw_common.get_b_array_from_towards_or_away(towards_or_away, cnt=len(curr_targets), device=device)

        curr_s0_env = env_torch_wrapper.EnvsTensorList(envs=[start_envs.envs[game_i]])

        curr_plans = policy.get_plan_envs_to_envs(s0=curr_s0_env.tile(len(curr_targets)), target=target_env, b=b)
        curr_plans = list(curr_plans.cpu().numpy())

        plans.append(curr_plans)

    return plans


@torch.no_grad()
def validate_puzzle_solving__impl(
        config: dict,
        method: str,
        device,
        envs_manager: env_base.BaseEnvsManager,
        model_keeper: model_mgmt.ModelKeeper,
        n_games_to_solve: int,
        policies: list[hw_policies.BasePolicy],
        towards_or_away_array: bool,
        tensorboard):
    model_keeper.eval()

    games_to_solve = []
    for game_i in range(n_games_to_solve):
        env_key, start_env = envs_manager.create_env_with_key()
        games_to_solve.append((env_key, start_env, []))

    targets_np = [start_env.get_target_states() for _, start_env, _ in games_to_solve]
    if config['evaluate']['targets'] == 'random':
        for game_i in range(len(targets_np)):
            ti = np.random.randint(len(targets_np[game_i]))
            targets_np[game_i] = targets_np[game_i][ti:ti+1, ...]
    targets_t = [torch.as_tensor(target_np, dtype=torch.float32, device=device) for target_np in targets_np]

    start_envs = env_torch_wrapper.EnvsTensorList(envs=[start_env.copy() for _, start_env, _ in games_to_solve])
    start_envs.to(device)

    AS = model_keeper.models['PLTA'].AS

    for policy in policies:
        stat_navigation_mse = dict()

        for towards_or_away in towards_or_away_array:

            stat_proposed_plan_lengths = []
            stat_solved_plan_lengths = []
            stat_n_solutions = []
            stat_mse = []
            stat_solved = []

            if method == 'one_shot':
                plans = _solve__one_shot(device, start_envs, targets_t, policy, towards_or_away)
            else:
                raise Exception(f"Unknown validation method `{method}`")

            for game_i in range(len(games_to_solve)):
                curr_plans = plans[game_i]

                stat_n_solutions.append(len(curr_plans))
                solution_plan = None

                for curr_plan in curr_plans:
                    copy_env = start_envs.envs[game_i].copy()
                    reward, done = copy_env.play_plan_1d(curr_plan, AS)

                    stat_proposed_plan_lengths.append(len(curr_plan))
                    stat_mse.append(np.sum(np.abs(np.expand_dims(copy_env.get_model_input_s(), 0) - targets_np[game_i])) / len(targets_np[game_i]))

                    if reward == 1:
                        solution_plan = curr_plan.copy()

                if solution_plan is not None:
                    stat_solved_plan_lengths.append(len(solution_plan))
                    stat_solved.append(1)
                else:
                    stat_solved.append(0)

            stat_navigation_mse[towards_or_away] = np.mean(stat_mse)

            validation_result = {
                'method': method,
                'towards_or_away': towards_or_away,
                'policy': str(policy),
                'solved_mean': np.mean(stat_solved),
                'mse_mean': np.mean(stat_mse),
                'games_cnt': len(stat_solved),
                'proposed_plan_length_mean': np.mean(stat_proposed_plan_lengths),
                'solved_plan_length_mean': np.mean(stat_solved_plan_lengths) if stat_solved_plan_lengths else 'None',
            }
            pprint.pprint(validation_result, width=10000, sort_dicts=False)

            if tensorboard is not None and towards_or_away:
                tensorboard.append_scalar(f"{str(policy)} solved mean", np.mean(stat_solved))

        if tensorboard is not None and True in stat_navigation_mse and False in stat_navigation_mse:
            tensorboard.append_scalar(f"{str(policy)} navigation spread", stat_navigation_mse[False] - stat_navigation_mse[True])


def go_evaluate(config, device):
    usage = helpers.UsageCounter()

    model_keeper = model_mgmt.ModelKeeper(config)
    model_keeper.to(device)
    envs_sampler = hw_experience_replay.MemoryEnvsSampler(model_keeper=model_keeper)
    usage.checkpoint("Checkpoint loaded")

    envs_manager = environments.create_envs_manager(config['env'])
    usage.checkpoint("Envs loaded")

    policies = []
    for policy_name in get_policies(model_keeper, config['evaluate']['policies']):
        planner_layer = get_policy(policy_name, model_keeper, envs_sampler)
        policies.append(planner_layer)
    
    n_games_to_solve = config['evaluate']['n_games_to_solve']
    method = config['evaluate']['method']
    towards_or_away_array = [True, False] if config['evaluate']['towards_or_away'] == 'both' else [False] if config['evaluate']['towards_or_away'] == 'away' else [True]

    usage.checkpoint("Pre-solve")
    validate_puzzle_solving__impl(config, method, device, envs_manager, model_keeper, n_games_to_solve, policies, towards_or_away_array, tensorboard=None)
    usage.checkpoint("Solving")

    usage.print_stats()