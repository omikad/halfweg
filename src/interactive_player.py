import torch
import numpy as np
import traceback

import environments
import helpers


@torch.no_grad()
def go_interactive_play(config: dict):
    envman = environments.create_envs_manager(config['env'])
    env = envman.create_env()
    step_i = 0

    while True:
        print()
        env.render_ascii()
        print(f"* Step {step_i}. Valid actions: {' '.join(map(str, np.where(np.array(env.get_valid_actions_mask()) == 1)[0]))}")
        print(f"Type your action, or type 'r' for random move, or type 'd' to see debug information")

        try:
            inp = input().strip().lower()
            if inp == 'r':
                action = env.get_random_action()
                print(f"Random move {action}")
            elif inp == 'd':
                enc = env.get_model_input_s()
                helpers.print_encoding("Model input", enc)
                continue
            else:
                action = int(inp)

            reward, done = env.step(action)
            print(f"Reward {reward}, done {done}")

        except Exception:
            traceback.print_exc()
            continue

        step_i += 1
        if done:
            env.render_ascii()
            break