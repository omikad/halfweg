__all__ = [
    "create_envs_manager",
    "create_model",
]


from environments import env_base


def create_envs_manager(env_config: dict) -> env_base.BaseEnvsManager:
    env_name = env_config['name']

    if env_name == 'boxoban':
        from environments.sokoban import sokoban_levels_manager
        levels_path = env_config['levels']
        levman = sokoban_levels_manager.SokobanLevelsManager()
        levman.setup_levels_generator__from_boxoban(levels_path)
        return levman

    elif env_name == 'sokoban':
        from environments.sokoban import sokoban_levels_manager
        levels_path = env_config['levels']
        maxsize = env_config.get('maxsize', '10,10')
        randomize = env_config.get('randomize', False)
        levman = sokoban_levels_manager.SokobanLevelsManager()
        levman.setup_levels_generator__from_levels_filepath(levels_path, maxsize, randomize)
        return levman

    raise Exception(f"Unknown env name `{env_name}`")


def create_model(model_name: str):
    if model_name.startswith('Sokoban_') or model_name.startswith('Boxoban_'):
        import environments.sokoban.sokoban_nn
        return getattr(environments.sokoban.sokoban_nn, model_name)()

    raise Exception(f"Unknown model name `{model_name}`")