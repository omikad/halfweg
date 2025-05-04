import os
import time
import torch

import helpers


class ModelKeeper:
    def __init__(self, config: dict) -> None:
        self.models = dict()
        self.optimizers = dict()
        self.long_memory: helpers.ExperienceReplay = None
        self.layer_targets = dict()
        self.iter_i = 0

        for model_key in ['PLTA', 'PLHW']:
            if model_key in config['model']:
                model = environments.create_model(config['model'][model_key]['class'])
                self.models[model_key] = model
                if 'learning_rate' in config['model'][model_key]:
                    lr = config['model'][model_key]['learning_rate']
                    wd = config['model'][model_key]['weight_decay']
                    self.optimizers[model_key] = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

        if 'checkpoint' in config['model']:
            checkpoint_filename = config['model']['checkpoint']
            self.load_from_checkpoint(checkpoint_filename)
            print(f"Loaded checkpoint content from `{checkpoint_filename}`:")
            if self.long_memory is not None:
                print("[MODEL_KEEPER]", self.long_memory.get_stats())
            print(f"Iteration {self.iter_i}")

        for key, model in self.models.items():
            total_parameters = sum(p.numel() for p in model.parameters())
            print("[MODEL_KEEPER]", f"Models: {key}: {model.__class__.__name__} Params cnt: {total_parameters}")
            # for name, par in model.named_parameters():
            #     print("         ", name, par.numel())

        print("[MODEL_KEEPER] Optimizers:", ', '.join(self.optimizers.keys()) if self.optimizers else 'none')


    def save_checkpoint(self, checkpoints_dir, log_prefix):
        data = { 'models': dict(), 'optimizers': dict() }

        for name, model in self.models.items():
            data['models'][name] = model.state_dict()

        for name, opt in self.optimizers.items():
            data['optimizers'][name] = opt.state_dict()

        data['long_memory'] = self.long_memory
        data['iter_i'] = self.iter_i
        data['layer_targets'] = self.layer_targets

        checkpoint_path = f"checkpoint_{ time.strftime('%Y%m%d-%H%M%S') }.ckpt"
        if checkpoints_dir is not None:
            checkpoint_path = os.path.join(checkpoints_dir, checkpoint_path)
        torch.save(data, checkpoint_path)
        print(f"{log_prefix}. Checkpoint saved to `{checkpoint_path}`")

    def eval(self):
        for _, model in self.models.items():
            model.eval()

    def to(self, device):
        for _, model in self.models.items():
            model.to(device)
        for _, optimizer in self.optimizers.items():
            helpers.optimizer_to(optimizer, device)

    def load_from_checkpoint(self, checkpoint_path: str):
        data = torch.load(checkpoint_path, weights_only=False)

        for name, model_state in data['models'].items():
            if name in self.models:
                self.models[name].load_state_dict(model_state)
        print(f"Models loaded from checkpoints: {', '.join(data['models'].keys())}")

        for name, opt_state in data.get('optimizers', dict()).items():
            if name in self.optimizers:
                self.optimizers[name].load_state_dict(opt_state)
            print(f"Optimizers loaded from checkpoints: {', '.join(data['optimizers'].keys())}")

        self.long_memory = data.get('long_memory', None)
        self.layer_targets = data.get('layer_targets', dict())
        self.iter_i = data.get('iter_i', 0)