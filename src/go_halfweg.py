import pprint
import sys
import numpy as np
import torch
import yaml

import hw_impl.evaluate as evaluate
import helpers
import interactive_player


def main():
    torch.set_num_threads(1)
    torch.autograd.set_detect_anomaly(False)

    if len(sys.argv) != 2:
        print(f"Usage: python -u go_halfweg.py configs/evaluate_boxoban_solve.yaml")
        return
    
    with open(sys.argv[1]) as fin:
        config = yaml.safe_load(fin)

    if config['infra']['log'] == 'tf':
        tensorboard = helpers.TensorboardSummaryWriter()
    else:
        tensorboard = helpers.MemorySummaryWriter()
    print("Tensorboard:", tensorboard.__class__.__name__)

    if config['infra']['device'] is None or config['infra']['device'] == 'cpu':
        device = 'cpu'
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Train device: {device}")

    pprint.pprint(config)

    if config['cmd'] == 'evaluate':
        evaluate.go_evaluate(config, device)
    elif config['cmd'] == 'interactive_play':
        interactive_player.go_interactive_play(config)

    if isinstance(tensorboard, helpers.MemorySummaryWriter):
        for key, val in tensorboard.points.items():
            m0 = np.min(val) if len(val) > 0 else '-'
            m1 = np.mean(val) if len(val) > 0 else '-'
            m2 = np.max(val) if len(val) > 0 else '-'
            l = val[-1] if len(val) > 0 else '-'
            print(f"Memory tensorboard: `{key}` has {len(val)} points (min, mean, max, last) = {m0, m1, m2, l}")

if __name__ == "__main__":
    main()


