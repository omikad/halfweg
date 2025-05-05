from collections import Counter, defaultdict
import resource
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


#################### Logging

class TensorboardSummaryWriter(SummaryWriter):
    """
    Wrapper around tensorboard to add points one by one
    """
    def __init__(self):
        super().__init__()
        self.points_cnt = Counter()
        self.figure_cnt = Counter()

    def append_scalar(self, name, value):
        step = self.points_cnt[name]
        self.points_cnt[name] += 1
        self.add_scalar(name, value, step)

    def add_figure(self, name, plt_gcf):
        step = self.figure_cnt[name]
        self.figure_cnt[name] += 1
        super().add_figure(name, plt_gcf, global_step=step)


class MemorySummaryWriter:
    def __init__(self):
        self.points = defaultdict(list)

    def append_scalar(self, name, value):
        self.points[name].append(value)

    def add_figure(self, name, plt_gcf):
        pass


#################### Pytorch

# https://discuss.pytorch.org/t/moving-optimizer-from-cpu-to-gpu/96068/2
def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


# Got from https://github.com/geochri/AlphaZero_Chess/blob/master/src/alpha_net.py
class ResBlock(torch.nn.Module):
    def __init__(self, planes, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = torch.nn.functional.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = torch.nn.functional.relu(out)
        return out


#################### Debug print

def print_tabs_content(print_tabs: list[list[str]]):
    print_tab_sizes = [max([len(line) for line in tab] + [0]) for tab in print_tabs]
    for li in range(max([ len(tab) for tab in print_tabs ])):
        content = []
        for ti, tab in enumerate(print_tabs):
            line = tab[li] if li < len(tab) else ''
            content.append(line + ' ' * (print_tab_sizes[ti] - len(line)))

        print('   '.join(content))


def print_encoding(name: str, inp: np.ndarray):
    print(f"Array {name} with shape {inp.shape}")

    if len(inp.shape) == 3:
        binary_channels = [chi for chi in range(inp.shape[0]) if np.sum(inp[chi, :, :] == 0) + np.sum(inp[chi, :, :] == 1) == inp.shape[1] * inp.shape[2]]

        for chi in range(inp.shape[0]):
            if chi not in binary_channels:
                print(f"Not binary channel {chi}:")
                print(inp[chi, :, :])

        if len(binary_channels) > 0:
            print("Binary channels content:")
            print_tabs = [[] for _ in range(inp.shape[2])]
            for row in range(inp.shape[1]):
                for col in range(inp.shape[2]):
                    if np.sum(inp[binary_channels, row, col]) == 0:
                        print_tabs[col].append('---')
                    else:
                        print_tabs[col].append(','.join([ str(chi) for chi in range(inp.shape[0]) if chi in binary_channels and inp[chi, row, col] == 1 ]))

            print_tabs_content(print_tabs)

    else:
        print(inp)



#################### Measure footprint

def get_mem_size():
    usage=resource.getrusage(resource.RUSAGE_SELF)
    return usage[2]


class UsageCounter:
    """
    Measure time/memory between calls
    """
    def __init__(self):
        self.names = []
        self.times = []
        self.mems = []
        self.checkpoint("")

    def checkpoint(self, name):
        self.names.append(name)
        self.times.append(time.time())
        self.mems.append(get_mem_size())

    def print_stats(self):
        checkpoint_times = defaultdict(list)
        checkpoint_mems = defaultdict(list)

        for i in range(1, len(self.names)):
            name = self.names[i]
            timedelta = self.times[i] - self.times[i - 1]
            memdelta = self.mems[i] - self.mems[i - 1]

            checkpoint_times[name].append(timedelta)
            checkpoint_mems[name].append(memdelta)

        def __r(x):
            return round(x, 5)

        print_tabs = [[] for _ in range(4)]
        for name in checkpoint_times:
            times, mems = checkpoint_times[name], checkpoint_mems[name]

            if len(times) == 1:
                print_tabs[0].append(name)
                print_tabs[1].append(f"{__r(np.sum(times)):>10} sec")
                print_tabs[2].append(f"{__r(np.sum(mems)):>10} kb")

            else:
                print_tabs[0].append(f"{name} time spent")
                print_tabs[1].append(f"{__r(np.sum(times)):>10} sec")
                print_tabs[2].append('')
                print_tabs[3].append(f"Hits {len(times)}; mean {__r(np.mean(times))}; min std max = {__r(np.min(times))} {__r(np.std(times))} {__r(np.max(times))}")

                print_tabs[0].append(f"{name} mem alloc")
                print_tabs[1].append('')
                print_tabs[2].append(f"{__r(np.sum(mems)):>10} kb")
                print_tabs[3].append(f"Hits {len(mems)}; mean {__r(np.mean(mems))}; min std max = {__r(np.min(mems))} {__r(np.std(mems))} {__r(np.max(mems))}")

        print_tabs_content(print_tabs)
