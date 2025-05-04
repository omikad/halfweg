import torch

import helpers


AS = 4   # (left, up, right, down)


############################################################################################ Board 10x10

v1_1010_PREDICT_STEPS = 4
v1_1010_WIDTH = 64
v1_1010_INPUT_SIZE = (4, 10, 10)
v1_1010_PLHW_LAYERS = 5


class Sokoban_PLTA_v1_1010(torch.nn.Module):  # 383124 params
    def __init__(self):
        super().__init__()

        self.AS = AS
        self.PREDICT_STEPS = v1_1010_PREDICT_STEPS
        self.INPUT_SIZE = v1_1010_INPUT_SIZE

        self.INPUT_W = self.INPUT_SIZE[1] - 2
        self.INPUT_H = self.INPUT_SIZE[2] - 2
       
        WIDTH = v1_1010_WIDTH
        assert WIDTH % 4 == 0

        self.model_path_construct = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=9, out_channels=WIDTH, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(),
            helpers.ResBlock(planes=WIDTH),
            helpers.ResBlock(planes=WIDTH),
            helpers.ResBlock(planes=WIDTH),
            helpers.ResBlock(planes=WIDTH),
            torch.nn.Flatten(),
            torch.nn.Linear(WIDTH * 64, self.PREDICT_STEPS * (AS + 1)),
        )

    def forward_model_board_normalize(self, ss):
        ss = ss.float()
        ss = ss[:, :, 1:-1, 1:-1]
        return ss

    def forward_model_target_actions__env_board(self, ss, ts, min_max_01):
        ss_normalized = self.forward_model_board_normalize(ss)
        ts_normalized = self.forward_model_board_normalize(ts)
        plan = self.forward_model_target_actions__enc(ss_normalized, ts_normalized, min_max_01)
        return plan

    def forward_model_target_actions__enc(self, ss, ts, min_max_01):

        min_max_01 = torch.tile(min_max_01.view(-1, 1, 1, 1), dims=(1, 1, self.INPUT_W, self.INPUT_H)).float()

        inp = torch.cat([ss, ts, min_max_01], dim=1)

        plan = self.model_path_construct(inp)

        plan = plan.view(plan.shape[0], self.PREDICT_STEPS, AS + 1)

        return plan


class Sokoban_PLHW_v1_1010(torch.nn.Module):  # 306372 params
    def __init__(self):
        super().__init__()

        self.AS = AS
        self.PREDICT_STEPS = v1_1010_PREDICT_STEPS
        self.INPUT_SIZE = v1_1010_INPUT_SIZE
        self.PLHW_LAYERS = v1_1010_PLHW_LAYERS

        self.INPUT_W = self.INPUT_SIZE[1] - 2
        self.INPUT_H = self.INPUT_SIZE[2] - 2

        WIDTH = v1_1010_WIDTH
        INPUT_CHANNELS = 4 + 4 + 1 + self.PLHW_LAYERS    # ss channels + ts channels + min_max_01 + ohe_layer_idx

        self.model_hw = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=INPUT_CHANNELS, out_channels=WIDTH, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(),
            helpers.ResBlock(planes=WIDTH),
            helpers.ResBlock(planes=WIDTH),
            helpers.ResBlock(planes=WIDTH),
            helpers.ResBlock(planes=WIDTH),
            torch.nn.Conv2d(in_channels=WIDTH, out_channels=4, kernel_size=3, stride=1, padding=1),
        )

    def forward_model_hw(self, ss, ts, min_max_01, layer_idx):
        # print(f"[forward_model_hw] ss {ss.shape} {ss.dtype}")
        # print(f"[forward_model_hw] ts {ts.shape} {ts.dtype}")
        # print(f"[forward_model_hw] min_max_01 {min_max_01.shape} {min_max_01.dtype}")
        # print(f"[forward_model_hw] layer_idx {layer_idx.shape} {layer_idx.dtype}")

        min_max_01 = torch.tile(min_max_01.view(-1, 1, 1, 1), dims=(1, 1, self.INPUT_W, self.INPUT_H)).float()
        # print(f"[forward_model_hw] min_max_01 {min_max_01.shape} {min_max_01.dtype}")

        layer_idx = torch.nn.functional.one_hot(layer_idx, num_classes=self.PLHW_LAYERS)
        # print(f"[forward_model_hw] layer_idx {layer_idx.shape} {layer_idx.dtype}")   # (B, hiera_size) torch.int64

        layer_idx = torch.tile(layer_idx.float().view(ss.shape[0], self.PLHW_LAYERS, 1, 1), dims=(1, 1, self.INPUT_W, self.INPUT_H))
        # print(f"[forward_model_hw] layer_idx {layer_idx.shape} {layer_idx.dtype}")   # (B, hiera_size) torch.int64

        inp = torch.cat([ss, ts, min_max_01, layer_idx], dim=1)
        # print(f"[forward_model_hw] inp {inp.shape} {inp.dtype}") 

        delta_hw = self.model_hw(inp)
        # print(f"[forward_model_hw] delta_hw {delta_hw.shape} {delta_hw.dtype}")

        hw = ss + delta_hw

        return hw
    



v2_1010_PREDICT_STEPS = 4
v2_1010_WIDTH = 128
v2_1010_INPUT_SIZE = (4, 10, 10)
v2_1010_PLHW_LAYERS = 5


class Sokoban_PLTA_v2_1010(torch.nn.Module):  # 3128596 params
    def __init__(self):
        super().__init__()

        self.AS = AS
        self.PREDICT_STEPS = v2_1010_PREDICT_STEPS
        self.INPUT_SIZE = v2_1010_INPUT_SIZE

        self.INPUT_W = self.INPUT_SIZE[1] - 2
        self.INPUT_H = self.INPUT_SIZE[2] - 2
       
        WIDTH = v2_1010_WIDTH
        assert WIDTH % 4 == 0

        self.model_path_construct = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=9, out_channels=WIDTH, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(),
            helpers.ResBlock(planes=WIDTH),
            helpers.ResBlock(planes=WIDTH),
            helpers.ResBlock(planes=WIDTH),
            helpers.ResBlock(planes=WIDTH),
            helpers.ResBlock(planes=WIDTH),

            helpers.ResBlock(planes=WIDTH),
            helpers.ResBlock(planes=WIDTH),
            helpers.ResBlock(planes=WIDTH),
            helpers.ResBlock(planes=WIDTH),
            helpers.ResBlock(planes=WIDTH),
            torch.nn.Flatten(),
            torch.nn.Linear(WIDTH * 64, self.PREDICT_STEPS * (AS + 1)),
        )

    def forward_model_board_normalize(self, ss):
        ss = ss.float()
        ss = ss[:, :, 1:-1, 1:-1]
        return ss

    def forward_model_target_actions__env_board(self, ss, ts, min_max_01):
        ss_normalized = self.forward_model_board_normalize(ss)
        ts_normalized = self.forward_model_board_normalize(ts)
        plan = self.forward_model_target_actions__enc(ss_normalized, ts_normalized, min_max_01)
        return plan

    def forward_model_target_actions__enc(self, ss, ts, min_max_01):

        min_max_01 = torch.tile(min_max_01.view(-1, 1, 1, 1), dims=(1, 1, self.INPUT_W, self.INPUT_H)).float()

        inp = torch.cat([ss, ts, min_max_01], dim=1)

        plan = self.model_path_construct(inp)

        plan = plan.view(plan.shape[0], self.PREDICT_STEPS, AS + 1)

        return plan


class Sokoban_PLHW_v2_1010(torch.nn.Module):  # 3565956 params
    def __init__(self):
        super().__init__()

        self.AS = AS
        self.PREDICT_STEPS = v2_1010_PREDICT_STEPS
        self.INPUT_SIZE = v2_1010_INPUT_SIZE
        self.PLHW_LAYERS = v2_1010_PLHW_LAYERS

        self.INPUT_W = self.INPUT_SIZE[1] - 2
        self.INPUT_H = self.INPUT_SIZE[2] - 2

        WIDTH = v2_1010_WIDTH
        INPUT_CHANNELS = 4 + 4 + 1 + self.PLHW_LAYERS    # ss channels + ts channels + min_max_01 + ohe_layer_idx

        self.model_hw = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=INPUT_CHANNELS, out_channels=WIDTH, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(),
            helpers.ResBlock(planes=WIDTH),
            helpers.ResBlock(planes=WIDTH),
            helpers.ResBlock(planes=WIDTH),
            helpers.ResBlock(planes=WIDTH),
            helpers.ResBlock(planes=WIDTH),

            helpers.ResBlock(planes=WIDTH),
            helpers.ResBlock(planes=WIDTH),
            helpers.ResBlock(planes=WIDTH),
            helpers.ResBlock(planes=WIDTH),
            helpers.ResBlock(planes=WIDTH),

            helpers.ResBlock(planes=WIDTH),
            helpers.ResBlock(planes=WIDTH),
            torch.nn.Conv2d(in_channels=WIDTH, out_channels=4, kernel_size=3, stride=1, padding=1),
        )

    def forward_model_hw(self, ss, ts, min_max_01, layer_idx):
        min_max_01 = torch.tile(min_max_01.view(-1, 1, 1, 1), dims=(1, 1, self.INPUT_W, self.INPUT_H)).float()

        layer_idx = torch.nn.functional.one_hot(layer_idx, num_classes=self.PLHW_LAYERS)

        layer_idx = torch.tile(layer_idx.float().view(ss.shape[0], self.PLHW_LAYERS, 1, 1), dims=(1, 1, self.INPUT_W, self.INPUT_H))

        inp = torch.cat([ss, ts, min_max_01, layer_idx], dim=1)

        delta_hw = self.model_hw(inp)

        hw = ss + delta_hw

        return hw
    


v3_1010_PREDICT_STEPS = 4
v3_1010_WIDTH = 64
v3_1010_INPUT_SIZE = (4, 10, 10)
v3_1010_PLHW_LAYERS = 5


class Sokoban_PLTA_v3_1010(torch.nn.Module):  # 161172 params
    def __init__(self):
        super().__init__()

        self.AS = AS
        self.PREDICT_STEPS = v3_1010_PREDICT_STEPS
        self.INPUT_SIZE = v3_1010_INPUT_SIZE

        self.INPUT_W = self.INPUT_SIZE[1] - 2
        self.INPUT_H = self.INPUT_SIZE[2] - 2
       
        WIDTH = v3_1010_WIDTH
        assert WIDTH % 4 == 0

        self.model_path_construct = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=9, out_channels=WIDTH, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(),
            helpers.ResBlock(planes=WIDTH),
            torch.nn.Flatten(),
            torch.nn.Linear(WIDTH * 64, self.PREDICT_STEPS * (AS + 1)),
        )

    def forward_model_board_normalize(self, ss):
        ss = ss.float()
        ss = ss[:, :, 1:-1, 1:-1]
        return ss

    def forward_model_target_actions__env_board(self, ss, ts, min_max_01):
        ss_normalized = self.forward_model_board_normalize(ss)
        ts_normalized = self.forward_model_board_normalize(ts)
        plan = self.forward_model_target_actions__enc(ss_normalized, ts_normalized, min_max_01)
        return plan

    def forward_model_target_actions__enc(self, ss, ts, min_max_01):
        min_max_01 = torch.tile(min_max_01.view(-1, 1, 1, 1), dims=(1, 1, self.INPUT_W, self.INPUT_H)).float()

        inp = torch.cat([ss, ts, min_max_01], dim=1)

        plan = self.model_path_construct(inp)

        plan = plan.view(plan.shape[0], self.PREDICT_STEPS, AS + 1)

        return plan


class Sokoban_PLHW_v3_1010(torch.nn.Module):  # xxx params
    def __init__(self):
        super().__init__()

        self.AS = AS
        self.PREDICT_STEPS = v3_1010_PREDICT_STEPS
        self.INPUT_SIZE = v3_1010_INPUT_SIZE
        self.PLHW_LAYERS = v3_1010_PLHW_LAYERS

        self.INPUT_W = self.INPUT_SIZE[1] - 2
        self.INPUT_H = self.INPUT_SIZE[2] - 2

        WIDTH = v3_1010_WIDTH
        INPUT_CHANNELS = 4 + 4 + 1 + self.PLHW_LAYERS    # ss channels + ts channels + min_max_01 + ohe_layer_idx

        self.model_hw = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=INPUT_CHANNELS, out_channels=WIDTH, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(),
            helpers.ResBlock(planes=WIDTH),
            helpers.ResBlock(planes=WIDTH),
            torch.nn.Conv2d(in_channels=WIDTH, out_channels=4, kernel_size=3, stride=1, padding=1),
        )

    def forward_model_hw(self, ss, ts, min_max_01, layer_idx):
        min_max_01 = torch.tile(min_max_01.view(-1, 1, 1, 1), dims=(1, 1, self.INPUT_W, self.INPUT_H)).float()

        layer_idx = torch.nn.functional.one_hot(layer_idx, num_classes=self.PLHW_LAYERS)

        layer_idx = torch.tile(layer_idx.float().view(ss.shape[0], self.PLHW_LAYERS, 1, 1), dims=(1, 1, self.INPUT_W, self.INPUT_H))

        inp = torch.cat([ss, ts, min_max_01, layer_idx], dim=1)

        delta_hw = self.model_hw(inp)

        hw = ss + delta_hw

        return hw
    


v4_1010_PREDICT_STEPS = 4
v4_1010_WIDTH = 96
v4_1010_INPUT_SIZE = (4, 10, 10)
v4_1010_PLHW_LAYERS = 5


class Sokoban_PLTA_v4_1010(torch.nn.Module):  # 1128404 params
    def __init__(self):
        super().__init__()

        self.AS = AS
        self.PREDICT_STEPS = v4_1010_PREDICT_STEPS
        self.INPUT_SIZE = v4_1010_INPUT_SIZE

        self.INPUT_W = self.INPUT_SIZE[1] - 2
        self.INPUT_H = self.INPUT_SIZE[2] - 2
       
        WIDTH = v4_1010_WIDTH
        assert WIDTH % 4 == 0

        self.model_path_construct = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=9, out_channels=WIDTH, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(),
            helpers.ResBlock(planes=WIDTH),
            helpers.ResBlock(planes=WIDTH),
            helpers.ResBlock(planes=WIDTH),
            helpers.ResBlock(planes=WIDTH),
            helpers.ResBlock(planes=WIDTH),
            helpers.ResBlock(planes=WIDTH),
            torch.nn.Flatten(),
            torch.nn.Linear(WIDTH * 64, self.PREDICT_STEPS * (AS + 1)),
        )

    def forward_model_board_normalize(self, ss):
        ss = ss.float()
        ss = ss[:, :, 1:-1, 1:-1]
        return ss

    def forward_model_target_actions__env_board(self, ss, ts, min_max_01):
        ss_normalized = self.forward_model_board_normalize(ss)
        ts_normalized = self.forward_model_board_normalize(ts)
        plan = self.forward_model_target_actions__enc(ss_normalized, ts_normalized, min_max_01)
        return plan

    def forward_model_target_actions__enc(self, ss, ts, min_max_01):
        min_max_01 = torch.tile(min_max_01.view(-1, 1, 1, 1), dims=(1, 1, self.INPUT_W, self.INPUT_H)).float()

        inp = torch.cat([ss, ts, min_max_01], dim=1)

        plan = self.model_path_construct(inp)

        plan = plan.view(plan.shape[0], self.PREDICT_STEPS, AS + 1)

        return plan


class Sokoban_PLHW_v4_1010(torch.nn.Module):  # 1013284 params
    def __init__(self):
        super().__init__()

        self.AS = AS
        self.PREDICT_STEPS = v4_1010_PREDICT_STEPS
        self.INPUT_SIZE = v4_1010_INPUT_SIZE
        self.PLHW_LAYERS = v4_1010_PLHW_LAYERS

        self.INPUT_W = self.INPUT_SIZE[1] - 2
        self.INPUT_H = self.INPUT_SIZE[2] - 2

        WIDTH = v4_1010_WIDTH
        INPUT_CHANNELS = 4 + 4 + 1 + self.PLHW_LAYERS    # ss channels + ts channels + min_max_01 + ohe_layer_idx

        self.model_hw = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=INPUT_CHANNELS, out_channels=WIDTH, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(),
            helpers.ResBlock(planes=WIDTH),
            helpers.ResBlock(planes=WIDTH),
            helpers.ResBlock(planes=WIDTH),
            helpers.ResBlock(planes=WIDTH),
            helpers.ResBlock(planes=WIDTH),
            helpers.ResBlock(planes=WIDTH),
            torch.nn.Conv2d(in_channels=WIDTH, out_channels=4, kernel_size=3, stride=1, padding=1),
        )

    def forward_model_hw(self, ss, ts, min_max_01, layer_idx):
        min_max_01 = torch.tile(min_max_01.view(-1, 1, 1, 1), dims=(1, 1, self.INPUT_W, self.INPUT_H)).float()

        layer_idx = torch.nn.functional.one_hot(layer_idx, num_classes=self.PLHW_LAYERS)

        layer_idx = torch.tile(layer_idx.float().view(ss.shape[0], self.PLHW_LAYERS, 1, 1), dims=(1, 1, self.INPUT_W, self.INPUT_H))

        inp = torch.cat([ss, ts, min_max_01, layer_idx], dim=1)

        delta_hw = self.model_hw(inp)

        hw = ss + delta_hw

        return hw