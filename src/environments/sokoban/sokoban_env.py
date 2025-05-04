import numpy as np
from numba import jit
from typing import Optional

from environments import env_base
import helpers


CHANNEL_WALL = 0
CHANNEL_PLAYER = 1
CHANNEL_BOX = 2
CHANNEL_GOAL = 3

TILE_WALL = 1 << CHANNEL_WALL
TILE_PLAYER = 1 << CHANNEL_PLAYER
TILE_BOX = 1 << CHANNEL_BOX
TILE_GOAL = 1 << CHANNEL_GOAL

DIRECTIONS = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])  # up, right, down, left


@jit(nopython=True)
def _compute_action_mask(board: np.ndarray, action_mask: np.ndarray, player_row: int, player_col: int) -> None:
    _, rows, cols = board.shape
    action_mask.fill(0)

    for di in range(4):
        dr, dc = DIRECTIONS[di]
        nr = player_row + dr
        nc = player_col + dc
        if 0 <= nr < rows and 0 <= nc < cols and board[CHANNEL_WALL, nr, nc] == 0:
            if board[CHANNEL_BOX, nr, nc] == 0:
                action_mask[di] = 1
            else:
                nnr = nr + dr
                nnc = nc + dc
                if 0 <= nnr < rows and 0 <= nnc < cols and (board[CHANNEL_BOX, nnr, nnc] + board[CHANNEL_WALL, nnr, nnc] == 0):
                    action_mask[di] = 1


@jit(nopython=True)
def _step(board: np.ndarray, action_mask: np.ndarray, player_row: int, player_col: int, action: int) -> tuple[int, bool]:
    dr, dc = DIRECTIONS[action]

    board[CHANNEL_PLAYER, player_row, player_col] = 0
    player_row += dr
    player_col += dc
    board[CHANNEL_PLAYER, player_row, player_col] = 1

    if board[CHANNEL_BOX, player_row, player_col] == 1:
        board[CHANNEL_BOX, player_row, player_col] = 0
        nnr = player_row + dr
        nnc = player_col + dc
        board[CHANNEL_BOX, nnr, nnc] = 1

    _compute_action_mask(board, action_mask, player_row, player_col)

    total_boxes = 0
    correct_boxes = 0
    corner_boxes = 0
    for r, c in zip(*np.where(board[CHANNEL_BOX, :, :] == 1)):
        total_boxes += 1
        if board[CHANNEL_GOAL, r, c] == 1:
            correct_boxes += 1
        else:
            for di in range(4):
                dr0, dc0 = DIRECTIONS[di]
                dr1, dc1 = DIRECTIONS[(di + 1) % 4]
                if board[CHANNEL_WALL, r + dr0, c + dc0] + board[CHANNEL_WALL, r + dr1, c + dc1] == 2:
                    corner_boxes += 1

    if total_boxes == correct_boxes:
        return 1, True
    if corner_boxes > 0 or np.sum(action_mask) == 0:
        return 0, True
    return 0, False


def get_square_symmetry_rotation(arr: np.ndarray, sym_idx: int) -> np.ndarray:
    assert len(arr.shape) == 3
    assert arr.shape[1] == arr.shape[2]
    assert 0 <= sym_idx <= 7

    if sym_idx == 0: return arr
    if sym_idx == 1: return np.rot90(arr, k=1, axes=(1, 2))
    if sym_idx == 2: return np.rot90(arr, k=2, axes=(1, 2))
    if sym_idx == 3: return np.rot90(arr, k=3, axes=(1, 2))
    arr = np.flip(arr, axis=1)
    if sym_idx == 4: return arr
    if sym_idx == 5: return np.rot90(arr, k=1, axes=(1, 2))
    if sym_idx == 6: return np.rot90(arr, k=2, axes=(1, 2))
    if sym_idx == 7: return np.rot90(arr, k=3, axes=(1, 2))


class Sokoban(env_base.BaseEnv):
    board: np.ndarray
    player_row: int
    player_col: int
    done: bool
    action_mask: np.ndarray

    def __init__(self, copy_from: Optional["Sokoban"], init_board: Optional[np.ndarray]) -> None:
        if copy_from is not None:
            self.board = np.copy(copy_from.board)
            self.player_row = copy_from.player_row
            self.player_col = copy_from.player_col
            self.done = copy_from.done
            self.action_mask = np.copy(copy_from.action_mask)
        else:
            assert init_board is not None
            assert len(init_board.shape) == 3
            self.board = np.copy(init_board)
            self.action_mask = np.zeros(4, dtype=np.int32)
            self.player_row, self.player_col = np.unravel_index(np.argmax(init_board[CHANNEL_PLAYER, :, :]), shape=init_board.shape[1:])
            self.done = False
            _compute_action_mask(self.board, self.action_mask, self.player_row, self.player_col)

    def get_valid_actions_mask(self):
        return self.action_mask

    def copy(self):
        return Sokoban(copy_from=self, init_board=None)

    def get_symmetrical_envs_count(self) -> int:
        return 1

    def get_symmetrical_env(self, sym_idx: int):
        return self

    def step(self, action: int):
        if self.action_mask[action] != 1 or self.done:
            return 0, self.done
        reward, self.done = _step(self.board, self.action_mask, self.player_row, self.player_col, action)
        dr, dc = DIRECTIONS[action]
        self.player_row += dr
        self.player_col += dc
        return reward, self.done
    
    def get_model_input_s(self) -> np.ndarray:
        return self.board.copy()

    def get_target_states(self) -> np.ndarray:
        targets = []
        board = self.board

        for r in range(board.shape[1]):
            for c in range(board.shape[2]):
                if board[CHANNEL_WALL, r, c] == 0 and board[CHANNEL_GOAL, r, c] == 0:
                    state = board.copy()
                    state[CHANNEL_BOX, :, :] = state[CHANNEL_GOAL, :, :]  # All boxes should be on goal states
                    state[CHANNEL_PLAYER, :, :] = 0
                    state[CHANNEL_PLAYER, r, c] = 1   # Put player on all possible positions
                    state = np.expand_dims(state, axis=0)
                    targets.append(state)
        targets = np.concatenate(targets, axis=0)
        return targets

    def all_boxes_correct(self) -> bool:
        for r, c in zip(*np.where(self.board[CHANNEL_BOX, :, :] == 1)):
            if self.board[CHANNEL_GOAL, r, c] != 1:
                return False
        return True

    def render_ascii(self):
        board = self.board

        print_tabs = [[] for _ in range(2)]

        print_tabs[0].append(f"Board size {board.shape[1:]}")

        # render board in tab 0
        for row in range(board.shape[1]):
            row_content = []
            for col in range(board.shape[2]):
                channels_cnt = np.sum(board[:, row, col])

                sym = ''
                if row == self.player_row and col == self.player_col:
                    if channels_cnt == 1:
                        sym = "\u2654"     # WHITE CHESS KING
                    elif board[CHANNEL_GOAL, row, col] == 1:
                        sym = "\u265A"     # BLACK CHESS KING
                elif channels_cnt == 0:
                    sym = ' '
                elif channels_cnt == 1:
                    if board[CHANNEL_WALL, row, col] == 1:
                        sym = "\u2588"     # SYM_FULL_BLOCK
                    elif board[CHANNEL_PLAYER, row, col] == 1:
                        sym = ' '
                    elif board[CHANNEL_BOX, row, col] == 1:
                        sym = "\u2610"     # BALLOT BOX
                    elif board[CHANNEL_GOAL, row, col] == 1:
                        sym = "."
                elif channels_cnt == 2 and board[CHANNEL_PLAYER, row, col] == 1 and board[CHANNEL_GOAL, row, col] == 1:
                    sym = '.'
                elif channels_cnt == 2 and board[CHANNEL_BOX, row, col] == 1 and board[CHANNEL_GOAL, row, col] == 1:
                    sym = "\u2612"     # BALLOT BOX WITH X

                if sym == '':
                    raise Exception(f"Incorrect board: {board[:, row, col]}, row, col = {row, col}, player row, col = {self.player_row, self.player_col}")

                row_content.append(sym)

            print_tabs[0].append('   ' + ''.join(row_content))

        # render move directions in tab 1
        print_tabs[1].append(f"Done: {self.done}")
        for di, name in enumerate(["up", "right", "down", "left"]):
            if self.action_mask[di] == 1:
                print_tabs[1].append(f"Move {name}: {di}")

        helpers.print_tabs_content(print_tabs)