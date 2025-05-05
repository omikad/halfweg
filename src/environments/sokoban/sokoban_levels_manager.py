import os
from typing import Optional
import base64
from numba import jit

import numpy as np
from environments import env_base
from environments.sokoban import sokoban_env
import helpers


def encode_2d_array(board: np.ndarray) -> str:
    assert len(board.shape) == 2
    byte_array = bytearray()

    def __flat():
        for r in range(board.shape[0]):
            for c in range(board.shape[1]):
                yield board[r, c]
            yield 15

    i = 0
    for code in __flat():
        if i % 2 == 0:
            byte_array.append(code)
        else:
            byte_array[-1] = byte_array[-1] * 16 + code
        i += 1

    encoded_string = base64.b64encode(byte_array).decode('utf-8')
    return encoded_string


def encode_env(env: sokoban_env.Sokoban) -> str:
    return encode_2d_array(board_3d_to_2d(env.get_model_input_s()))


def decode_env(encoded_board: str) -> sokoban_env.Sokoban:
    byte_array = base64.b64decode(encoded_board)

    board = [[]]
    for code in byte_array:
        part0, part1 = divmod(code, 16)

        for part in [part0, part1]:
            if part == 15:
                board.append([])
            else:
                board[-1].append(part)

    if len(board[-1]) == 0:
        board.pop()

    env = sokoban_env.Sokoban(copy_from=None, init_board=board_2d_to_3d(np.array(board)))
    return env


def create_board_lines(board_np: np.ndarray) -> list[str]:
    res = []
    for row in range(board_np.shape[0]):
        chars = []
        for col in range(board_np.shape[1]):
            ch = board_np[row, col]
            if (ch & (1 << sokoban_env.CHANNEL_WALL)) != 0:
                sym = '#'
            elif (ch & (1 << sokoban_env.CHANNEL_PLAYER)) != 0:
                if (ch & (1 << sokoban_env.CHANNEL_GOAL)) != 0:
                    sym = '+'
                else:
                    sym = '@'
            elif (ch & (1 << sokoban_env.CHANNEL_BOX)) != 0:
                if (ch & (1 << sokoban_env.CHANNEL_GOAL)) != 0:
                    sym = '*'
                else:
                    sym = '$'
            elif (ch & (1 << sokoban_env.CHANNEL_GOAL)) != 0:
                sym = '.'
            else:
                sym = ' '
            chars.append(sym)
        res.append(''.join(chars))
    return res


@jit(nopython=True)
def board_3d_to_2d(board: np.ndarray) -> np.ndarray:
    assert len(board.shape) == 3
    res = np.zeros((board.shape[1], board.shape[2]), dtype=board.dtype)
    res += (board[sokoban_env.CHANNEL_WALL, :, :] > 0) * sokoban_env.TILE_WALL
    res += (board[sokoban_env.CHANNEL_PLAYER, :, :] > 0) * sokoban_env.TILE_PLAYER
    res += (board[sokoban_env.CHANNEL_BOX, :, :] > 0) * sokoban_env.TILE_BOX
    res += (board[sokoban_env.CHANNEL_GOAL, :, :] > 0) * sokoban_env.TILE_GOAL
    return res


@jit(nopython=True)
def board_2d_to_3d(board: np.ndarray) -> np.ndarray:
    assert len(board.shape) == 2
    res = np.zeros((4, *board.shape), dtype=board.dtype)
    for ch in range(4):
        res[ch, ...] = (board & (1 << ch)) != 0
    return res


@jit(nopython=False)
def split_to_level_boards(level_filepath: str, all_boards: np.ndarray):
    assert len(all_boards.shape) == 3
    for player_row, player_col in zip(*np.where(all_boards[sokoban_env.CHANNEL_PLAYER, :, :] > 0)):
        vis = set()
        queue = [(player_row, player_col)]
        vis.add(queue[0])
        qi = 0
        while qi < len(queue):
            row, col = queue[qi]
            for dr, dc in sokoban_env.DIRECTIONS:
                nr, nc = row + dr, col + dc
                if not (0 <= nr < all_boards.shape[1] and 0 <= nc < all_boards.shape[2]):
                    # helpers.print_encoding("Incorrect board surrounding", all_boards[:, max(0, row - 5) : min(all_boards.shape[1], row + 5), :10])
                    raise Exception(f"Filename `{level_filepath}` is incorrect, expected all levels to be surrounded by walls")
                if all_boards[sokoban_env.CHANNEL_WALL, nr, nc] == 0 and (nr, nc) not in vis:
                    vis.add((nr, nc))
                    queue.append((nr, nc))
            qi += 1

        queue = np.array(queue)
        board = all_boards[:, np.min(queue[:, 0] - 1) : np.max(queue[:, 0]) + 2, np.min(queue[:, 1]) - 1 : np.max(queue[:, 1]) + 2]
        board = board_3d_to_2d(board)
        yield board


@jit(nopython=True)
def load_levels_from_lines(lines: list[str]) -> np.ndarray:
    # symbols:
    #   Wall	#
    #   Player	@
    #   Player on goal square	+
    #   Box	$
    #   Box on goal square	*
    #   Goal square	.
    #   Floor	(Space)
    n = len(lines)
    m = 0
    for line in lines:
        m = max(m, len(line))

    all_boards = np.zeros((4, n, m), dtype=np.int32)
    for r, line in enumerate(lines):
        for c, sym in enumerate(line):
            if sym == '#':
                all_boards[sokoban_env.CHANNEL_WALL, r, c] = 1
            elif sym == '@':
                all_boards[sokoban_env.CHANNEL_PLAYER, r, c] = 1
            elif sym == '+':
                all_boards[sokoban_env.CHANNEL_PLAYER, r, c] = 1
                all_boards[sokoban_env.CHANNEL_GOAL, r, c] = 1
            elif sym == '$':
                all_boards[sokoban_env.CHANNEL_BOX, r, c] = 1
            elif sym == '*':
                all_boards[sokoban_env.CHANNEL_BOX, r, c] = 1
                all_boards[sokoban_env.CHANNEL_GOAL, r, c] = 1
            elif sym == '.':
                all_boards[sokoban_env.CHANNEL_GOAL, r, c] = 1
    return all_boards


def load_file_all_boards(level_filepath):
    def __parse(file):
        lines = []
        for line in file:
            line = line.rstrip()
            ok = True
            for player_sym in '+@':
                if player_sym in line:
                    if ('#' not in line) or (line.index(player_sym) < line.index('#')):
                        ok = False
            if ok:
                lines.append(line)
        all_boards = load_levels_from_lines(lines)
        return all_boards

    with open(level_filepath) as file:
        try:
            return __parse(file)
        except Exception as e:
            print("Problem with file", level_filepath)
            print(e)

    print("Retrying")
    with open(level_filepath, encoding='latin-1') as file:
        return __parse(file)


def load_level(level_filepath):
    if ':' in level_filepath:
        level_filepath, level_i = level_filepath.split(':')
        level_i = int(level_i)
    else:
        level_i = 0

    all_boards = load_file_all_boards(level_filepath)
    for li, board in enumerate(split_to_level_boards(level_filepath, all_boards)):
        if li == level_i:
            return board

    raise Exception(f"File `{level_filepath}` has no level {level_i}")


def load_all_levels(root_path: str, maxsize: Optional[str]):
    levels = []

    maxrows, maxcols = 1e9, 1e9
    if maxsize is not None:
        maxrows, maxcols = map(int, maxsize.split(','))

    files = []
    if root_path.endswith('.txt'):
        files.append(root_path)
    else:
        for root, dirs, _files in os.walk(root_path):
            for filename in _files:
                level_filepath = os.path.join(root, filename)
                files.append(level_filepath)

    for level_filepath in files:
        allboards = load_file_all_boards(level_filepath)
        # print(f"File {level_filepath}, loaded into {allboards.shape}")
        for level_i, board_np in enumerate(split_to_level_boards(level_filepath, allboards)):
            if board_np.shape[0] <= maxrows and board_np.shape[0] <= maxcols:
                key = f"{level_filepath}:{level_i}"

                item = {
                    'level_filepath': level_filepath,
                    'level_i': level_i,
                    'board': create_board_lines(board_np)
                }

                levels.append((key, board_np, item))

    return levels


class SokobanLevelsManager(env_base.BaseEnvsManager):
    def __init__(self):
        self.levels_list = None
        self.gen_shape = None
        self.gen_random = None

    def setup_levels_generator__from_levels_filepath(self, levels_path: str, maxsize: str, randomize: bool):
        self.levels_list = []
        maxrows, maxcols = map(int, maxsize.split(','))
        self.gen_shape = (maxrows, maxcols)
        self.gen_random = randomize
        for key, board_np, parsed_item in load_all_levels(levels_path, maxsize):
            self.levels_list.append((key, board_np))

    def setup_levels_generator__from_boxoban(self, levels_path: str):
        self.setup_levels_generator__from_levels_filepath(levels_path, "10,10", True)

    def prepare_env(self, board: np.ndarray) -> env_base.BaseEnv:
        assert len(board.shape) == 2
        if board.shape != self.gen_shape:
            new_board = np.zeros(self.gen_shape, dtype=board.dtype)

            if self.gen_random:
                dr = np.random.randint(self.gen_shape[0] - board.shape[0] + 1)
                dc = np.random.randint(self.gen_shape[1] - board.shape[1] + 1)
            else:
                dr = 0
                dc = 0
            new_board[dr:dr + board.shape[0], dc:dc + board.shape[1]] = board
            board = new_board

        env = sokoban_env.Sokoban(copy_from=None, init_board=board_2d_to_3d(board))

        if self.gen_random:
            sym_idx = np.random.randint(env.get_symmetrical_envs_count())
            env = env.get_symmetrical_env(sym_idx)

        return env

    def create_env_with_key(self) -> tuple[str, env_base.BaseEnv]:
        key, board = self.levels_list[np.random.randint(len(self.levels_list))]
        env = self.prepare_env(board)
        return key, env

    def create_env(self) -> env_base.BaseEnv:
        return self.create_env_with_key()[1]