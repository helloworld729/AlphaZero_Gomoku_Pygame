# -*- coding: utf-8 -*-

from __future__ import print_function

import copy
import random
import time
from collections import deque

import numpy as np
import pygame

# 初始化
pygame.init()
pygame.display.set_caption('五子棋')

# 全局配置：是否打印详细日志（默认关闭以提升性能）
IS_VERBOSE = False  # 设为 True 启用详细的 print 输出


# ==================== MCTS 相关类 ====================

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class TreeNode(object):
    """A node in the MCTS tree."""

    def __init__(self, parent, prior_p, name=None):
        self._parent = parent
        self.name = name  # name表示当前节点的一维 坐标值：0-63
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0  # 访问次数
        self._Q = 0         # exploited  实际价值
        self._u = 0         # explored   探索价值
        self._P = prior_p   # 先验概率
        if name < 0 and IS_VERBOSE:
            print("TreeNode:init: 初始化节点{}".format(name))

    def expand(self, action_priors):
        """Expand tree by creating new children."""
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob, action)
            else:
                assert False
        if IS_VERBOSE:
            print("TreeNode:expand: 扩展了{}个节点".format(len(self._children)))

    def select(self, c_puct):
        """Select action among children that gives maximum action value Q plus bonus u(P)."""
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """Update node values from leaf evaluation."""
        self._n_visits += 1
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors."""
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        if IS_VERBOSE:
            print("TreeNode:update_recursive: 价值回溯，当前节点={}, 节点价值(上帝视角)={}".format(self.name, leaf_value))
        self.update(leaf_value)

    def get_value(self, c_puct):
        """Calculate and return the value for this node."""
        # UCT = Q + u，其中 u = c_puct * P * sqrt(parent_visits) / (1 + node_visits)
        self._u = (c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        """Check if leaf node."""
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """An implementation of Monte Carlo Tree Search."""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000, visualize_callback=None, visualize_delay=0.5):
        if IS_VERBOSE:
            print("MCTS:init: 初始化 博弈树 MCTS")
        self._root = TreeNode(None, 1.0, -1)
        self._policy = policy_value_fn  # 策略网络->计算UCT中的先验概率需要
        self._c_puct = c_puct        # 常数->计算UCT需要
        self._n_playout = n_playout  # 推演次数，要执行某个动作前推演的次数
        self._visualize_callback = visualize_callback  # 可视化回调函数
        self._visualize_delay = visualize_delay  # 每次可视化后的延迟时间

    def _playout(self, state):
        # 输入board状态，执行节点推演
        """Run a single playout from the root to the leaf."""
        if IS_VERBOSE:
            print("MCTS:_playout: 开始推演, 此时根结点={}, 是否为叶子节点={}".format(self._root.name, self._root.is_leaf()))
        node = self._root
        while(1):
            if node.is_leaf():
                if IS_VERBOSE:
                    print("MCTS:_playout: 已经是叶子节点")
                break
            # 选择
            if IS_VERBOSE:
                print("MCTS:_playout: 不是叶子节点")
            action, node = node.select(self._c_puct)
            if IS_VERBOSE:
                print("MCTS:_playout: 执行select函数， 选择的action={}".format(action))
            state.do_move(action)
        # 评估
        if IS_VERBOSE:
            print("MCTS:_playout: 已到达叶子结点{}, 当前选手={}, 执行策略推理(过滤非法节点)".format(node.name, state.get_current_player()))
        action_probs, leaf_value = self._policy(state)
        if IS_VERBOSE:
            print("MCTS:_playout: 在叶子节点执行，当前state的【价值评估】(当前选手视角)=", leaf_value)

        end, winner = state.game_end()
        if not end:
            # 扩展
            if IS_VERBOSE:
                print("MCTS:_playout: node={}, 对当前叶子节点【执行子节点扩展】".format(node.name))
            node.expand(action_probs)

            # 扩展后立即可视化（如果启用了回调）
            if self._visualize_callback:
                if IS_VERBOSE:
                    print("MCTS:_playout: 扩展完成，触发可视化刷新")
                self._visualize_callback()
                time.sleep(self._visualize_delay)
        else:
            if winner == -1:  # tie
                leaf_value = 0.0
            else:
                leaf_value = (1.0 if winner == state.get_current_player() else -1.0)
                if IS_VERBOSE:
                    print("MCTS:_playout: 游戏结束, 价值评估矫正为1")

        # 回溯
        if IS_VERBOSE:
            print("MCTS:_playout: 开始价值回溯")
        node.update_recursive(-leaf_value)

    def get_move_probs(self, state, temp=1e-3):
        # 输入board状态 -> 在根节点执行推演 -> 在根节点计算 动作和概率分布
        """Run all playouts sequentially and return the available actions and their corresponding probabilities."""
        if IS_VERBOSE:
            print("MCTS:get_move_probs: 总共需要执行{}次推演".format(self._n_playout))
        for n in range(self._n_playout):
            if IS_VERBOSE:
                print("#" * 30, " ⬇️虚拟推演{}⬇️ ".format(n + 1), "#" * 30)
                print("MCTS:get_move_probs: MCTS现在深拷贝棋盘(搜索树唯一)，并开始执行第{}次推演".format(n + 1))
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)
        if IS_VERBOSE:
            print("MCTS:get_move_probs: 推演完毕！")

        if IS_VERBOSE:
            print("MCTS:get_move_probs: 获取[(动作, 节点访问次数)]")
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)

        if IS_VERBOSE:
            print("MCTS:get_move_probs: 基于访问次数, 计算节点第执行概率")
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))
        if IS_VERBOSE:
            print("MCTS:get_move_probs: 返回动作与概率")
        return acts, act_probs

    def set_root(self, last_move):
        """Step forward in the tree, keeping everything we already know about the subtree."""
        if last_move in self._root._children:
            if IS_VERBOSE:
                print("MCTS:set_root: 搜索树复用, 根节点设置为={},其父节点设置为None".format(last_move))
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            if IS_VERBOSE:
                print("MCTS:set_root: 搜索树重置")
            self._root = TreeNode(None, 1.0, -1)


class MCTSPlayer(object):
    """AI player based on MCTS"""

    def __init__(self, policy_value_function, c_puct=5, n_playout=2000, is_selfplay=0,
                 visualize_callback=None, visualize_delay=0.5):
        if IS_VERBOSE:
            print("MCTSPlayer:init: 初始化 博弈树玩家 MCTSPlayer")
        self.mcts = MCTS(policy_value_function, c_puct, n_playout, visualize_callback, visualize_delay)
        self._is_selfplay = is_selfplay
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.set_root(-1)

    def get_action(self, board, temp=1e-3, return_prob=0):
        """基于游戏局面，结合MCTS搜索，最终输出一个具体的落子动作"""
        sensible_moves = board.availables
        if IS_VERBOSE:
            print("MCTSPlayer:get_action: 有效动作集合大小={}, 明细={}".format(len(sensible_moves), sensible_moves))

        move_probs = np.zeros(board.width*board.height)
        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(board, temp)
            if IS_VERBOSE:
                print("##############################  ⬆️虚拟推演end⬆️  ##############################")

            move_probs[list(acts)] = probs
            if IS_VERBOSE:
                print("MCTSPlayer:get_action: 动作集合", acts)
                print("MCTSPlayer:get_action: 概率集合(输出3个元素)", move_probs[:3])

            # 选择最优动作
            if self._is_selfplay:
                move = acts[np.argmax(probs)]
                if IS_VERBOSE:
                    print("MCTSPlayer:get_action: 最终狄拉克采样动作={}".format(move))
                # 注意：不在这里调用 set_root，而是在外部（game层）调用
                # 这样可以在显示搜索树后再切换根节点
            else:
                move = np.random.choice(acts, p=probs)
                # 人机对战时重置搜索树
                self.mcts.set_root(-1)

            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            if IS_VERBOSE:
                print("WARNING: the board is full")

    def __str__(self):
        return "MCTS {}".format(self.player)


# ==================== 棋盘 相关类 ====================

class Board(object):
    """board for the game"""

    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 8))
        self.height = int(kwargs.get('height', 8))
        # board states stored as a dict,
        # key: move as location on the board,
        # value: player as pieces type
        self.states = {}  # move -> player
        # need how many pieces in a row to win
        self.n_in_row = int(kwargs.get('n_in_row', 5))
        self.players = [1, 2]  # player1 and player2

    def init_board(self, start_player=0):
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('board width and height can not be '
                            'less than {}'.format(self.n_in_row))
        self.current_player = self.players[start_player]  # start player
        # keep available moves in a list
        self.availables = list(range(self.width * self.height))
        self.states = {}
        self.last_move = -1

    def move_to_location(self, move):
        """
        3*3 board's moves like:
        6 7 8
        3 4 5
        0 1 2
        and move 5's location is (1,2)
        """
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location):
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self):
        """return the board state from the perspective of the current player.
        state shape: 4*width*height
        """

        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            square_state[0][move_curr // self.width,
                            move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width,
                            move_oppo % self.height] = 1.0
            # indicate the last move location
            square_state[2][self.last_move // self.width,
                            self.last_move % self.height] = 1.0
        if len(self.states) % 2 == 0:
            square_state[3][:, :] = 1.0  # indicate the colour to play
        return square_state[:, ::-1, :]

    def do_move(self, move):
        self.states[move] = self.current_player
        self.availables.remove(move)
        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )
        self.last_move = move

    def has_a_winner(self):
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        moved = list(set(range(width * height)) - set(self.availables))
        if len(moved) < self.n_in_row *2-1:
            return False, -1

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            if (w in range(width - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            if (h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                return True, player

            if (w in range(width - n + 1) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player

            if (w in range(n - 1, width) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player

        return False, -1

    def game_end(self):
        """Check whether the game is ended or not"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player


class Game(object):
    """game server"""

    # def __init__(self, board, **kwargs):
    #     self.board = board
    def __init__(self, board, window_size=600, grid_size=8,  **kwargs):
        self.board = board

        if IS_VERBOSE:
            print("Game:init: 初始化Game")
        # 扩展窗口：左侧棋盘(600) + 中间信息(200) + 右侧搜索树(800) = 1600 x 600
        self.screen = pygame.display.set_mode((1600, 600))  # 整合棋盘和搜索树
        self.window_size = window_size  # 棋盘尺寸
        self.grid_size = grid_size      # 单格尺寸
        self.cell_size = self.window_size // self.grid_size  # 每个cell的size
        self.policy_value_fn = None  # 用于计算胜率的策略函数
        self.win_rates = {'player1': 0.5, 'player2': 0.5}  # 初始胜率
        # 重开按钮的位置和尺寸
        self.restart_button_rect = pygame.Rect(620, 520, 160, 50)

        # MCTS 搜索树可视化区域（右侧）
        self.tree_area_x = 800  # 搜索树区域起始x坐标
        self.tree_area_width = 800  # 搜索树区域宽度

        # 前端搜索树坐标显示格式：'1d' (一维) 或 '2d' (二维，默认)
        self.position_format = '2d'

        # MCTS Player (整合到 Game 中，不再使用回调)
        self.mcts_player = None

    def simpleGraphic(self, board, player1, player2):
        """Draw the board and show game info"""
        width = board.width
        height = board.height

        print("Player", player1, "with X".rjust(3))
        print("Player", player2, "with O".rjust(3))
        print()
        for x in range(width):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(height - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                loc = i * width + j
                p = board.states.get(loc, -1)
                if p == player1:
                    print('X'.center(8), end='')
                elif p == player2:
                    print('O'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n')

    # 走棋
    def machineStep(self, board):
        game_board = [[0] * self.board.width for _ in range(self.board.height)]

        # 先将所有棋子位置更新到 game_board
        for pos, playerNum in board.states.items():
            x, y = self.board.move_to_location(pos)  # h, w
            game_board[y][7-x] = playerNum

        # 检查游戏是否结束
        end, winner = board.game_end()
        if end:
            # 游戏结束，将获胜方的胜率设置为100%
            if winner == 1:
                self.win_rates['player1'] = 1.0  # 玩家1获胜，胜率100%
                self.win_rates['player2'] = 0.0
            elif winner == 2:
                self.win_rates['player1'] = 0.0
                self.win_rates['player2'] = 1.0  # 玩家2获胜，胜率100%
            else:
                # 平局，双方胜率50%
                self.win_rates['player1'] = 0.5
                self.win_rates['player2'] = 0.5
        else:
            # 游戏未结束，使用神经网络评估胜率
            if self.policy_value_fn is not None and len(board.states) > 0:
                try:
                    _, value = self.policy_value_fn(board)
                    # value 是当前玩家（即将落子）的评估值（-1到1）
                    # 由于已经切换到下一个玩家，所以要取反来表示刚落子玩家的胜率
                    current_player = board.get_current_player()
                    last_player = 3 - current_player  # 刚刚落子的玩家（1变2，2变1）

                    # value 是当前玩家视角，刚落子玩家的胜率需要取反
                    if last_player == 1:
                        # 刚落子的是玩家1（黑子）
                        self.win_rates['player1'] = (-value + 1) / 2  # 取反后转换到0-1
                        self.win_rates['player2'] = 1 - self.win_rates['player1']
                    else:
                        # 刚落子的是玩家2（白子）
                        self.win_rates['player2'] = (-value + 1) / 2
                        self.win_rates['player1'] = 1 - self.win_rates['player2']
                except:
                    pass  # 如果评估失败，保持原有胜率

        # 最后统一更新一次屏幕（包含棋子和胜率）
        self.update_screen(game_board)
        pygame.display.flip()

    # 更新屏幕
    def update_screen(self, game_board):
        # 绘制棋盘
        self.draw_board()

        # 获取最后一步落子位置
        last_move = self.board.last_move
        last_move_pos = None
        if last_move != -1:
            h, w = self.board.move_to_location(last_move)
            last_move_pos = (w, 7 - h)  # 转换为 game_board 坐标

        # 绘制棋子
        font_coord = pygame.font.Font(None, 36)  # 棋子上的坐标字体（18*2=36）
        for i in range(self.board.height):
            for j in range(self.board.width):
                center_x = i * self.cell_size + self.cell_size // 2
                center_y = j * self.cell_size + self.cell_size // 2

                # 绘制黑子
                if game_board[i][j] == 1:
                    pygame.draw.circle(self.screen, (0, 0, 0),
                                       (center_x, center_y), self.cell_size // 2 - 2)
                    # 如果是最后一步，添加红点标记
                    if last_move_pos and (i, j) == last_move_pos:
                        pygame.draw.circle(self.screen, (255, 0, 0),
                                         (center_x, center_y), 6)

                    # 在黑子上显示坐标（白色文字）
                    h, w = 7 - j, i  # 转换回棋盘坐标
                    move = self.board.location_to_move([h, w])
                    if self.position_format == '2d':
                        coord_str = f"{h},{w}"
                    else:
                        coord_str = f"{move}"
                    coord_text = font_coord.render(coord_str, True, (255, 255, 255))
                    text_rect = coord_text.get_rect(center=(center_x, center_y))
                    self.screen.blit(coord_text, text_rect)

                # 绘制白子
                elif game_board[i][j] == 2:
                    pygame.draw.circle(self.screen, (255, 255, 255),
                                       (center_x, center_y), self.cell_size // 2 - 2)
                    # 如果是最后一步，添加红点标记
                    if last_move_pos and (i, j) == last_move_pos:
                        pygame.draw.circle(self.screen, (255, 0, 0),
                                         (center_x, center_y), 6)

                    # 在白子上显示坐标（黑色文字）
                    h, w = 7 - j, i  # 转换回棋盘坐标
                    move = self.board.location_to_move([h, w])
                    if self.position_format == '2d':
                        coord_str = f"{h},{w}"
                    else:
                        coord_str = f"{move}"
                    coord_text = font_coord.render(coord_str, True, (0, 0, 0))
                    text_rect = coord_text.get_rect(center=(center_x, center_y))
                    self.screen.blit(coord_text, text_rect)

        # 绘制胜率信息
        self.draw_win_rates()

        # 绘制 MCTS 搜索树（右侧）
        self.draw_mcts_tree()

    # 绘制胜率信息
    def draw_win_rates(self):
        # 在棋盘右侧显示胜率信息
        info_x = self.window_size + 20  # 信息区域起始x坐标

        # 设置字体
        font_title = pygame.font.Font(None, 36)
        font_large = pygame.font.Font(None, 48)
        font_normal = pygame.font.Font(None, 32)
        font_small = pygame.font.Font(None, 24)

        # 标题
        title_text = font_title.render("Game Info", True, (50, 50, 50))
        self.screen.blit(title_text, (info_x, 20))

        # 显示最新落子坐标
        if self.board.last_move != -1:
            h, w = self.board.move_to_location(self.board.last_move)
            last_move_label = font_small.render("Last Move:", True, (100, 100, 100))
            self.screen.blit(last_move_label, (info_x, 55))

            coord_text = font_normal.render(f"({h}, {w})", True, (255, 0, 0))  # 红色坐标 (行, 列)
            self.screen.blit(coord_text, (info_x, 75))

        # 显示下一步落子方
        current_player = self.board.get_current_player()
        last_player = 3 - current_player
        next_turn_text = font_small.render(f"Next: {'Black' if current_player == 1 else 'White'}", True, (100, 100, 100))
        self.screen.blit(next_turn_text, (info_x, 115))

        # 分隔线
        pygame.draw.line(self.screen, (150, 150, 150), (info_x, 140), (info_x + 160, 140), 2)

        # 玩家1（黑子）信息
        y_offset = 160
        player1_label = font_normal.render("Black", True, (0, 0, 0))
        self.screen.blit(player1_label, (info_x, y_offset))

        # 绘制黑子示例
        pygame.draw.circle(self.screen, (0, 0, 0), (info_x - 20, y_offset + 15), 12)

        # 玩家1胜率
        win_rate_1 = self.win_rates['player1'] * 100
        rate_text_1 = font_large.render(f"{win_rate_1:.1f}%", True, (0, 0, 0))
        self.screen.blit(rate_text_1, (info_x, y_offset + 40))

        # 绘制胜率条
        bar_width = 160
        bar_height = 30
        bar_x = info_x
        bar_y = y_offset + 100

        # 背景条
        pygame.draw.rect(self.screen, (200, 200, 200), (bar_x, bar_y, bar_width, bar_height))
        # 填充条（黑色玩家）
        fill_width = int(bar_width * self.win_rates['player1'])
        pygame.draw.rect(self.screen, (0, 0, 0), (bar_x, bar_y, fill_width, bar_height))
        # 边框
        pygame.draw.rect(self.screen, (100, 100, 100), (bar_x, bar_y, bar_width, bar_height), 2)

        # 玩家2（白子）信息
        y_offset = 300
        player2_label = font_normal.render("White", True, (100, 100, 100))
        self.screen.blit(player2_label, (info_x, y_offset))

        # 绘制白子示例
        pygame.draw.circle(self.screen, (255, 255, 255), (info_x - 20, y_offset + 15), 12)
        pygame.draw.circle(self.screen, (0, 0, 0), (info_x - 20, y_offset + 15), 12, 2)

        # 玩家2胜率
        win_rate_2 = self.win_rates['player2'] * 100
        rate_text_2 = font_large.render(f"{win_rate_2:.1f}%", True, (100, 100, 100))
        self.screen.blit(rate_text_2, (info_x, y_offset + 40))

        # 绘制胜率条
        bar_y = y_offset + 100
        # 背景条
        pygame.draw.rect(self.screen, (200, 200, 200), (bar_x, bar_y, bar_width, bar_height))
        # 填充条（白色玩家）
        fill_width = int(bar_width * self.win_rates['player2'])
        pygame.draw.rect(self.screen, (255, 255, 255), (bar_x, bar_y, fill_width, bar_height))
        # 边框
        pygame.draw.rect(self.screen, (100, 100, 100), (bar_x, bar_y, bar_width, bar_height), 2)

        # 绘制"重开一局"按钮（右下角）
        pygame.draw.rect(self.screen, (34, 139, 34), self.restart_button_rect, border_radius=8)  # 绿色
        pygame.draw.rect(self.screen, (50, 50, 50), self.restart_button_rect, 2, border_radius=8)  # 深色边框

        restart_text = font_normal.render("Restart", True, (255, 255, 255))
        restart_text_rect = restart_text.get_rect(center=self.restart_button_rect.center)
        self.screen.blit(restart_text, restart_text_rect)

    # 绘制棋盘
    def draw_board(self):
        self.screen.fill((220, 179, 92))  # 木质背景
        # 右侧信息区背景
        pygame.draw.rect(self.screen, (240, 240, 240), (self.window_size, 0, 200, 600))

        # 绘制棋盘网格
        for i in range(self.grid_size):
            pygame.draw.line(self.screen, (0, 0, 0), (self.cell_size // 2, self.cell_size // 2 + i * self.cell_size),
                             (self.window_size - self.cell_size // 2, self.cell_size // 2 + i * self.cell_size))
            pygame.draw.line(self.screen, (0, 0, 0), (self.cell_size // 2 + i * self.cell_size, self.cell_size // 2),
                             (self.cell_size // 2 + i * self.cell_size, self.window_size - self.cell_size // 2))

        # 绘制坐标标识
        font_coord = pygame.font.Font(None, 24)

        # 横坐标（顶部，0-7）
        for i in range(self.grid_size):
            coord_text = font_coord.render(str(i), True, (50, 50, 50))
            coord_rect = coord_text.get_rect(center=(i * self.cell_size + self.cell_size // 2, 15))
            self.screen.blit(coord_text, coord_rect)

        # 纵坐标（左侧，0-7，从上到下）
        for i in range(self.grid_size):
            coord_text = font_coord.render(str(7 - i), True, (50, 50, 50))
            coord_rect = coord_text.get_rect(center=(15, i * self.cell_size + self.cell_size // 2))
            self.screen.blit(coord_text, coord_rect)

    def get_human_action(self):
        move=-1
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos

                # 检查是否点击了重开按钮
                if self.restart_button_rect.collidepoint(mouse_pos):
                    return -2  # 返回特殊值表示重开游戏

                # 正常落子（只在棋盘区域）
                if mouse_pos[0] < self.window_size:  # 确保点击在棋盘区域
                    x, y = mouse_pos[0] // self.cell_size, mouse_pos[1] // self.cell_size
                    newx = 7 - y
                    newy = x
                    move = self.board.location_to_move([newx, newy])
                    print("human: ", move)
        return move

    def draw_mcts_tree(self):
        """在右侧区域绘制 MCTS 搜索树"""
        if self.mcts_player is None or not hasattr(self.mcts_player, 'mcts'):
            return

        mcts = self.mcts_player.mcts
        if mcts._root is None:
            return

        # 绘制搜索树区域背景
        tree_bg_rect = pygame.Rect(self.tree_area_x, 0, self.tree_area_width, 600)
        pygame.draw.rect(self.screen, (40, 40, 50), tree_bg_rect)

        # 绘制标题
        font_title = pygame.font.Font(None, 28)
        font_small = pygame.font.Font(None, 16)
        font_normal = pygame.font.Font(None, 20)

        title = font_title.render("MCTS Search Tree", True, (255, 255, 100))
        self.screen.blit(title, (self.tree_area_x + 20, 20))

        # 收集搜索树节点（采样版本）
        # 获取 c_puct 参数用于计算 UCT 值
        c_puct = mcts._c_puct if hasattr(mcts, '_c_puct') else 5
        node_levels = self._bfs_traverse_tree(mcts._root)

        if not node_levels:
            no_data_text = font_normal.render("No tree data yet", True, (150, 150, 150))
            self.screen.blit(no_data_text, (self.tree_area_x + 300, 300))
            return

        # 绘制树结构
        self._draw_tree_in_area(node_levels, font_small, font_normal)

    def _bfs_traverse_tree(self, root):
        """BFS遍历搜索树（采样版本）"""
        if root is None:
            return []

        levels = []
        queue = deque([(root, 0)])
        leaf_sample_prob = 0.05  # 5%采样（比独立窗口多一点，便于观察）

        while queue:# and len(levels) < 5:  # 最多显示5层
            node, level = queue.popleft()

            # 判断是否为叶子节点
            is_leaf = node.is_leaf()

            # 叶子节点采样
            if is_leaf and level > 0:
                if random.random() > leaf_sample_prob:
                    continue

            # 扩展层级列表（如果当前层级不存在则创建，同一层级的多个节点只会创建一次）
            while len(levels) <= level:
                levels.append([])

            # 添加节点信息
            node_info = {
                'node': node,
                'action': node.name,
                'visits': node._n_visits,
                'Q': node._Q,
                'P': node._P,  # 先验概率
                'is_leaf': is_leaf
            }
            levels[level].append(node_info)

            # 添加子节点（限制数量）
            if level < 40:  # 最多4层深度
                children = list(node._children.items())
                # 按访问次数排序，只显示前20个
                children.sort(key=lambda x: x[1]._n_visits, reverse=True)
                for action, child in children[:20]:
                    queue.append((child, level + 1))

        return levels

    def _draw_tree_in_area(self, node_levels, font_small, font_normal):
        """在指定区域绘制树"""
        # 颜色定义
        COLOR_NODE_DEFAULT = (120, 120, 120)
        COLOR_NODE_SELECTED = (255, 50, 50)
        COLOR_NODE_ROOT = (100, 200, 100)
        COLOR_NODE_MAX_UCT = (255, 165, 0)  # 橙色 - UCT 最大的节点
        COLOR_EDGE = (80, 80, 100)
        COLOR_TEXT = (220, 220, 220)

        node_radius = 8
        level_height = 100
        node_positions = {}

        # 找出每层访问次数最多的节点（实际会被选择的动作）
        max_visit_nodes = set()
        for level_idx, level_nodes in enumerate(node_levels):
            if level_idx > 0 and len(level_nodes) > 0:  # 跳过根节点层
                # 找出该层访问次数最多的节点（这才是实际会被选择的）
                max_visit_node = max(level_nodes, key=lambda n: n.get('visits', 0))
                max_visit_nodes.add(id(max_visit_node['node']))

        # 第一遍：记录位置并绘制边
        for level_idx, level_nodes in enumerate(node_levels):
            y = 80 + level_idx * level_height
            num_nodes = len(level_nodes)

            if num_nodes == 1:
                start_x = self.tree_area_x + self.tree_area_width // 2
                spacing = 0
            else:
                max_width = self.tree_area_width - 100
                spacing = min(max_width / num_nodes, 60)
                start_x = self.tree_area_x + 50 + (self.tree_area_width - 100 - spacing * (num_nodes - 1)) // 2

            for node_idx, node_info in enumerate(level_nodes):
                x = int(start_x + node_idx * spacing)
                node = node_info['node']
                node_positions[id(node)] = (x, y)

                # 绘制边
                if node._parent is not None:
                    parent_id = id(node._parent)
                    if parent_id in node_positions:
                        px, py = node_positions[parent_id]
                        pygame.draw.line(self.screen, COLOR_EDGE, (px, py), (x, y), 1)

        # 第二遍：绘制节点
        for level_idx, level_nodes in enumerate(node_levels):
            for node_info in level_nodes:
                node = node_info['node']
                if id(node) not in node_positions:
                    continue

                x, y = node_positions[id(node)]
                visits = node_info['visits']

                # 决定颜色
                if level_idx == 0:
                    # 根节点
                    color = COLOR_NODE_ROOT
                elif id(node) in max_visit_nodes:
                    # 访问次数最多的节点（实际会被选择）- 橙色
                    color = COLOR_NODE_MAX_UCT
                elif visits > 50:
                    color = COLOR_NODE_SELECTED
                elif visits > 0:
                    ratio = min(visits / 50, 1.0)
                    r = int(120 + (255 - 120) * ratio)
                    g = int(120 - 70 * ratio)
                    b = int(120 - 70 * ratio)
                    color = (r, g, b)
                else:
                    color = COLOR_NODE_DEFAULT

                # 绘制节点
                pygame.draw.circle(self.screen, color, (x, y), node_radius)
                pygame.draw.circle(self.screen, COLOR_TEXT, (x, y), node_radius, 1)

                # 绘制信息（所有非根节点都显示坐标）
                action = node_info['action']
                if action >= 0:  # 非根节点（根节点的 action 是 -1）
                    # 根据 position_format 格式化动作显示
                    if self.position_format == '2d':
                        # 二维坐标格式，与 last_move 显示一致 (行, 列)
                        h, w = self.board.move_to_location(action)
                        action_str = f"({h}, {w})"
                    else:
                        # 一维坐标格式
                        action_str = f"{action}"

                    action_text = font_small.render(action_str, True, COLOR_TEXT)
                    text_rect = action_text.get_rect(center=(x, y - node_radius - 22))
                    self.screen.blit(action_text, text_rect)

                    # 访问次数
                    visit_text = font_small.render(f"N:{visits}", True, COLOR_TEXT)
                    text_rect = visit_text.get_rect(center=(x, y + node_radius + 10))
                    self.screen.blit(visit_text, text_rect)

        # 绘制统计信息
        total_nodes = sum(len(level) for level in node_levels)
        stats_text = font_normal.render(f"Showing {total_nodes} nodes", True, (180, 180, 180))
        self.screen.blit(stats_text, (self.tree_area_x + 20, 560))

    def start_play(self, player1, player2, start_player=0, is_shown=1, human_player=None,
                   visualize_playout=False, playout_delay=0.5):
        """
        start a game between two players
        human_player: None (纯AI对战), 1 (player1是人类), 或 2 (player2是人类)
        visualize_playout: 是否启用每次 playout 的可视化（会显著降低速度）
        playout_delay: 每次 playout 可视化后的延迟时间（秒）
        """
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')
        self.board.init_board(start_player)
        p1, p2 = self.board.players
        player1.set_player_ind(p1)  # 1
        player2.set_player_ind(p2)  # 2
        players = {p1: player1, p2: player2}

        # 如果启用 playout 可视化，为 AI 玩家设置回调
        if visualize_playout and is_shown:
            if hasattr(player1, 'mcts'):
                player1.mcts._visualize_callback = lambda: self.machineStep(self.board)
                player1.mcts._visualize_delay = playout_delay
                print(f"[可视化] Player1 启用 playout 推演可视化")
            if hasattr(player2, 'mcts'):
                player2.mcts._visualize_callback = lambda: self.machineStep(self.board)
                player2.mcts._visualize_delay = playout_delay
                print(f"[可视化] Player2 启用 playout 推演可视化")
        else:
            # 禁用可视化
            if hasattr(player1, 'mcts'):
                player1.mcts._visualize_callback = None
            if hasattr(player2, 'mcts'):
                player2.mcts._visualize_callback = None

        while True:
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]

            # 判断是否需要人类交互
            if human_player is not None and current_player == human_player:
                move = self.get_human_action()

                # 检查是否点击了重开按钮
                if move == -2:
                    print("重新开始游戏...")
                    self.board.init_board(start_player)
                    self.win_rates = {'player1': 0.5, 'player2': 0.5}  # 重置胜率
                    self.machineStep(self.board)  # 刷新界面
                    continue
            else:
                # AI对战
                move = player_in_turn.get_action(self.board)
                # if is_shown:
                #     print(f"玩家{current_player}落子: ", move)

            if move != -1:
                self.board.do_move(move)  # l轮换

                # AI落子后添加延时（仅在有人类玩家时）
                if human_player is not None and current_player != human_player:
                    time.sleep(1.0)

            if is_shown:
                self.machineStep(self.board)

            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player", winner)
                    else:
                        print("Game end. Tie")

                # 如果有人类玩家，游戏结束后继续等待重开；否则直接返回
                if human_player is not None:
                    # 游戏结束后继续等待，用户可以点击重开按钮
                    continue
                else:
                    # 纯AI对战，直接返回胜者
                    return winner

    def start_self_play(self, player, is_shown=0, temp=1e-3, visualize_playout=False, playout_delay=0.5):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training

        参数:
            visualize_playout: 是否启用每次 playout 的可视化（会显著降低速度）
            playout_delay: 每次 playout 可视化后的延迟时间（秒）
        """
        # 如果启用 playout 可视化，设置回调
        if visualize_playout and is_shown:
            player.mcts._visualize_callback = lambda: self.machineStep(self.board)
            player.mcts._visualize_delay = playout_delay
            print(f"[可视化] 启用 playout 推演可视化，延迟={playout_delay}秒，预计每步耗时={player.mcts._n_playout * playout_delay}秒")
        else:
            player.mcts._visualize_callback = None

        # 完成一次完整的对弈，然后返回
        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []
        while True:
            # t1: MCTS 搜索（内部会执行推演）
            move, move_probs = player.get_action(self.board,
                                                 temp=temp,
                                                 return_prob=1)

            if is_shown and IS_VERBOSE:
                print(f"t2: MCTS 搜索完成，最优动作={move}")

            # store the data
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)

            # 最终推演结果的结果
            # if is_shown:
            #     self.machineStep(self.board)  # 刷新界面，显示落子前的棋盘和新搜索树
                # time.sleep(5)

            if IS_VERBOSE:
                print(f"t3: 棋盘落子 move={move}")
            self.board.do_move(move)
            # 更新搜索树根节点（搜索树复用）
            player.mcts.set_root(move)

            # 新的推演起点
            if is_shown:
                self.machineStep(self.board)  # 刷新界面，显示落子前的棋盘和新搜索树
                # time.sleep(5)

            end, winner = self.board.game_end()
            if end:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if winner != -1:  # 不是平局
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # reset MCTS root node
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)
