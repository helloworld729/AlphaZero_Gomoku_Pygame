# -*- coding: utf-8 -*-

from __future__ import print_function

import copy
import random
import time
from collections import deque

import numpy as np
import pygame

from env.Board import Board
from env.MCTSPlayer import MCTSPlayer
# 初始化
pygame.init()
pygame.display.set_caption('五子棋')

# 全局配置：是否打印详细日志（默认关闭以提升性能）
IS_VERBOSE = False  # 设为 True 启用详细的 print 输出


class Game(object):
    """game server"""

    # def __init__(self, board, **kwargs):
    #     self.board = board
    def __init__(self, width, height, is_verbose=False,  **kwargs):

        self.is_verbose = is_verbose
        # 扩展窗口：左侧棋盘(600) + 中间信息(200) + 右侧搜索树(800) = 1600 x 600
        self.screen = pygame.display.set_mode((1600, 600))  # 整合棋盘和搜索树
        self.window_size = 600  # 棋盘尺寸
        self.grid_size = 8      # 网格数量
        self.cell_size = self.window_size // self.grid_size  # 每个cell的size
        self.policy_value_fn = None  # 用于计算胜率的策略函数

        # 初始化棋盘
        self.board = Board(width=width,
                           height=height,
                           n_in_row=5)

        # 初始化 AI选手 MCTS Player
        self.mcts_player = MCTSPlayer(policy_value_function=None,
                                      c_puct=5,
                                      n_playout=1000,
                                      is_selfplay=1)

        # 初始化 胜率值
        self.win_rates = {'player1': 0.5, 'player2': 0.5}

        # 初始化 重开按钮
        self.restart_button_rect = pygame.Rect(620, 520, 160, 50)

        # MCTS 搜索树可视化区域（右侧）
        self.tree_area_x = 800  # 搜索树区域起始x坐标
        self.tree_area_width = 800  # 搜索树区域宽度

        # 前端搜索树坐标显示格式：'1d' (一维) 或 '2d' (二维，默认)
        self.position_format = '2d'


        if self.is_verbose:
            print("Game:init: 初始化Game")

    # 走棋
    def machineStep(self, board):
        game_board = [[0] * self.board.width for _ in range(self.board.height)]

        # 先将所有棋子位置更新到 game_board
        for pos, playerNum in board.states.items():
            x, y = self.board.move_to_location(pos)  # h, w
            game_board[y][self.board.width-1-x] = playerNum

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
            last_move_pos = (w, self.board.width - h)  # 转换为 game_board 坐标

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
                    h, w = self.board.width - j, i  # 转换回棋盘坐标
                    coord_str = f"{h},{w}"
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
                    h, w = self.board.width-1 - j, i  # 转换回棋盘坐标
                    coord_str = f"{h},{w}"
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
            coord_text = font_coord.render(str(self.board.width-1 - i), True, (50, 50, 50))
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
                    newx = self.board.width - y
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
                    time.sleep(0.1)

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

    def start_self_play(self, is_shown=0, temp=1e-3, visualize_playout=False, playout_delay=0.5):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training

        参数:
            visualize_playout: 是否启用每次 playout 的可视化（会显著降低速度）
            playout_delay: 每次 playout 可视化后的延迟时间（秒）
        """
        # 如果启用 playout 可视化，设置回调
        if visualize_playout and is_shown:
            self.mcts_player.mcts._visualize_callback = lambda: self.machineStep(self.board)
            self.mcts_player.mcts._visualize_delay = playout_delay
            print(f"[可视化] 启用 playout 推演可视化，延迟={playout_delay}秒，预计每步耗时={self.mcts_player.mcts._n_playout * playout_delay}秒")
        else:
            self.mcts_player.mcts._visualize_callback = None

        # 完成一次完整的对弈，然后返回
        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []
        while True:
            # t1: MCTS 搜索（内部会执行推演）  log
            move, move_probs = self.mcts_player.get_action(self.board,
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

            if self.is_verbose:
                print(f"t3: 棋盘落子 move={move}")
            self.board.do_move(move)
            # 更新搜索树根节点（搜索树复用）
            self.mcts_player.mcts.set_root(move)

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
                self.mcts_player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)
