import random
from collections import deque

import pygame
from player.MCTSPlayer import MCTS
from env.board import Board
# 初始化
pygame.init()
pygame.display.set_caption('五子棋')


class PygameDisplay(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height
        # 扩展窗口：左侧棋盘(600) + 中间信息(200) + 右侧搜索树(800) = 1600 x 600
        self.screen = pygame.display.set_mode((1600, 600))  # 整合棋盘和搜索树
        self.window_size = 600  # 棋盘尺寸
        self.grid_size = self.width  # 网格数量
        self.cell_size = self.window_size // self.grid_size

        # 初始化 重开按钮
        self.restart_button_rect = pygame.Rect(620, 520, 160, 50)

        # MCTS 搜索树可视化区域（右侧）
        self.tree_area_x = 800  # 搜索树区域起始x坐标
        self.tree_area_width = 800  # 搜索树区域宽度

        # 前端搜索树坐标显示格式：'1d' (一维) 或 '2d' (二维，默认)
        self.position_format = '2d'

    # 更新屏幕
    def update_screen(self, board, mcts):
        board_nums = [[0] * board.width for _ in range(board.height)]
        for pos, playerNum in board.states.items():
            x, y = board.move_to_location(pos)  # h, w
            board_nums[y][board.width - 1 - x] = playerNum

        # 绘制棋盘
        self.draw_board()

        # 获取最后一步落子位置
        last_move_pos = None
        last_move = board.last_move
        if last_move != -1:
            h, w = board.move_to_location(last_move)
            last_move_pos = (w, self.width - 1 - h)  # 转换为 pygame 坐标

        # 绘制棋子
        font_coord = pygame.font.Font(None, 36)  # 棋子上的坐标字体（18*2=36）
        for i in range(self.height):
            for j in range(self.width):
                center_x = i * self.cell_size + self.cell_size // 2
                center_y = j * self.cell_size + self.cell_size // 2

                # 绘制黑子
                if board_nums[i][j] == 1:
                    pygame.draw.circle(self.screen, (0, 0, 0),
                                       (center_x, center_y), self.cell_size // 2 - 2)
                    # 如果是最后一步，添加红点标记
                    if last_move_pos and (i, j) == last_move_pos:
                        pygame.draw.circle(self.screen, (255, 0, 0),
                                           (center_x, center_y), 6)

                    # 在黑子上显示坐标（白色文字）
                    h, w = board.width - 1 - j, i  # 转换回棋盘坐标
                    coord_str = f"{h},{w}"
                    coord_text = font_coord.render(coord_str, True, (255, 255, 255))
                    text_rect = coord_text.get_rect(center=(center_x, center_y))
                    self.screen.blit(coord_text, text_rect)

                # 绘制白子
                elif board_nums[i][j] == 2:
                    pygame.draw.circle(self.screen, (255, 255, 255),
                                       (center_x, center_y), self.cell_size // 2 - 2)
                    # 如果是最后一步，添加红点标记
                    if last_move_pos and (i, j) == last_move_pos:
                        pygame.draw.circle(self.screen, (255, 0, 0),
                                           (center_x, center_y), 6)

                    # 在白子上显示坐标（黑色文字）
                    h, w = board.width - 1 - j, i  # 转换回棋盘坐标
                    coord_str = f"{h},{w}"
                    coord_text = font_coord.render(coord_str, True, (0, 0, 0))
                    text_rect = coord_text.get_rect(center=(center_x, center_y))
                    self.screen.blit(coord_text, text_rect)

        # 绘制 MCTS 搜索树（右侧）
        self.draw_mcts_tree(board, mcts)

        pygame.display.flip()

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
            coord_text = font_coord.render(str(self.width -1 - i), True, (50, 50, 50))
            coord_rect = coord_text.get_rect(center=(15, i * self.cell_size + self.cell_size // 2))
            self.screen.blit(coord_text, coord_rect)

    def draw_mcts_tree(self, board:Board, mcts:MCTS):
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
        node_levels = self._bfs_traverse_tree(mcts._root)

        if not node_levels:
            no_data_text = font_normal.render("No tree data yet", True, (150, 150, 150))
            self.screen.blit(no_data_text, (self.tree_area_x + 300, 300))
            return

        # 绘制游戏信息
        self.draw_game_info(board)

        # 绘制树结构
        self._draw_tree_in_area(board, node_levels, font_small, font_normal)

    # 绘制胜率信息
    def draw_game_info(self, board:Board):
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
        if board.last_move != -1:
            h, w = board.move_to_location(board.last_move)
            last_move_label = font_small.render("Last Move:", True, (100, 100, 100))
            self.screen.blit(last_move_label, (info_x, 55))

            coord_text = font_normal.render(f"({h}, {w})", True, (255, 0, 0))  # 红色坐标 (行, 列)
            self.screen.blit(coord_text, (info_x, 75))

        # 显示下一步落子方
        current_player = board.get_current_player()
        next_turn_text = font_small.render(f"Next: {'Black' if current_player == 1 else 'White'}", True, (100, 100, 100))
        self.screen.blit(next_turn_text, (info_x, 115))

        # 分隔线
        pygame.draw.line(self.screen, (150, 150, 150), (info_x, 140), (info_x + 160, 140), 2)

        # 绘制"重开一局"按钮（右下角）
        pygame.draw.rect(self.screen, (34, 139, 34), self.restart_button_rect, border_radius=8)  # 绿色
        pygame.draw.rect(self.screen, (50, 50, 50), self.restart_button_rect, 2, border_radius=8)  # 深色边框
        restart_text = font_normal.render("Restart", True, (255, 255, 255))
        restart_text_rect = restart_text.get_rect(center=self.restart_button_rect.center)
        self.screen.blit(restart_text, restart_text_rect)

    def _bfs_traverse_tree(self, root):
        """BFS遍历搜索树（采样版本）"""
        if root is None:
            return []

        levels = []
        queue = deque([(root, 0)])
        leaf_sample_prob = 0.05  # 5%采样（比独立窗口多一点，便于观察）

        while queue:  # and len(levels) < 5:  # 最多显示5层
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

    def _draw_tree_in_area(self, board, node_levels, font_small, font_normal):
        """在指定区域绘制树"""
        # 颜色定义
        COLOR_NODE_DEFAULT = (120, 120, 120)
        COLOR_NODE_SELECTED = (255, 50, 50)
        COLOR_NODE_ROOT = (100, 200, 100)
        COLOR_NODE_MAX_UCT = (255, 165, 0)  # 橙色 - UCT 最大的节点
        COLOR_EDGE = (80, 80, 100)
        COLOR_TEXT = (220, 220, 220)

        node_radius = 8  # 搜索树节点半径
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
                        h, w = board.move_to_location(action)
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

    def get_human_action(self, board:Board, width, height):
        move = -1
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos

                # 检查是否点击了重开按钮
                if self.restart_button_rect.collidepoint(mouse_pos):
                    return -2  # 返回特殊值表示重开游戏

                # 正常落子（只在棋盘区域）
                if mouse_pos[0] < self.window_size:  # 确保点击在棋盘区域
                    x, y = mouse_pos[0] // self.cell_size, mouse_pos[1] // self.cell_size
                    move = board.location_to_move([width - y, x])
                    print("human: ", move)
        return move
