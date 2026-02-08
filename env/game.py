# -*- coding: utf-8 -*-

from __future__ import print_function

import time

import numpy as np

from env.board import Board
from env.pygameDisplay import PygameDisplay
from player.MCTSPlayer import MCTSPlayer

# 全局配置：是否打印详细日志（默认关闭以提升性能）
IS_VERBOSE = False  # 设为 True 启用详细的 print 输出


class Game(object):
    """game server"""
    def __init__(self, width, height, is_verbose=False,  **kwargs):

        self.is_verbose = is_verbose

        # 初始化棋盘
        self.width = width
        self.height = height
        self.board = Board(width=width,
                           height=height,
                           n_in_row=5)

        # 初始化 AI选手 MCTS Player
        self.mcts_player = MCTSPlayer(policy_value_function=None,
                                      c_puct=5,
                                      n_playout=1000,
                                      is_selfplay=1)

        self.pygameDisplay = PygameDisplay(width, height)

        # 训练时长追踪
        self.training_start_time = None  # 训练开始时间（时间戳）

        if self.is_verbose:
            print("Game:init: 初始化Game")

    def start_training_timer(self):
        """开始训练计时"""
        self.training_start_time = time.time()

    def get_training_time_str(self):
        """获取格式化的训练时长字符串 (例: '1 h 23 m 45 s')"""
        if self.training_start_time is None:
            return "N/A"

        elapsed_seconds = int(time.time() - self.training_start_time)
        hours = elapsed_seconds // 3600
        minutes = (elapsed_seconds % 3600) // 60
        seconds = elapsed_seconds % 60

        return f"{hours} h {minutes} m {seconds} s"

    def update_training_time_display(self):
        """更新显示的训练时长"""
        self.pygameDisplay.training_time = self.get_training_time_str()

    def start_play(self, player1, player2, is_shown=1):
        self.board.init_board()
        p1, p2 = self.board.players  # [1, 2]
        player1.set_player_ind(p1)   # 设置索引编号
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}

        while True:
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]

            if "human" in player_in_turn.__str__():
                move = self.pygameDisplay.get_human_action(self.board, self.board.width, self.board.height)

                # 检查是否点击了重开按钮
                if move == -2:
                    print("重新开始游戏...")
                    self.board.init_board()
                    self.pygameDisplay.update_screen(self.board, self.mcts_player.mcts)
                    continue
            else:
                # AI对战
                move = player_in_turn.get_action(self.board)
            if move != -1:
                self.board.do_move(move)  # l轮换

                # AI落子后添加延时（仅在有人类玩家时）
                if "human" not in player_in_turn.__str__():
                    time.sleep(0.1)

            if is_shown:
                # 人机对战 / alpha 和 pure对战的时候，推演完毕后会充值搜索树，此时pygame不会渲染MCTS
                self.update_training_time_display()  # 更新训练时长
                self.pygameDisplay.update_screen(self.board, self.mcts_player.mcts)

            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player", winner)
                    else:
                        print("Game end. Tie")

                # 如果有人类玩家，游戏结束后继续等待重开；否则直接返回
                if "human" not in player1.__str__() or "human" not in player2.__str__():
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
            self.mcts_player.mcts._visualize_callback = lambda: self.pygameDisplay.update_screen(self.board, self.mcts_player.mcts)
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
                self.update_training_time_display()  # 更新训练时长
                self.pygameDisplay.update_screen(self.board, self.mcts_player.mcts)  # 刷新界面，显示落子前的棋盘和新搜索树
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
