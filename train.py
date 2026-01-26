# -*- coding: utf-8 -*-

from __future__ import print_function

import random
from collections import defaultdict, deque

import numpy as np

from env.game import Game
from player.MCTSPlayer import MCTSPlayer
from player.mcts_pure import MCTSPlayer as MCTS_Pure
# from policy_value_net import PolicyValueNet  # Theano and Lasagne
from policy_value_net_pytorch import PolicyValueNet  # Pytorch

IS_VERBOSE = False

class TrainPipeline():
    def __init__(self, init_model=None):
        print("TrainPipeline:init: 初始化: TrainPipeline")
        # params of the board and the game
        # training params
        self.learn_rate = 5e-5  # 原始值=2e-3->5e-5
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.n_playout = 1000  # num of simulations for each move， 原始值=400 访问的广度
        self.c_puct = 5  # 原始值=5,表示探索的系数
        self.buffer_size = 10000
        self.batch_size = 512  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.02
        self.check_freq = 50  # 多少次进行一次 和纯MCTS的 对局评估， 原始值=50
        self.game_batch_num = 2000  # 训练多少次自我对弈  1500
        self.best_win_ratio = 0.0
        self.is_shown_pygame = 1  # 是否展示/刷新 pygame界面
        self.pure_mcts_playout_num = 1000

        # Game相关
        self.board_width = 10
        self.board_height = 10
        self.game = Game(width=self.board_width, height=self.board_height, is_verbose=IS_VERBOSE)

        if init_model:
            # 基于checkPoint继续训练
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height,
                                                   model_file=init_model)
        else:
            # start training from a new policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height)
        self.game.mcts_player.mcts.set_policy(self.policy_value_net.policy_value_fn)

    # 数据扩充
    def get_equi_data(self, play_data):
        """augment the data set by rotation and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        """
        extend_data = []
        for state, mcts_prob, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_prob.reshape(self.board_height, self.board_width)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data

    # 收集自我 博弈 数据
    def collect_selfplay_data(self, n_games=1, visualize_playout=False, playout_delay=0.5):
        """collect self-play data for training

        参数:
            n_games: 自我对弈的局数
            visualize_playout: 是否启用 playout 推演可视化（会显著降低速度）
            playout_delay: 每次 playout 可视化后的延迟时间（秒）
        """
        for i in range(n_games):
            print("Game:collect_selfplay_data: 开始自我博弈")
            winner, play_data = self.game.start_self_play(is_shown=self.is_shown_pygame,
                                                          temp=self.temp,
                                                          visualize_playout=visualize_playout,
                                                          playout_delay=playout_delay)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            # augment the data
            play_data = self.get_equi_data(play_data)
            print("TrainPipeline: collect_selfplay_data：数据入栈")
            self.data_buffer.extend(play_data)

    def policy_update(self):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                    state_batch,
                    mcts_probs_batch,
                    winner_batch,
                    self.learn_rate*self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1)
            )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))

        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(kl, self.lr_multiplier, loss, entropy,
                        explained_var_old, explained_var_new))
        return loss, entropy

    def policy_evaluate(self, n_games=10):
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=self.n_playout)
        pure_mcts_player = MCTS_Pure(c_puct=5,
                                     n_playout=self.pure_mcts_playout_num)
        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play(current_mcts_player,
                                          pure_mcts_player,
                                          start_player=i % 2,
                                          is_shown=1)
            win_cnt[winner] += 1
        win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
                self.pure_mcts_playout_num,
                win_cnt[1], win_cnt[2], win_cnt[-1]))
        return win_ratio

    def run(self):
        """run the training pipeline"""
        try:
            # 启动训练计时器
            self.game.start_training_timer()

            for i in range(self.game_batch_num):
                print("TrainPipeline:run: 收集数据")
                # 从 __main__ 中获取配置（如果有的话）
                visualize_playout = getattr(self, 'visualize_playout', False)
                playout_delay = getattr(self, 'playout_delay', 0.5)
                self.collect_selfplay_data(self.play_batch_size, visualize_playout, playout_delay)
                print("batch i:{}, episode_len:{}".format(i+1, self.episode_len))
                # 更新模型
                if len(self.data_buffer) > self.batch_size:
                    print("TrainPipeline:run: 开始模型训练")
                    loss, entropy = self.policy_update()

                # check the performance of the current model,
                # and save the model params
                if (i+1) % self.check_freq == 0:
                    print("已经训练: {}轮".format(i+1))
                    win_ratio = self.policy_evaluate()
                    self.policy_value_net.save_model('./current_policy_{}_{}_5.model'.format(self.board_width, self.board_width))
                    if win_ratio > self.best_win_ratio:
                        print("相较于MCTS@{}, 截至目前的最佳胜率={} !!!!!!!!".format(self.pure_mcts_playout_num, win_ratio))
                        self.best_win_ratio = win_ratio
                        # update the best_policy
                        self.policy_value_net.save_model('./best_policy_{}_{}_5_realGood.model'.format(self.board_width, self.board_width))
                        if (self.best_win_ratio == 1.0 and
                                self.pure_mcts_playout_num < 5000):
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0
        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == '__main__':
    # training_pipeline = TrainPipeline(init_model='best_policy_8_8_5_realGood.model')
    training_pipeline = TrainPipeline()

    # 推演可视化配置（可选，会显著降低训练速度）
    # 注:搜索树的展示有2个粒度，① 推演前后的展示，② 推演过程的展示，这里配置的是后者
    training_pipeline.visualize_playout = False
    training_pipeline.playout_delay = 2

    training_pipeline.run()

    # 输出训练时间
    print("训练时间: {}s".format(training_pipeline.game.get_training_time_str()))

