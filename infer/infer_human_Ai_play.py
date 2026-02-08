# -*- coding: utf-8 -*-

from __future__ import print_function
from env.game import Game
from model.policy_value_net_pytorch import PolicyValueNet  # Pytorch
from player.Human import Human
from player.MCTSPlayer import MCTSPlayer


def run():
    width, height, n = 8, 8, 5
    model_file = ('/Users/aihesuannaidemaomi/PycharmProjects/AlphaZero_Gomoku_Pygame/saved_models'
                  '/best_policy_8_8_5_realGood.model')
    human_first = True
    try:
        game = Game(width=width, height=height)
        # ############### human VS AI ###################
        best_policy = PolicyValueNet(width, height, model_file=model_file)
        mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=1000)

        # 将 mcts_player 传递给游戏对象，用于实时显示搜索树
        game.mcts_player = mcts_player

        # 启动训练计时器（用于显示模型训练时长）
        game.start_training_timer()
        human_player = Human()
        if human_first:
            game.start_play(human_player, mcts_player, is_shown=1)
        else:
            game.start_play(mcts_player, human_player, is_shown=1)
    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    run()
