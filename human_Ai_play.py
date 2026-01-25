# -*- coding: utf-8 -*-

from __future__ import print_function

from env.game import Board, Game, MCTSPlayer
# from policy_value_net import PolicyValueNet  # Theano and Lasagne
from policy_value_net_pytorch import PolicyValueNet  # Pytorch


# from policy_value_net_tensorflow import PolicyValueNet # Tensorflow
# from policy_value_net_keras import PolicyValueNet  # Keras

class Human(object):
    """
    human player
    """

    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def __str__(self):
        return "Human {}".format(self.player)


def run():
    n = 5
    width, height = 8, 8
    model_file = 'best_policy_8_8_5_realGood.model'

    # PLAYOUT 推演可视化配置（可选，会显著降低 AI 思考速度）
    VISUALIZE_PLAYOUT = True  # 设为 True 启用每次 playout 的可视化
    PLAYOUT_DELAY = 0.0  # 每次 playout 可视化后的延迟时间（秒）

    try:
        board = Board(width=width, height=height, n_in_row=n)
        game = Game(board)

        # ############### human VS AI ###################
        # load the trained policy_value_net in either Theano/Lasagne, PyTorch or TensorFlow

        best_policy = PolicyValueNet(width, height, model_file = model_file)
        mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=400)

        # 将 mcts_player 传递给游戏对象，用于实时显示搜索树
        game.mcts_player = mcts_player

        # 将策略函数传递给游戏对象用于胜率评估
        game.policy_value_fn = best_policy.policy_value_fn

        # load the provided model (trained in Theano/Lasagne) into a MCTS player written in pure numpy
        # try:
        #     policy_param = pickle.load(open(model_file, 'rb'))
        # except:
        #     policy_param = pickle.load(open(model_file, 'rb'),
        #                                encoding='bytes')  # To support python3
        # best_policy = PolicyValueNetNumpy(width, height, policy_param)
        # mcts_player = MCTSPlayer(best_policy.policy_value_fn,
        #                          c_puct=5,
        #                          n_playout=400)  # set larger n_playout for better performance

        # uncomment the following line to play with pure MCTS (it's much weaker even with a larger n_playout)
        # mcts_player = MCTS_Pure(c_puct=5, n_playout=1000)

        # human player, input your move in the format: 2,3
        human = Human()

        # set start_player=0 for human first
        # human_player=1 表示 player1 (human) 是人类玩家
        game.start_play(human, mcts_player, start_player=1, is_shown=1, human_player=1,
                       visualize_playout=VISUALIZE_PLAYOUT, playout_delay=PLAYOUT_DELAY)
    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    run()
