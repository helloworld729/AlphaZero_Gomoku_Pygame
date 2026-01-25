# -*- coding: utf-8 -*-

from __future__ import print_function

import copy
import random
import time
from collections import deque

import numpy as np

#############################
#        TreeNode           #
#        Mcts               #
#        MctsPlayer         #
#############################


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class TreeNode(object):
    """A node in the MCTS tree."""

    def __init__(self, parent, prior_p, name=None, is_verbose=False):
        self._parent = parent
        self.name = name  # name表示当前节点的一维 坐标值：0-63
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0  # 访问次数
        self._Q = 0         # exploited  实际价值
        self._u = 0         # explored   探索价值
        self._P = prior_p   # 先验概率
        self.is_verbose = is_verbose
        if self.is_verbose:
            print("TreeNode:init: 初始化节点{}".format(name))

    def expand(self, action_priors):
        """Expand tree by creating new children."""
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob, action)
            else:
                assert False
        if self.is_verbose:
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
        if self.is_verbose:
            print("TreeNode:update_recursive: 价值回溯，当前节点={}, 节点价值(上帝视角)={}".format(self.name, leaf_value))
        self.update(leaf_value)

    def get_value(self, c_puct):
        """Calculate and return the value for this node."""
        # UCT = Q + u
        self._u = (c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        """Check if leaf node."""
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """An implementation of Monte Carlo Tree Search."""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000, visualize_callback=None, visualize_delay=0.5, is_verbose=False):
        self._root = TreeNode(None, 1.0, -1)
        self._policy = policy_value_fn  # 策略网络->计算UCT中的先验概率需要
        self._c_puct = c_puct           # 常数->计算UCT需要
        self._n_playout = n_playout     # 推演次数，要执行某个动作前推演的次数
        self._visualize_callback = visualize_callback  # 可视化回调函数
        self._visualize_delay = visualize_delay        # 每次可视化后的延迟时间
        self.is_verbose = is_verbose
        if self.is_verbose:
            print("MCTS:init: 初始化 博弈树 MCTS")

    def set_policy(self, policy_value_fn):
        self._policy = policy_value_fn

    def _playout(self, state):
        # 输入board状态，执行节点推演
        """Run a single playout from the root to the leaf."""
        if self.is_verbose:
            print("MCTS:_playout: 开始推演, 此时根结点={}, 是否为叶子节点={}".format(self._root.name, self._root.is_leaf()))
        node = self._root
        while(1):
            if node.is_leaf():
                if self.is_verbose:
                    print("MCTS:_playout: 已经是叶子节点")
                break
            # 选择
            if self.is_verbose:
                print("MCTS:_playout: 不是叶子节点")
            action, node = node.select(self._c_puct)
            if self.is_verbose:
                print("MCTS:_playout: 执行select函数， 选择的action={}".format(action))
            state.do_move(action)

        end, winner = state.game_end()
        if not end:
            # 评估 扩展
            next_probs, next_value = self._policy(state)
            node.expand(next_probs)

            # 回溯
            node.update_recursive(-next_value)

            # 扩展后立即可视化（如果启用了回调）
            if self._visualize_callback:
                self._visualize_callback()
                time.sleep(self._visualize_delay)
        else:
            if winner == -1:  # tie
                node_value = 0.0
            else:
                node_value = 1
            node.update_recursive(node_value)

    def get_move_probs(self, state, temp=1e-3):
        # 输入board状态 -> 在根节点执行推演 -> 在根节点计算 动作和概率分布
        """Run all playouts sequentially and return the available actions and their corresponding probabilities."""
        if self.is_verbose:
            print("MCTS:get_move_probs: 总共需要执行{}次推演".format(self._n_playout))
        for n in range(self._n_playout):
            if self.is_verbose:
                print("#" * 30, " ⬇️虚拟推演{}⬇️ ".format(n + 1), "#" * 30)
                print("MCTS:get_move_probs: MCTS现在深拷贝棋盘(搜索树唯一)，并开始执行第{}次推演".format(n + 1))
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)
        if self.is_verbose:
            print("MCTS:get_move_probs: 推演完毕！")

        if self.is_verbose:
            print("MCTS:get_move_probs: 获取[(动作, 节点访问次数)]")
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)

        if self.is_verbose:
            print("MCTS:get_move_probs: 基于访问次数, 计算节点第执行概率")
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))
        if self.is_verbose:
            print("MCTS:get_move_probs: 返回动作与概率")
        return acts, act_probs

    def set_root(self, last_move):
        """Step forward in the tree, keeping everything we already know about the subtree."""
        if last_move in self._root._children:
            if self.is_verbose:
                print("MCTS:set_root: 搜索树复用, 根节点设置为={},其父节点设置为None".format(last_move))
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            if self.is_verbose:
                print("MCTS:set_root: 搜索树重置")
            self._root = TreeNode(None, 1.0, -1)


class MCTSPlayer(object):
    """AI player based on MCTS"""
    def __init__(self, policy_value_function, c_puct=5, n_playout=2000, is_selfplay=0,
                 visualize_callback=None, visualize_delay=0.5, is_verbose=False):
        # visualize_callback 回调函数
        self.mcts = MCTS(policy_value_function, c_puct, n_playout, visualize_callback, visualize_delay)
        self._is_selfplay = is_selfplay
        self.player = None  # 玩家索引号
        self.is_verbose = is_verbose
        if self.is_verbose:
            print("MCTSPlayer:init: 初始化 博弈树玩家 MCTSPlayer")

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.set_root(-1)

    def get_action(self, board, temp=1e-3, return_prob=0):
        """基于游戏局面，结合MCTS搜索，最终输出一个具体的落子动作"""
        sensible_moves = board.availables
        if self.is_verbose:
            print("MCTSPlayer:get_action: 有效动作集合大小={}, 明细={}".format(len(sensible_moves), sensible_moves))

        move_probs = np.zeros(board.width*board.height)
        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(board, temp)
            if self.is_verbose:
                print("##############################  ⬆️虚拟推演end⬆️  ##############################")

            move_probs[list(acts)] = probs
            if self.is_verbose:
                print("MCTSPlayer:get_action: 动作集合", acts)
                print("MCTSPlayer:get_action: 概率集合(输出3个元素)", move_probs[:3])

            # 选择最优动作
            if self._is_selfplay:
                move = acts[np.argmax(probs)]
                if self.is_verbose:
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
            if self.is_verbose:
                print("WARNING: the board is full")

    def __str__(self):
        return "MCTS {}".format(self.player)
