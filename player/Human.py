# -*- coding: utf-8 -*-
class Human(object):
    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def __str__(self):
        return "human {}".format(self.player)