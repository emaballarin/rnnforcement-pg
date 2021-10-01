#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from environments.tmaze import TMaze

mytmaze = TMaze(100, -100, 5, "up", -1, -5, False)

print(mytmaze.state_bwa)
mytmaze.take_action(3)
mytmaze.take_action(3)
mytmaze.take_action(3)
mytmaze.take_action(3)
mytmaze.take_action(3)
mytmaze.take_action(3)
print(mytmaze.gameover)
mytmaze.take_action(0)
print(mytmaze.gameover)
