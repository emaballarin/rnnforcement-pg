#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tmaze import TMaze

pippo = TMaze(0, 0, 10)

print(pippo._gw.position)
pippo.take_action(3)
pippo.take_action(3)
pippo.take_action(3)
pippo.take_action(3)
pippo.take_action(3)
pippo.take_action(3)
pippo.take_action(3)
pippo.take_action(3)
pippo.take_action(3)
pippo.take_action(1)
print(pippo._gw.position)
print(pippo.gameover)
