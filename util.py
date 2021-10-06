#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union

realnum = Union[float, int]


class SimpleSumAverager:
    def __init__(self) -> None:
        self._average: realnum
        self._elements: int
        self.reset()

    def consider(self, to_consider: realnum) -> None:
        new_elements: int = self._elements + 1
        self._average: realnum = (
            (self._average * self._elements) + to_consider
        ) / new_elements
        self._elements: int = new_elements

    @property
    def average(self) -> realnum:
        return self._average

    @property
    def elements(self) -> int:
        return self._elements

    @property
    def sum(self) -> realnum:
        return self.average * self.elements

    def reset(self) -> None:
        self._average: realnum = 0.0
        self._elements: int = 0
