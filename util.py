#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---- IMPORTS ----
from typing import Union


# ---- CUSTOM TYPES ----
realnum = Union[float, int]


# ---- CLASSES ----
class SimpleSumAverager:
    """
    A simple class to keep track of sums and averages of real-valued quantities.
    """

    def __init__(self) -> None:
        """
        Instantiate a new SimpleSumAverager object.
        """
        self._average: realnum
        self._elements: int
        self.reset()

    def consider(self, to_consider: realnum) -> None:
        """
        Consider a new value in the sum/average being kept track of.

        Args:
            to_consider (Union[float, int]): The new value to consider.
        """
        new_elements: int = self._elements + 1
        self._average: realnum = (
            (self._average * self._elements) + to_consider
        ) / new_elements
        self._elements: int = new_elements

    @property
    def average(self) -> realnum:
        """
        Average the values being kept track of.

        Returns:
            Union[float, int]: The average quantity being kept track of.
        """
        return self._average

    @property
    def elements(self) -> int:
        """
        Counts the values being kept track of.

        Returns:
            int: Number of values being kept track of.
        """
        return self._elements

    @property
    def sum(self) -> realnum:
        """
        Sum the values being kept track of.

        Returns:
            Union[float, int]: Sum of the values being kept track of.
        """
        return self.average * self.elements

    def reset(self) -> None:
        """
        Reset the SimpleSumAverager object, as it would be after a fresh instantiation.
        """
        self._average: realnum = 0.0
        self._elements: int = 0
