import numpy as np
import torch
from path import Path


class ModelSaver():

    def __init__(self):
        self._previous_acc = 0.
        self._current_acc = 0.
        self.fold = 0

    @property
    def previous_acc(self):
        return self._previous_acc

    @property
    def current_acc(self):
        return self._current_acc

    @current_acc.setter
    def current_acc(self, value):
        self._current_acc = value

    @previous_acc.setter
    def previous_acc(self, value):
        self._previous_acc = value

    def __set_accuracy(self, accuracy):
        self.previous_acc, self.current_acc = self.current_acc, accuracy

    def save_if_best(self, accuracy, state_dict, fold):

        if self.fold < fold:
            self.fold = fold
            self._current_acc = 0.
            self._previous_acc = 0.

        if accuracy > self.current_acc:
            self.__set_accuracy(accuracy)
            torch.save(state_dict, '{}_best_state.pkl'.format(fold))
