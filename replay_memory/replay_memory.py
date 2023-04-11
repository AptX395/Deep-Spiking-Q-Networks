# coding =utf-8

import random


class ReplayMemory:
    def __init__(self, max_size):
        self._max_size = max_size
        self._memory = list()
        self._cur_index = 0
        self._length = 0

    def append(self, transition):
        if self._length < self._max_size:
            self._memory.append(transition)
            self._length += 1
        else:
            self._cur_index = self._cur_index % self._max_size
            self._memory[self._cur_index] = transition
            self._cur_index += 1

    def sample(self, batch_size):
        minibatch = random.sample(self._memory, batch_size)
        frame_stacks, actions, rewards, masks, next_frame_stacks = zip(*minibatch)
        return frame_stacks, actions, rewards, masks, next_frame_stacks
