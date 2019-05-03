#!/usr/bin/env python3

import torch
import unittest
from test.means._base_mean_test_case import BaseMeanTestCase
from gpytorch.means import LinearMean


class TestLinearMean(BaseMeanTestCase, unittest.TestCase):
    def create_mean(self):
        return LinearMean(input_size=4, bias=True)


class TestLinearMeanBatch(BaseMeanTestCase, unittest.TestCase):
    batch_shape = torch.Size([3])

    def create_mean(self):
        return LinearMean(input_size=4, batch_shape=self.__class__.batch_shape, bias=True)


class TestLinearMeanMultiBatch(BaseMeanTestCase, unittest.TestCase):
    batch_shape = torch.Size([5, 2])

    def create_mean(self):
        return LinearMean(input_size=4, batch_shape=self.__class__.batch_shape, bias=True)
