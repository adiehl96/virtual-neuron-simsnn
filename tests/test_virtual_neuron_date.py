import unittest
import numpy as np
from snnsimsnn.concepts.reimplementation_date.constant_function import (
    run as constant_function,
)
from snnsimsnn.concepts.reimplementation_date.multiply_neg_one import (
    run as multiply_neg_one,
)

from snnsimsnn.concepts.reimplementation_date.two_addition import (
    run as two_addition,
)

from snnsimsnn.concepts.reimplementation_date.n_addition import run as n_addition


class TestVirtualNeuron(unittest.TestCase):
    """ """

    # Initialize test object
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rng = np.random.default_rng(12345)

    def test_constant_function(self):
        for _ in range(10):
            if not constant_function(plotting=False):
                raise Exception("Incorrect Results!")

    def test_multiply_neg_one_function(self):
        for _ in range(10):
            if not multiply_neg_one(plotting=False):
                raise Exception("Incorrect Results!")

    def test_two_addition_function(self):
        for _ in range(10):
            if not two_addition(plotting=False):
                raise Exception("Incorrect Results!")

    def test_n_addition_function(self):
        for _ in range(10):
            if not n_addition(plotting=False):
                raise Exception("Incorrect Results!")


if __name__ == "__main__":
    unittest.main()
