import unittest
import numpy as np
import itertools
from snnsimsnn.concepts.virtual_neuron import run as virtual_neuron


class TestAddition(unittest.TestCase):
    """ """

    # Initialize test object
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rng = np.random.default_rng(12345)

    def test_main_functions(self):
        # run algorithm 1
        # for _ in range(100):
        #     a = self.rng.integers(0, 5)
        #     b = self.rng.integers(0, 5)
        abs = list(itertools.permutations(list(range(10)), 2))
        for a, b in abs:
            print(a, b)
            result = virtual_neuron(a=a, b=b, plotting=False)
            expected_result = a + b
            self.assertEqual(expected_result, result)


if __name__ == "__main__":
    unittest.main()
