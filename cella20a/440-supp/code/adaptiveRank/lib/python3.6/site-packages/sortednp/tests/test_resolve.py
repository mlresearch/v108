
import unittest
import numpy as np
import sortednp as snp

class ResolveTestCase(unittest.TestCase):
    
    def test_resolve_array(self):
        """
        Check that the resolve function returns the numpy array passed to it.
        """
        a = np.arange(3)

        a = snp.resolve(a)
        self.assertEqual(list(a), [0, 1, 2])
        self.assertEqual(a.dtype, 'int')
    
    def test_resolve_lambda(self):
        """
        Check that the resolve function returns the numpy array returned by
        the lambda passed to it.
        """
        l = lambda: np.arange(3)

        a = snp.resolve(l)
        self.assertEqual(list(a), [0, 1, 2])
        self.assertEqual(a.dtype, 'int')


