
from abc import ABCMeta, abstractmethod
import unittest
import numpy as np

import sortednp as snp

class KWayIntersectBase(metaclass=ABCMeta):
    """
    Define general test cases for the k-way-intersect method. Sub-classes need to
    implement have to overwrite the dtype method.
    """

    @abstractmethod
    def get_dtype(self):
        """
        Returns the numpy data type, which should be used for all tests.
        """
        pass
    

    def test_simple_0_sorted(self):
        """
        Check that a TypeError is raised when no arrays are passed to the
        intersecter.
        """
        self.assertRaises(TypeError, snp.kway_intersect)

    def test_simple_1_sorted(self):
        """
        Check that the same array is returned, if only one sorted array is
        passed to the intersecter.
        """
        a = np.array([1, 3, 7], dtype=self.get_dtype())

        i = snp.kway_intersect(a)
        self.assertEqual(list(i), [1, 3, 7])
        self.assertEqual(i.dtype, self.get_dtype())

    def test_simple_2_sorted(self):
        """
        Check that the correctly intersected array is returned when called with
        two sorted arrays.
        """
        a = np.array([1, 3, 7], dtype=self.get_dtype())
        b = np.array([3, 5, 7, 9], dtype=self.get_dtype())

        i = snp.kway_intersect(a, b)
        self.assertEqual(list(i), [3, 7])
        self.assertEqual(i.dtype, self.get_dtype())

    def test_simple_4_sorted(self):
        """
        Check that the correctly intersected array is returned when called with
        four sorted arrays.
        """
        a = np.array([1, 3, 5, 7], dtype=self.get_dtype())
        b = np.array([0, 1, 3, 7, 9], dtype=self.get_dtype())
        c = np.array([1, 2, 3], dtype=self.get_dtype())
        d = np.array([1, 3, 10], dtype=self.get_dtype())

        i = snp.kway_intersect(a, b, c, d)
        self.assertEqual(list(i), [1, 3])
        self.assertEqual(i.dtype, self.get_dtype())

    def test_simple_0_not_sorted(self):
        """
        Check that a TypeError is raised when no arrays are passed to the
        intersecter and assume_sorted is False.
        """
        self.assertRaises(TypeError, snp.kway_intersect, assume_sorted=False)

    def test_simple_1_not_sorted(self):
        """
        Check that the same array is returned, if only one sorted array is
        passed to the intersecter and assume_sorted is False.
        """
        a = np.array([3, 1, 7], dtype=self.get_dtype())

        i = snp.kway_intersect(a, assume_sorted=False)
        self.assertEqual(list(i), [1, 3, 7])
        self.assertEqual(i.dtype, self.get_dtype())

    def test_simple_2_not_sorted(self):
        """
        Check that the correctly intersected array is returned when called with
        two sorted arrays and assume_sorted is False.
        """
        a = np.array([1, 7, 3], dtype=self.get_dtype())
        b = np.array([9, 3, 5, 7], dtype=self.get_dtype())

        i = snp.kway_intersect(a, b, assume_sorted=False)
        self.assertEqual(list(i), [3, 7])
        self.assertEqual(i.dtype, self.get_dtype())

    def test_simple_4_not_sorted(self):
        """
        Check that the correctly intersected array is returned when called with
        four sorted arrays and assume_sorted is False.
        """
        a = np.array([3, 5, 1, 7], dtype=self.get_dtype())
        b = np.array([3, 7, 1, 9], dtype=self.get_dtype())
        c = np.array([2, 1, 3], dtype=self.get_dtype())
        d = np.array([1, 10, 3], dtype=self.get_dtype())

        i = snp.kway_intersect(a, b, c, d, assume_sorted=False)
        self.assertEqual(list(i), [1, 3])
        self.assertEqual(i.dtype, self.get_dtype())

    def test_simple_1_callable_1(self):
        """
        Check that the generated array is returned if a single generated is
        given.
        """

        a = lambda: np.array([3, 5, 7], dtype=self.get_dtype())

        i = snp.kway_intersect(a)
        self.assertEqual(list(i), [3, 5, 7])
        self.assertEqual(i.dtype, self.get_dtype())

    def test_simple_3_callable_3(self):
        """
        Check that the correctly intersected array is returned when called with
        three callable objects.
        """
        a = lambda: np.array([1, 3, 5, 7], dtype=self.get_dtype())
        b = lambda: np.array([1, 3, 7, 9], dtype=self.get_dtype())
        c = lambda: np.array([1, 2, 3], dtype=self.get_dtype())

        i = snp.kway_intersect(a, b, c)
        self.assertEqual(list(i), [1, 3])
        self.assertEqual(i.dtype, self.get_dtype())

    def test_simple_3_callable_1(self):
        """
        Check that the correctly intersected array is returned when called with
        one callable and two arrays.
        """
        a = lambda: np.array([1, 3, 5, 7], dtype=self.get_dtype())
        b = np.array([1, 3, 7, 9], dtype=self.get_dtype())
        c = np.array([1, 2, 3], dtype=self.get_dtype())

        i = snp.kway_intersect(a, b, c)
        self.assertEqual(list(i), [1, 3])
        self.assertEqual(i.dtype, self.get_dtype())

        i = snp.kway_intersect(b, a, c)
        self.assertEqual(list(i), [1, 3])
        self.assertEqual(i.dtype, self.get_dtype())

        i = snp.kway_intersect(c, b, a)
        self.assertEqual(list(i), [1, 3])
        self.assertEqual(i.dtype, self.get_dtype())

    def test_simple_1_callable_1_not_sorted(self):
        """
        Check that the generated array is returned if a single generated is
        given and assume_sorted is False.
        """

        a = lambda: np.array([3, 7, 5], dtype=self.get_dtype())

        i = snp.kway_intersect(a, assume_sorted=False)
        self.assertEqual(list(i), [3, 5, 7])
        self.assertEqual(i.dtype, self.get_dtype())

    def test_simple_3_callable_3_not_sorted(self):
        """
        Check that the correctly intersected array is returned when called with
        three callable objects and assume_sorted is False.
        """
        a = lambda: np.array([5, 1, 3, 7], dtype=self.get_dtype())
        b = lambda: np.array([4, 3, 1, 9], dtype=self.get_dtype())
        c = lambda: np.array([2, 1, 3], dtype=self.get_dtype())

        i = snp.kway_intersect(a, b, c, assume_sorted=False)
        self.assertEqual(list(i), [1, 3])
        self.assertEqual(i.dtype, self.get_dtype())

    def test_simple_3_callable_1_not_sorted(self):
        """
        Check that the correctly intersected array is returned when called with
        one callable and two arrays and assume_sorted is False.
        """
        a = lambda: np.array([5, 1, 3, 7], dtype=self.get_dtype())
        b = np.array([4, 3, 1, 9], dtype=self.get_dtype())
        c = np.array([2, 1, 3], dtype=self.get_dtype())

        i = snp.kway_intersect(a, b, c, assume_sorted=False)
        self.assertEqual(list(i), [1, 3])
        self.assertEqual(i.dtype, self.get_dtype())

        i = snp.kway_intersect(b, a, c, assume_sorted=False)
        self.assertEqual(list(i), [1, 3])
        self.assertEqual(i.dtype, self.get_dtype())

        i = snp.kway_intersect(c, b, a, assume_sorted=False)
        self.assertEqual(list(i), [1, 3])
        self.assertEqual(i.dtype, self.get_dtype())

class KWayIntersectTestCase_Double(KWayIntersectBase, unittest.TestCase):
    def get_dtype(self):
        return 'float'


