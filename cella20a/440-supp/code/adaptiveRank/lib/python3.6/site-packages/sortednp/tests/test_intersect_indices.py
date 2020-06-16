
from abc import ABCMeta, abstractmethod
import sys
import weakref
import unittest
from unittest import TestCase as TC
import numpy as np

import sortednp as snp

class Base(metaclass=ABCMeta):
    """
    Define general test cases for the intersect method. Sub-classes need to
    implement have to overwrite the dtype method.
    """


    def assertListAlmostEqual(self, a, b, *args, **kwds):
        """
        Check that the given lists are almost equal.
        """
        for A, B in zip(a, b):
            self.assertAlmostEqual(A, B, *args, **kwds)

    def test_assertListAlmostEqual_pass(self):
        """
        Check that assertListAlmostEqual raises no exception, if the given
        values are almost equal.
        """
        a = [0, 1, 2 + 1e-9, 10]
        b = [0, 1, 2       , 10]

        self.assertListAlmostEqual(a, b)

    def test_assertListAlmostEqual_fail(self):
        """
        Check that assertListAlmostEqual raises an exception, if the given
        values differ.
        """
        a = [0, 1, 2 + 1e-3, 10]
        b = [0, 1, 2       , 10]

        self.assertRaises(AssertionError, self.assertListAlmostEqual, a, b)
                

    @abstractmethod
    def get_kwds(self):
        """
        Additional keywords passed to the intersect method. By overwriting
        this, test cases can change the search algorithm.
        """
        pass

    @abstractmethod
    def get_dtype(self):
        """
        Returns the numpy data type, which should be used for all tests.
        """
        pass

    def test_issue_example(self):
        """
        Use example from issue !17.
        """
        a = np.array([2,4,6,8,10], dtype=self.get_dtype())
        b = np.array([1,2,3,4], dtype=self.get_dtype())

        i, (i_a, i_b) = snp.intersect(a, b, indices=True, **self.get_kwds())

        self.assertEqual(list(i), [2, 4])
        self.assertEqual(i.dtype, self.get_dtype())

        self.assertEqual(list(i_a), [0, 1])
        self.assertEqual(i_a.dtype, "intp")
        self.assertEqual(list(i_b), [1, 3])
        self.assertEqual(i_b.dtype, "intp")
    
    def test_simple_middle(self):
        """
        Check that intersect returns only elements, which are present in both
        non-empty input arrays. The common elements are neither at the
        beginning nor at the end of the arrays.
        """
        a = np.array([2, 3, 6, 7, 9], dtype=self.get_dtype())
        b = np.array([1, 3, 7, 8, 10], dtype=self.get_dtype())

        i, (i_a, i_b) = snp.intersect(a, b, indices=True, **self.get_kwds())

        self.assertEqual(list(i), [3, 7])
        self.assertEqual(i.dtype, self.get_dtype())

        self.assertEqual(list(i_a), [1, 3])
        self.assertEqual(i_a.dtype, "intp")
        self.assertEqual(list(i_b), [1, 2])
        self.assertEqual(i_b.dtype, "intp")

    def test_simple_end_single(self):
        """
        Check that intersect returns only elements, which are present in both
        non-empty input arrays. One common element is at the end of one array.
        """
        a = np.array([2, 3, 6, 7, 9], dtype=self.get_dtype())
        b = np.array([1, 3, 7, 9, 10], dtype=self.get_dtype())

        i, (i_a, i_b) = snp.intersect(a, b, indices=True, **self.get_kwds())

        self.assertEqual(list(i), [3, 7, 9])
        self.assertEqual(i.dtype, self.get_dtype())

        self.assertEqual(list(i_a), [1, 3, 4])
        self.assertEqual(i_a.dtype, "intp")
        self.assertEqual(list(i_b), [1, 2, 3])
        self.assertEqual(i_b.dtype, "intp")

    def test_simple_end_both(self):
        """
        Check that intersect returns only elements, which are present in both
        non-empty input arrays. One common element is at the end of both
        arrays.
        """
        a = np.array([2, 3, 6, 7, 9], dtype=self.get_dtype())
        b = np.array([1, 3, 7, 9], dtype=self.get_dtype())

        i, (i_a, i_b) = snp.intersect(a, b, indices=True, **self.get_kwds())

        self.assertEqual(list(i), [3, 7, 9])
        self.assertEqual(i.dtype, self.get_dtype())

        self.assertEqual(list(i_a), [1, 3, 4])
        self.assertEqual(i_a.dtype, "intp")
        self.assertEqual(list(i_b), [1, 2, 3])
        self.assertEqual(i_b.dtype, "intp")

    def test_simple_beginning_single(self):
        """
        Check that intersect returns only elements, which are present in both
        non-empty input arrays. One common element is at the beginning of one array.
        """
        a = np.array([2, 3, 6, 7, 8], dtype=self.get_dtype())
        b = np.array([1, 2, 3, 7, 9, 10], dtype=self.get_dtype())

        i, (i_a, i_b) = snp.intersect(a, b, indices=True, **self.get_kwds())

        self.assertEqual(list(i), [2, 3, 7])
        self.assertEqual(i.dtype, self.get_dtype())

        self.assertEqual(list(i_a), [0, 1, 3])
        self.assertEqual(i_a.dtype, "intp")
        self.assertEqual(list(i_b), [1, 2, 3])
        self.assertEqual(i_b.dtype, "intp")

    def test_simple_beginning_both(self):
        """
        Check that intersect returns only elements, which are present in both
        non-empty input arrays. One common element is at the end of both
        arrays.
        """
        a = np.array([2, 3, 6, 7, 8], dtype=self.get_dtype())
        b = np.array([2, 3, 7, 9, 10], dtype=self.get_dtype())

        i, (i_a, i_b) = snp.intersect(a, b, indices=True, **self.get_kwds())

        self.assertEqual(list(i), [2, 3, 7])
        self.assertEqual(i.dtype, self.get_dtype())

        self.assertEqual(list(i_a), [0, 1, 3])
        self.assertEqual(i_a.dtype, "intp")
        self.assertEqual(list(i_b), [0, 1, 2])
        self.assertEqual(i_b.dtype, "intp")

    def test_empty_intersect(self):
        """
        Check that intersect returns an empty array, if the non-empty inputs
        do not have any elements in common.
        """
        a = np.array([1, 3, 5, 10], dtype=self.get_dtype())
        b = np.array([2, 4, 7, 8, 20], dtype=self.get_dtype())

        i, (i_a, i_b) = snp.intersect(a, b, indices=True, **self.get_kwds())

        self.assertEqual(list(i), [])
        self.assertEqual(len(i), 0)
        self.assertEqual(i.dtype, self.get_dtype())

        self.assertEqual(list(i_a), [])
        self.assertEqual(len(i_a), 0)
        self.assertEqual(i_a.dtype, "intp")
        self.assertEqual(list(i_b), [])
        self.assertEqual(len(i_b), 0)
        self.assertEqual(i_b.dtype, "intp")

    def test_empty_input_single(self):
        """
        Check that intersect returns an empty array, if one of the input arrays
        is empty.
        """
        a = np.array([], dtype=self.get_dtype())
        b = np.array([2, 4, 7, 8, 20], dtype=self.get_dtype())

        i, (i_a, i_b) = snp.intersect(a, b, indices=True, **self.get_kwds())
        self.assertEqual(list(i), [])
        self.assertEqual(len(i), 0)
        self.assertEqual(i.dtype, self.get_dtype())
        self.assertEqual(list(i_a), [])
        self.assertEqual(len(i_a), 0)
        self.assertEqual(i_a.dtype, "intp")
        self.assertEqual(list(i_b), [])
        self.assertEqual(len(i_b), 0)
        self.assertEqual(i_b.dtype, "intp")

        i, (i_b, i_a) = snp.intersect(b, a, indices=True, **self.get_kwds())
        self.assertEqual(list(i), [])
        self.assertEqual(len(i), 0)
        self.assertEqual(i.dtype, self.get_dtype())
        self.assertEqual(list(i_a), [])
        self.assertEqual(len(i_a), 0)
        self.assertEqual(i_a.dtype, "intp")
        self.assertEqual(list(i_b), [])
        self.assertEqual(len(i_b), 0)
        self.assertEqual(i_b.dtype, "intp")

    def test_empty_input_both(self):
        """
        Check that intersect returns an empty array, both  input arrays are
        empty.
        """
        a = np.array([], dtype=self.get_dtype())
        b = np.array([], dtype=self.get_dtype())

        i, (i_a, i_b) = snp.intersect(a, b, indices=True, **self.get_kwds())
        self.assertEqual(list(i), [])
        self.assertEqual(len(i), 0)
        self.assertEqual(i.dtype, self.get_dtype())

        self.assertEqual(list(i_a), [])
        self.assertEqual(len(i_a), 0)
        self.assertEqual(i_a.dtype, "intp")
        self.assertEqual(list(i_b), [])
        self.assertEqual(len(i_b), 0)
        self.assertEqual(i_b.dtype, "intp")

    def test_contained(self):
        """
        Check that intersect returns the common elements, if one array is
        contained in the other.
        """
        a = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=self.get_dtype())
        b = np.array([4, 5, 7], dtype=self.get_dtype())

        i, (i_a, i_b) = snp.intersect(a, b, indices=True, **self.get_kwds())
        self.assertEqual(list(i), [4, 5, 7])
        self.assertEqual(i.dtype, self.get_dtype())

        self.assertEqual(list(i_a), [3, 4, 6])
        self.assertEqual(i_a.dtype, "intp")
        self.assertEqual(list(i_b), [0, 1, 2])
        self.assertEqual(i_b.dtype, "intp")

        i, (i_b, i_a) = snp.intersect(b, a, indices=True, **self.get_kwds())
        self.assertEqual(list(i), [4, 5, 7])
        self.assertEqual(i.dtype, self.get_dtype())

        self.assertEqual(list(i_a), [3, 4, 6])
        self.assertEqual(i_a.dtype, "intp")
        self.assertEqual(list(i_b), [0, 1, 2])
        self.assertEqual(i_b.dtype, "intp")

    def test_identical(self):
        """
        Check that intersect returns a copy, if the same array is passed in
        twice.
        """
        a = np.array([3, 4, 6, 8], dtype=self.get_dtype())

        i, (i_a, i_a_prime) = snp.intersect(a, a, indices=True, **self.get_kwds())
        self.assertEqual(list(i), [3, 4, 6, 8])
        self.assertEqual(list(a), [3, 4, 6, 8])
        self.assertEqual(i.dtype, self.get_dtype())

        self.assertEqual(list(i_a), [0, 1, 2, 3])
        self.assertEqual(i_a.dtype, "intp")
        self.assertEqual(list(i_a_prime), [0, 1, 2, 3])
        self.assertEqual(i_a_prime.dtype, "intp")

        i[0] = 1
        self.assertEqual(list(a), [3, 4, 6, 8])

    def test_separated(self):
        """
        Check that intersect returns an empty array, if all elements are
        greater than all elements in the other.
        """
        a = np.array([1, 2, 3], dtype=self.get_dtype())
        b = np.array([4, 6, 8], dtype=self.get_dtype())

        i, (i_a, i_b) = snp.intersect(a, b, indices=True, **self.get_kwds())
        self.assertEqual(list(i), [])
        self.assertEqual(len(i), 0)
        self.assertEqual(i.dtype, self.get_dtype())

        self.assertEqual(list(i_a), [])
        self.assertEqual(len(i_a), 0)
        self.assertEqual(i_a.dtype, "intp")
        self.assertEqual(list(i_b), [])
        self.assertEqual(len(i_b), 0)
        self.assertEqual(i_b.dtype, "intp")

    def test_duplicates_same(self):
        """
        Check that duplicates in the same array are dropped if not present in
        the other array.
        """
        a = np.array([1, 2, 2, 3], dtype=self.get_dtype())
        b = np.array([1, 6, 8], dtype=self.get_dtype())

        i, (i_a, i_b) = snp.intersect(a, b, indices=True, **self.get_kwds())
        self.assertEqual(list(i), [1])
        self.assertEqual(i.dtype, self.get_dtype())

        self.assertEqual(list(i_a), [0])
        self.assertEqual(i_a.dtype, "intp")
        self.assertEqual(list(i_b), [0])
        self.assertEqual(i_b.dtype, "intp")

        a = np.array([1, 2, 2, 3], dtype=self.get_dtype())
        b = np.array([1, 2, 4, 6, 8], dtype=self.get_dtype())

        i, (i_a, i_b) = snp.intersect(a, b, indices=True, **self.get_kwds())
        self.assertEqual(list(i), [1, 2])
        self.assertEqual(i.dtype, self.get_dtype())

        self.assertEqual(list(i_a), [0, 1])
        self.assertEqual(i_a.dtype, "intp")
        self.assertEqual(list(i_b), [0, 1])
        self.assertEqual(i_b.dtype, "intp")

    def test_duplicates_both(self):
        """
        Check that duplicates in the same array are kept if they are also
        duplicated in the other array.
        """
        a = np.array([1, 2, 2, 3], dtype=self.get_dtype())
        b = np.array([2, 2, 4, 6, 8], dtype=self.get_dtype())

        i, (i_a, i_b) = snp.intersect(a, b, indices=True, **self.get_kwds())
        self.assertEqual(list(i), [2, 2])
        self.assertEqual(i.dtype, self.get_dtype())

        self.assertEqual(list(i_a), [1, 2])
        self.assertEqual(i_a.dtype, "intp")
        self.assertEqual(list(i_b), [0, 1])
        self.assertEqual(i_b.dtype, "intp")

    def test_raise_multi_dim(self):
        """
        Check that passing in a multi dimensional array raises an exception.
        """
        a = np.zeros((10, 2), dtype=self.get_dtype())
        b = np.array([2, 3, 5, 6], dtype=self.get_dtype())

        self.assertRaises(ValueError, snp.intersect, a, b, indices=True, **self.get_kwds())
        self.assertRaises(ValueError, snp.intersect, b, a, indices=True, **self.get_kwds())
        self.assertRaises(ValueError, snp.intersect, a, a, indices=True, **self.get_kwds())
        
    def test_raise_non_array(self):
        """
        Check that passing in a non-numpy-array raises an exception.
        """
        b = np.array([2, 3, 5, 6], dtype=self.get_dtype())

        self.assertRaises(TypeError, snp.intersect, 3, b, indices=True, **self.get_kwds())
        self.assertRaises(TypeError, snp.intersect, b, 2, indices=True, **self.get_kwds())
        self.assertRaises(TypeError, snp.intersect, 3, "a", indices=True, **self.get_kwds())

    def test_reference_counting_principle(self):
        """
        Check that the reference counting works as expected with standard
        numpy arrays.
        """

        # Create inputs
        a = np.arange(10, dtype=self.get_dtype()) * 3
        b = np.arange(10, dtype=self.get_dtype()) * 2 + 5

        # Check ref count for input. Numpy arrays have two references.
        self.assertEqual(sys.getrefcount(a), 2)
        self.assertEqual(sys.getrefcount(b), 2)

        # Create weak refs for inputs
        weak_a = weakref.ref(a)
        weak_b = weakref.ref(b)

        self.assertEqual(sys.getrefcount(a), 2)
        self.assertEqual(sys.getrefcount(b), 2)
        self.assertIsNotNone(weak_a())
        self.assertIsNotNone(weak_b())

        # Delete a
        del a
        self.assertEqual(sys.getrefcount(b), 2)
        self.assertIsNone(weak_a())
        self.assertIsNotNone(weak_b())

        # Delete b
        del b
        self.assertIsNone(weak_a())
        self.assertIsNone(weak_b())

    def test_reference_counting(self):
        """
        Check that the reference counting is done correctly.
        """

        # Create inputs
        a = np.arange(10, dtype=self.get_dtype()) * 3
        b = np.arange(10, dtype=self.get_dtype()) * 2 + 5

        # Check ref count for input. Numpy arrays have two references.
        self.assertEqual(sys.getrefcount(a), 2)
        self.assertEqual(sys.getrefcount(b), 2)

        # Create weak refs for inputs
        weak_a = weakref.ref(a)
        weak_b = weakref.ref(b)

        self.assertEqual(sys.getrefcount(a), 2)
        self.assertEqual(sys.getrefcount(b), 2)
        self.assertIsNotNone(weak_a())
        self.assertIsNotNone(weak_b())

        ## Intersect
        i, (i_a, i_b) = snp.intersect(a, b, indices=True, **self.get_kwds())

        self.assertEqual(sys.getrefcount(a), 2)
        self.assertEqual(sys.getrefcount(b), 2)
        self.assertEqual(sys.getrefcount(i), 2)
        self.assertEqual(sys.getrefcount(i_a), 2)
        self.assertEqual(sys.getrefcount(i_b), 2)

        # Create weakrefs
        weak_i = weakref.ref(i)
        weak_i_a = weakref.ref(i_a)
        weak_i_b = weakref.ref(i_b)
        self.assertEqual(sys.getrefcount(a), 2)
        self.assertEqual(sys.getrefcount(b), 2)
        self.assertEqual(sys.getrefcount(i), 2)
        self.assertEqual(sys.getrefcount(i_a), 2)
        self.assertEqual(sys.getrefcount(i_b), 2)
        self.assertIsNotNone(weak_a())
        self.assertIsNotNone(weak_b())
        self.assertIsNotNone(weak_i())
        self.assertIsNotNone(weak_i_a())
        self.assertIsNotNone(weak_i_b())

        # Delete a
        del a
        self.assertEqual(sys.getrefcount(b), 2)
        self.assertEqual(sys.getrefcount(i), 2)
        self.assertEqual(sys.getrefcount(i_a), 2)
        self.assertEqual(sys.getrefcount(i_b), 2)
        self.assertIsNone(weak_a())
        self.assertIsNotNone(weak_b())
        self.assertIsNotNone(weak_i())
        self.assertIsNotNone(weak_i_a())
        self.assertIsNotNone(weak_i_b())

        # Delete b
        del b
        self.assertEqual(sys.getrefcount(i), 2)
        self.assertEqual(sys.getrefcount(i_a), 2)
        self.assertEqual(sys.getrefcount(i_b), 2)
        self.assertIsNone(weak_a())
        self.assertIsNone(weak_b())
        self.assertIsNotNone(weak_i())
        self.assertIsNotNone(weak_i_a())
        self.assertIsNotNone(weak_i_b())

        # Delete i
        del i
        self.assertEqual(sys.getrefcount(i_a), 2)
        self.assertEqual(sys.getrefcount(i_b), 2)
        self.assertIsNone(weak_a())
        self.assertIsNone(weak_b())
        self.assertIsNone(weak_i())
        self.assertIsNotNone(weak_i_a())
        self.assertIsNotNone(weak_i_b())

        # Delete i_a
        del i_a
        self.assertEqual(sys.getrefcount(i_b), 2)
        self.assertIsNone(weak_a())
        self.assertIsNone(weak_b())
        self.assertIsNone(weak_i())
        self.assertIsNone(weak_i_a())
        self.assertIsNotNone(weak_i_b())

        # Delete i_b
        del i_b
        self.assertIsNone(weak_a())
        self.assertIsNone(weak_b())
        self.assertIsNone(weak_i())
        self.assertIsNone(weak_i_a())
        self.assertIsNone(weak_i_b())

class ITC_Double:
    def get_dtype(self):
        return 'float64'

    def test_type_limites(self):
        """
        Ensure that intersecting works with numbers specific to this data type.
        """
        a = np.array([-1.3e300, -1.2e300, -2.3e-200, 3.14e20], dtype=self.get_dtype())
        b = np.array([-1.3e300, -1.1e300, -2.3e-200, 3.14e20], dtype=self.get_dtype())

        i = snp.intersect(a, b, **self.get_kwds())
        self.assertListAlmostEqual(list(i), [-1.3e300, -2.3e-200, 3.14e20])
        self.assertEqual(i.dtype, self.get_dtype())

class ITC_Float:
    def get_dtype(self):
        return 'float32'

    def test_type_limites(self):
        """
        Ensure that intersecting works with numbers specific to this data type.
        """
        a = np.array([-1.3e30, -1.2e30, -2.3e-20, 3.14e20], dtype=self.get_dtype())
        b = np.array([-1.3e30, -1.1e30, -2.3e-20, 3.14e20], dtype=self.get_dtype())

        i = snp.intersect(a, b, **self.get_kwds())
        i_corr = np.array([-1.3e30, -2.3e-20, 3.14e20],
            dtype=self.get_dtype())
        self.assertListAlmostEqual(list(i), list(i_corr), places=3)
        self.assertEqual(i.dtype, self.get_dtype())


class ITC_Int8:
    def get_dtype(self):
        return 'int8'
    def test_type_limites(self):
        """
        Ensure that intersecting works with numbers specific to this data type.
        """
        a = np.array([-128, 3, 4, 127], dtype=self.get_dtype())
        b = np.array([-128, 3, 2, 127], dtype=self.get_dtype())

        i = snp.intersect(a, b, **self.get_kwds())
        self.assertEqual(list(i), [-128, 3, 127])
        self.assertEqual(i.dtype, self.get_dtype())

class ITC_Int16:
    def get_dtype(self):
        return 'int16'
    def test_type_limites(self):
        """
        Ensure that intersecting works with numbers specific to this data type.
        """
        a = np.array([-32768, 3, 4, 32767], dtype=self.get_dtype())
        b = np.array([-32768, 3, 2, 32767], dtype=self.get_dtype())

        i = snp.intersect(a, b, **self.get_kwds())
        self.assertEqual(list(i), [-32768, 3, 32767])
        self.assertEqual(i.dtype, self.get_dtype())

class ITC_Int32:
    def get_dtype(self):
        return 'int32'
    def test_type_limites(self):
        """
        Ensure that intersecting works with numbers specific to this data type.
        """
        a = np.array([-2147483647, 3, 4, 2147483647], dtype=self.get_dtype())
        b = np.array([-2147483647, 3, 2, 2147483647], dtype=self.get_dtype())

        i = snp.intersect(a, b, **self.get_kwds())
        self.assertEqual(list(i), [-2147483647, 3, 2147483647])
        self.assertEqual(i.dtype, self.get_dtype())

class ITC_Int64:
    def get_dtype(self):
        return 'int64'
    def test_type_limites(self):
        """
        Ensure that intersecting works with numbers specific to this data type.
        """
        a = np.array([-9223372036854775807, 3, 4, 9223372036854775807], dtype=self.get_dtype())
        b = np.array([-9223372036854775807, 3, 2, 9223372036854775807], dtype=self.get_dtype())

        i = snp.intersect(a, b, **self.get_kwds())
        self.assertEqual(list(i), [-9223372036854775807, 3, 9223372036854775807])
        self.assertEqual(i.dtype, self.get_dtype())
 
 
class ITC_UInt8:
    def get_dtype(self):
        return 'uint8'
        a = np.array([0, 3, 4, 255], dtype=self.get_dtype())
        b = np.array([0, 3, 2, 255], dtype=self.get_dtype())

        i = snp.intersect(a, b, **self.get_kwds())
        self.assertEqual(list(i), [0, 3, 255])
        self.assertEqual(i.dtype, self.get_dtype())

class ITC_UInt16:
    def get_dtype(self):
        return 'uint16'
    def test_type_limites(self):
        """
        Ensure that intersecting works with numbers specific to this data type.
        """
        a = np.array([0, 3, 4, 65535], dtype=self.get_dtype())
        b = np.array([0, 3, 2, 65535], dtype=self.get_dtype())

        i = snp.intersect(a, b, **self.get_kwds())
        self.assertEqual(list(i), [0, 3, 65535])
        self.assertEqual(i.dtype, self.get_dtype())

class ITC_UInt32:
    def get_dtype(self):
        return 'uint32'
    def test_type_limites(self):
        """
        Ensure that intersecting works with numbers specific to this data type.
        """
        a = np.array([0, 3, 4, 4294967295], dtype=self.get_dtype())
        b = np.array([0, 3, 2, 4294967295], dtype=self.get_dtype())

        i = snp.intersect(a, b, **self.get_kwds())
        self.assertEqual(list(i), [0, 3, 4294967295])
        self.assertEqual(i.dtype, self.get_dtype())

class ITC_UInt64:
    def get_dtype(self):
        return 'uint64'
    def test_type_limites(self):
        """
        Ensure that intersecting works with numbers specific to this data type.
        """
        a = np.array([0, 3, 4, 18446744073709551615], dtype=self.get_dtype())
        b = np.array([0, 3, 2, 18446744073709551615], dtype=self.get_dtype())

        i = snp.intersect(a, b, **self.get_kwds())
        self.assertEqual(list(i), [0, 3, 18446744073709551615])
        self.assertEqual(i.dtype, self.get_dtype())

class ITC_Default:
    def get_kwds(self):
        """
        Use default value of search algorithm.
        """
        return {}

class ITC_Simple:
    def get_kwds(self):
        """
        Use simple search.
        """
        return {"algorithm": snp.SIMPLE_SEARCH}

class ITC_Binary:
    def get_kwds(self):
        """
        Use binary search.
        """
        return {"algorithm": snp.BINARY_SEARCH}

class ITC_Galloping:
    def get_kwds(self):
        """
        Use galloping search.
        """
        return {"algorithm": snp.GALLOPING_SEARCH}

class ITC_Default_Double(ITC_Default, ITC_Double, TC, Base): pass
class ITC_Default_Float(ITC_Default, ITC_Float, TC, Base): pass
class ITC_Default_Int8(ITC_Default, ITC_Int8, TC, Base): pass
class ITC_Default_Int16(ITC_Default, ITC_Int16, TC, Base): pass
class ITC_Default_Int32(ITC_Default, ITC_Int32, TC, Base): pass
class ITC_Default_Int64(ITC_Default, ITC_Int64, TC, Base): pass
class ITC_Default_UInt8(ITC_Default, ITC_UInt8, TC, Base): pass
class ITC_Default_UInt16(ITC_Default, ITC_UInt16, TC, Base): pass
class ITC_Default_UInt32(ITC_Default, ITC_UInt32, TC, Base): pass
class ITC_Default_UInt64(ITC_Default, ITC_UInt64, TC, Base): pass

class ITC_Simple_Double(ITC_Simple, ITC_Double, TC, Base): pass
class ITC_Simple_Float(ITC_Simple, ITC_Float, TC, Base): pass
class ITC_Simple_Int8(ITC_Simple, ITC_Int8, TC, Base): pass
class ITC_Simple_Int16(ITC_Simple, ITC_Int16, TC, Base): pass
class ITC_Simple_Int32(ITC_Simple, ITC_Int32, TC, Base): pass
class ITC_Simple_Int64(ITC_Simple, ITC_Int64, TC, Base): pass
class ITC_Simple_UInt8(ITC_Simple, ITC_UInt8, TC, Base): pass
class ITC_Simple_UInt16(ITC_Simple, ITC_UInt16, TC, Base): pass
class ITC_Simple_UInt32(ITC_Simple, ITC_UInt32, TC, Base): pass
class ITC_Simple_UInt64(ITC_Simple, ITC_UInt64, TC, Base): pass

class ITC_Binary_Double(ITC_Binary, ITC_Double, TC, Base): pass
class ITC_Binary_Float(ITC_Binary, ITC_Float, TC, Base): pass
class ITC_Binary_Int8(ITC_Binary, ITC_Int8, TC, Base): pass
class ITC_Binary_Int16(ITC_Binary, ITC_Int16, TC, Base): pass
class ITC_Binary_Int32(ITC_Binary, ITC_Int32, TC, Base): pass
class ITC_Binary_Int64(ITC_Binary, ITC_Int64, TC, Base): pass
class ITC_Binary_UInt8(ITC_Binary, ITC_UInt8, TC, Base): pass
class ITC_Binary_UInt16(ITC_Binary, ITC_UInt16, TC, Base): pass
class ITC_Binary_UInt32(ITC_Binary, ITC_UInt32, TC, Base): pass
class ITC_Binary_UInt64(ITC_Binary, ITC_UInt64, TC, Base): pass

class ITC_Galloping_Double(ITC_Galloping, ITC_Double, TC, Base): pass
class ITC_Galloping_Float(ITC_Galloping, ITC_Float, TC, Base): pass
class ITC_Galloping_Int8(ITC_Galloping, ITC_Int8, TC, Base): pass
class ITC_Galloping_Int16(ITC_Galloping, ITC_Int16, TC, Base): pass
class ITC_Galloping_Int32(ITC_Galloping, ITC_Int32, TC, Base): pass
class ITC_Galloping_Int64(ITC_Galloping, ITC_Int64, TC, Base): pass
class ITC_Galloping_UInt8(ITC_Galloping, ITC_UInt8, TC, Base): pass
class ITC_Galloping_UInt16(ITC_Galloping, ITC_UInt16, TC, Base): pass
class ITC_Galloping_UInt32(ITC_Galloping, ITC_UInt32, TC, Base): pass
class ITC_Galloping_UInt64(ITC_Galloping, ITC_UInt64, TC, Base): pass

class ITC_TypeError:
    def test_invalid_type(self):
        """
        Ensure that intersect raises an exception, if it is called with an
        unsupported type.
        """
        a = np.array([1, 3, 7], dtype='complex')
        b = np.array([2, 5, 6], dtype='complex')

        self.assertRaises(ValueError, snp.intersect, a, b, **self.get_kwds())

    def test_different_types(self):
        """
        Ensure that intersect raises an exception, if it is called with two
        different types.
        """
        a = np.array([1, 3, 7], dtype='float32')
        b = np.array([2, 5, 6], dtype='float64')

        self.assertRaises(ValueError, snp.intersect, a, b, **self.get_kwds())


class ITC_Default_TypeError(ITC_Default, ITC_TypeError, TC): pass
class ITC_Simple_TypeError(ITC_Simple, ITC_TypeError, TC): pass
class ITC_Binary_TypeError(ITC_Binary, ITC_TypeError, TC): pass
class ITC_Galloping_TypeError(ITC_Galloping, ITC_TypeError, TC): pass
