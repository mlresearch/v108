
from abc import ABCMeta, abstractmethod
import sys
import weakref
import unittest
import numpy as np

import sortednp as snp

class MergeBase(metaclass=ABCMeta):
    """
    Define general test cases for the merge method. Sub-classes need to
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
    def get_dtype(self):
        """
        Returns the numpy data type, which should be used for all tests.
        """
        pass
    

    def test_simple(self):
        """
        Check that merging two non-empty arrays returns the union of the two
        arrays.
        """
        a = np.array([1, 3, 7], dtype=self.get_dtype())
        b = np.array([2, 5, 6], dtype=self.get_dtype())

        m = snp.merge(a, b)
        self.assertEqual(list(m), [1, 2, 3, 5, 6, 7])
        self.assertEqual(m.dtype, self.get_dtype())

    def test_separated(self):
        """
        Check that merging two non-empty arrays returns the union of the two
        arrays if all element in on array are greater than all elements in the
        other. This tests the copy parts of the implementation.
        """
        a = np.array([1, 3, 7], dtype=self.get_dtype())
        b = np.array([9, 10, 16], dtype=self.get_dtype())

        m = snp.merge(a, b)
        self.assertEqual(list(m), [1, 3, 7, 9, 10, 16])
        self.assertEqual(m.dtype, self.get_dtype())

    def test_empty_single(self):
        """
        Check that merging two arrays returns a copy of the first one if
        the other is empty.
        """
        a = np.array([1, 3, 7], dtype=self.get_dtype())
        b = np.array([], dtype=self.get_dtype())

        m = snp.merge(a, b)
        self.assertEqual(list(m), [1, 3, 7])
        self.assertEqual(list(a), [1, 3, 7])
        self.assertEqual(m.dtype, self.get_dtype())
        m[0] = 0
        self.assertEqual(list(a), [1, 3, 7])

        m = snp.merge(b, a)
        self.assertEqual(list(m), [1, 3, 7])
        self.assertEqual(list(a), [1, 3, 7])
        self.assertEqual(m.dtype, self.get_dtype())
        m[0] = 0
        self.assertEqual(list(a), [1, 3, 7])


    def test_empty_both(self):
        """
        Check that merging two empty arrays returns an empty array.
        """
        a = np.array([], dtype=self.get_dtype())
        b = np.array([], dtype=self.get_dtype())

        m = snp.merge(a, b)
        self.assertEqual(list(m), [])
        self.assertEqual(len(m), 0)
        self.assertEqual(m.dtype, self.get_dtype())


    def test_identical(self):
        """
        Check that merging two identical arrays returns each element twice.
        """
        a = np.array([1, 3, 7], dtype=self.get_dtype())

        m = snp.merge(a, a)
        self.assertEqual(list(m), [1, 1, 3, 3, 7, 7])
        self.assertEqual(m.dtype, self.get_dtype())


    def test_duplicates_same(self):
        """
        Check that duplications in a single array are passed to the result.
        """
        a = np.array([1, 3, 3, 7], dtype=self.get_dtype())
        b = np.array([2, 5, 6], dtype=self.get_dtype())

        m = snp.merge(a, b)
        self.assertEqual(list(m), [1, 2, 3, 3, 5, 6, 7])
        self.assertEqual(m.dtype, self.get_dtype())

    def test_duplicates_other(self):
        """
        Check that duplications in the other array are passed to the result.
        """
        a = np.array([1, 3, 7], dtype=self.get_dtype())
        b = np.array([2, 3, 5, 6], dtype=self.get_dtype())

        m = snp.merge(a, b)
        self.assertEqual(list(m), [1, 2, 3, 3, 5, 6, 7])
        self.assertEqual(m.dtype, self.get_dtype())

    def test_duplicates_both(self):
        """
        Check that duplications in a single and the other array are both passed to
        the result.
        """
        a = np.array([1, 3, 3, 7], dtype=self.get_dtype())
        b = np.array([2, 3, 5, 6], dtype=self.get_dtype())

        m = snp.merge(a, b)
        self.assertEqual(list(m), [1, 2, 3, 3, 3, 5, 6, 7])
        self.assertEqual(m.dtype, self.get_dtype())
        

    def test_raise_multi_dim(self):
        """
        Check that passing in a multi dimensional array raises an exception.
        """
        a = np.zeros((10, 2), dtype=self.get_dtype())
        b = np.array([2, 3, 5, 6], dtype=self.get_dtype())

        self.assertRaises(ValueError, snp.merge, a, b)
        self.assertRaises(ValueError, snp.merge, b, a)
        self.assertRaises(ValueError, snp.merge, a, a)
        
    def test_raise_non_array(self):
        """
        Check that passing in a non-numpy-array raises an exception.
        """
        b = np.array([2, 3, 5, 6], dtype=self.get_dtype())

        self.assertRaises(TypeError, snp.merge, 3, b)
        self.assertRaises(TypeError, snp.merge, b, 2)
        self.assertRaises(TypeError, snp.merge, 3, "a")
        
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
        m = snp.merge(a, b)

        self.assertEqual(sys.getrefcount(a), 2)
        self.assertEqual(sys.getrefcount(b), 2)
        self.assertEqual(sys.getrefcount(m), 2)

        # Create weakref for m
        weak_m = weakref.ref(m)
        self.assertEqual(sys.getrefcount(a), 2)
        self.assertEqual(sys.getrefcount(b), 2)
        self.assertEqual(sys.getrefcount(m), 2)
        self.assertIsNotNone(weak_a())
        self.assertIsNotNone(weak_b())
        self.assertIsNotNone(weak_m())

        # Delete a
        del a
        self.assertEqual(sys.getrefcount(b), 2)
        self.assertEqual(sys.getrefcount(m), 2)
        self.assertIsNone(weak_a())
        self.assertIsNotNone(weak_b())
        self.assertIsNotNone(weak_m())

        # Delete b
        del b
        self.assertEqual(sys.getrefcount(m), 2)
        self.assertIsNone(weak_a())
        self.assertIsNone(weak_b())
        self.assertIsNotNone(weak_m())

        # Delete m
        del m
        self.assertIsNone(weak_a())
        self.assertIsNone(weak_b())
        self.assertIsNone(weak_m())

    def test_reference_counting_early_exit_type(self):
        """
        Check that the reference counts of the input arrary does not change
        even when the the method exists premature due to incompatible inputs
        types.
        """
        a = np.array(10)

        self.assertEqual(sys.getrefcount(a), 2)
        self.assertRaises(TypeError, snp.merge, a, [1, 2])
        self.assertEqual(sys.getrefcount(a), 2)

        self.assertEqual(sys.getrefcount(a), 2)
        self.assertRaises(TypeError, snp.merge, [1, 2], a)
        self.assertEqual(sys.getrefcount(a), 2)

    def test_reference_counting_early_exit_dim(self):
        """
        Check that the reference counts of the input arrary does not change
        even when the the method exists premature due multidimensional input
        arrays.
        """
        a = np.zeros((10, 2))
        b = np.arange(10)

        self.assertEqual(sys.getrefcount(a), 2)
        self.assertEqual(sys.getrefcount(b), 2)
        self.assertRaises(ValueError, snp.merge, a, b)
        self.assertEqual(sys.getrefcount(a), 2)
        self.assertEqual(sys.getrefcount(b), 2)

        self.assertEqual(sys.getrefcount(a), 2)
        self.assertRaises(ValueError, snp.merge, b, a)
        self.assertEqual(sys.getrefcount(a), 2)
        self.assertEqual(sys.getrefcount(b), 2)

class MergeTestCase_Double(MergeBase, unittest.TestCase):
    def get_dtype(self):
        return 'float64'

    def test_type_limites(self):
        """
        Ensure that merging works with numbers specific to this data type.
        """
        a = np.array([-1.3e300, -1.2e300, -2.3e-200], dtype=self.get_dtype())
        b = np.array([-1.1e300, 3.14e20], dtype=self.get_dtype())

        m = snp.merge(a, b)
        self.assertListAlmostEqual(list(m),
            [-1.3e300, -1.2e300, -1.1e300, -2.3e-200, 3.14e20])
        self.assertEqual(m.dtype, self.get_dtype())

class MergeTestCase_Float(MergeBase, unittest.TestCase):
    def get_dtype(self):
        return 'float32'

    def test_type_limites(self):
        """
        Ensure that merging works with numbers specific to this data type.
        """
        a = np.array([-1.3e30, -1.2e30, -2.3e-20], dtype=self.get_dtype())
        b = np.array([-1.1e30, 3.14e20], dtype=self.get_dtype())

        m = snp.merge(a, b)
        i_corr = np.array([-1.3e30, -1.2e30, -1.1e30, -2.3e-20, 3.14e20],
            dtype=self.get_dtype())
        self.assertListAlmostEqual(list(m), list(i_corr), places=3)
        self.assertEqual(m.dtype, self.get_dtype())


class MergeTestCase_Int8(MergeBase, unittest.TestCase):
    def get_dtype(self):
        return 'int8'
    def test_type_limites(self):
        """
        Ensure that merging works with numbers specific to this data type.
        """
        a = np.array([2, 127], dtype=self.get_dtype())
        b = np.array([-128, 4], dtype=self.get_dtype())

        m = snp.merge(a, b)
        self.assertEqual(list(m), [-128, 2, 4, 127])
        self.assertEqual(m.dtype, self.get_dtype())

class MergeTestCase_Int16(MergeBase, unittest.TestCase):
    def get_dtype(self):
        return 'int16'
    def test_type_limites(self):
        """
        Ensure that merging works with numbers specific to this data type.
        """
        a = np.array([2, 32767], dtype=self.get_dtype())
        b = np.array([-32768, 4], dtype=self.get_dtype())

        m = snp.merge(a, b)
        self.assertEqual(list(m), [-32768, 2, 4, 32767])
        self.assertEqual(m.dtype, self.get_dtype())

class MergeTestCase_Int32(MergeBase, unittest.TestCase):
    def get_dtype(self):
        return 'int32'
    def test_type_limites(self):
        """
        Ensure that merging works with numbers specific to this data type.
        """
        a = np.array([2, 2147483647], dtype=self.get_dtype())
        b = np.array([-2147483647, 4], dtype=self.get_dtype())

        m = snp.merge(a, b)
        self.assertEqual(list(m), [-2147483647, 2, 4, 2147483647])
        self.assertEqual(m.dtype, self.get_dtype())

class MergeTestCase_Int64(MergeBase, unittest.TestCase):
    def get_dtype(self):
        return 'int64'
    def test_type_limites(self):
        """
        Ensure that merging works with numbers specific to this data type.
        """
        a = np.array([2, 9223372036854775807], dtype=self.get_dtype())
        b = np.array([-9223372036854775807, 4], dtype=self.get_dtype())

        m = snp.merge(a, b)
        self.assertEqual(list(m), [-9223372036854775807, 2, 4, 9223372036854775807])
        self.assertEqual(m.dtype, self.get_dtype())
 
 
class MergeTestCase_UInt8(MergeBase, unittest.TestCase):
    def get_dtype(self):
        return 'uint8'
        a = np.array([2, 255], dtype=self.get_dtype())
        b = np.array([0, 4], dtype=self.get_dtype())

        m = snp.merge(a, b)
        self.assertEqual(list(m), [0, 2, 4, 255])
        self.assertEqual(m.dtype, self.get_dtype())
class MergeTestCase_UInt16(MergeBase, unittest.TestCase):
    def get_dtype(self):
        return 'uint16'
    def test_type_limites(self):
        """
        Ensure that merging works with numbers specific to this data type.
        """
        a = np.array([2, 65535], dtype=self.get_dtype())
        b = np.array([0, 4], dtype=self.get_dtype())

        m = snp.merge(a, b)
        self.assertEqual(list(m), [0, 2, 4, 65535])
        self.assertEqual(m.dtype, self.get_dtype())
class MergeTestCase_UInt32(MergeBase, unittest.TestCase):
    def get_dtype(self):
        return 'uint32'
    def test_type_limites(self):
        """
        Ensure that merging works with numbers specific to this data type.
        """
        a = np.array([2, 4294967295], dtype=self.get_dtype())
        b = np.array([0, 4], dtype=self.get_dtype())

        m = snp.merge(a, b)
        self.assertEqual(list(m), [0, 2, 4, 4294967295])
        self.assertEqual(m.dtype, self.get_dtype())
class MergeTestCase_UInt64(MergeBase, unittest.TestCase):
    def get_dtype(self):
        return 'uint64'
        """
        Ensure that merging works with numbers specific to this data type.
        """
        a = np.array([2, 18446744073709551615], dtype=self.get_dtype())
        b = np.array([0, 4], dtype=self.get_dtype())

        m = snp.merge(a, b)
        self.assertEqual(list(m), [0, 2, 4, 18446744073709551615])
        self.assertEqual(m.dtype, self.get_dtype())


class MergeTestCase_TypeError(unittest.TestCase):
    def test_invalid_type(self):
        """
        Ensure that merge raises an exception, if it is called with an
        unsupported type.
        """
        a = np.array([1, 3, 7], dtype='complex')
        b = np.array([2, 5, 6], dtype='complex')

        self.assertRaises(ValueError, snp.merge, a, b)

    def test_different_types(self):
        """
        Ensure that merge raises an exception, if it is called with two
        different types.
        """
        a = np.array([1, 3, 7], dtype='float32')
        b = np.array([2, 5, 6], dtype='float64')

        self.assertRaises(ValueError, snp.merge, a, b)

class MergeNonCContiguousTestCase(unittest.TestCase):
    """
    Check that merge works correctly with the issues of non-c-contiguous
    arrays. See Issue 22,
    https://gitlab.sauerburger.com/frank/sortednp/issues/22.
    """

    def test_non_cc_second(self):
        """
        Check that using a non-c-contiguous array as the second argument
        returns the correct value.

        Repeat the test 1000 times because memory issues might be flaky.
        """
        for i in range(1000):
            a = np.array([[1, 2, 3], [3, 4, 5], [7, 9, 10]]) 
            nonzero_row, nonzero_col = np.nonzero(a)
            
            y = nonzero_col[0:3]
            x = np.array([0, 1, 5])
            
            try:
                self.assertEqual(list(snp.merge(x, y)), [0, 0, 1, 1, 2, 5])
            except ValueError:
                pass
                # Test case not suiteable for 32-bit

    def test_non_cc_first(self):
        """
        Check that using a non-c-contiguous array as the first argument
        returns the correct value.

        Repeat the test 1000 times because memory issues might be flaky.
        """
        for i in range(1000):
            a = np.array([[1, 2, 3], [3, 4, 5], [7, 9, 10]]) 
            nonzero_row, nonzero_col = np.nonzero(a)
            
            x = nonzero_col[0:3]
            y = np.array([0, 1, 5])
            
            try:
                self.assertEqual(list(snp.merge(x, y)), [0, 0, 1, 1, 2, 5])
            except ValueError:
                pass
                # Test case not suiteable for 32-bit

    def test_non_cc_both(self):
        """
        Check that using a non-c-contiguous array as the both arguments
        returns the correct value.

        Repeat the test 1000 times because memory issues might be flaky.
        """
        for i in range(1000):
            a = np.array([[1, 2, 3], [3, 4, 5], [7, 9, 10]]) 
            nonzero_row, nonzero_col = np.nonzero(a)
            
            x = nonzero_col[0:3]
            y = nonzero_col[0:3]

            try:
                self.assertEqual(list(snp.merge(x, y)), [0, 0, 1, 1, 2, 2])
            except ValueError:
                pass
                # Test case not suiteable for 32-bit
