"""
Sortednp (sorted numpy) is a python package which provides methods to perform
efficient set operations on sorted numpy arrays. This includes intersecting
and merging sorted numpy arrays. The returned intersections and unions are
also sorted.
"""

from sortednp._internal import merge, intersect

__version__ = "0.2.1"

SIMPLE_SEARCH = 1
BINARY_SEARCH = 2
GALLOPING_SEARCH = 3

def resolve(obj):
    """
    Helper function.

    Check whether the given object is callable. If yes, return its return
    value, otherwise return the object itself.
    """
    return obj() if callable(obj) else obj

def kway_merge(*arrays, assume_sorted=True, **kwds):
    """
    Merge all given arrays and return the result. Depending on the optional
    flag assume_sorted, the function sorts the arrays before merging.

    The method raises a TypeError, if the array count is zero.

    The arguments can load the arrays on the fly. If an argument is callable,
    its return value is used as an array instead of the argument itself. This
    make it possible to load one array after another to avoid having all
    arrays in memory at the same time..

    Note on the performance: The function merges the arrays one-by-one. This
    not the most performant implementation. Use the module heapq for more
    efficient ways to merge sorted arrays.
    """
    if not arrays:
        raise TypeError("Merge expects at least one array.")

    arrays = list(arrays)
    merge_result = arrays.pop()
    merge_result = resolve(merge_result)
    if not assume_sorted:
        merge_result.sort()
    for array in arrays:
        array = resolve(array)
        if not assume_sorted:
            array.sort()
        merge_result = merge(merge_result, array, **kwds)
    return merge_result

def kway_intersect(*arrays, assume_sorted=True, **kwds):
    """
    Intersect all given arrays and return the result. Depending on the
    optional flag assume_sorted, the function sort sorts the arrays prior to
    intersecting.

    The method raises a TypeError, if the array count is zero.

    The arguments can load the arrays on the fly. If an argument is callable,
    its return value is used as an array instead of the argument itself. This
    make it possible to load one array after another to avoid having all
    arrays in memory at the same time..

    Note on the performance: The function intersects the arrays one-by-one.
    This is not the most performant implementation.
    """

    if not arrays:
        raise TypeError("Merge expects at least one array.")

    # start with smallest non-callable
    inf = float('inf')
    len_array = [(inf if callable(a) else len(a), a) for a in arrays]
    len_array = sorted(len_array, key=lambda x: x[0])
    arrays = [a for l, a in len_array]

    intersect_result = arrays.pop()
    intersect_result = resolve(intersect_result)
    if not assume_sorted:
        intersect_result.sort()
    for array in arrays:
        array = resolve(array)
        if not assume_sorted:
            array.sort()
        intersect_result = intersect(intersect_result, array, **kwds)
    return intersect_result
