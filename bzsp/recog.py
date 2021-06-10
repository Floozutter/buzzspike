import cv2
import numpy
from numpy import ndarray

def union_of_intersecting_components(a_bimg: ndarray, b_bimg: ndarray) -> ndarray:
    # find the intersecting labels (labels for components that participate in intersections)
    inter = cv2.bitwise_and(a_bimg, b_bimg)
    n, _, stats, _ = cv2.connectedComponentsWithStats(inter)
    _, a_labeled = cv2.connectedComponents(a_bimg)
    _, b_labeled = cv2.connectedComponents(b_bimg)
    a_intersecting_labels: set[int] = set()
    b_intersecting_labels: set[int] = set()
    for inter_label in range(1, n):
        top, left = stats[inter_label, cv2.CC_STAT_TOP], stats[inter_label, cv2.CC_STAT_LEFT]
        a_intersecting_labels.add(a_labeled[top, left])
        b_intersecting_labels.add(b_labeled[top, left])
    # use the intersecting labels to make masks that describe where to bitwise_or
    a_where = numpy.isin(a_labeled, list(a_intersecting_labels))
    b_where = numpy.isin(b_labeled, list(b_intersecting_labels))
    where = a_where | b_where
    return numpy.bitwise_or(a_bimg, b_bimg, out = numpy.zeros_like(inter), where = where)
