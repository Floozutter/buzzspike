import cv2
import numpy
from numpy import ndarray
from typing import NamedTuple, Any

class Kill(NamedTuple):
    by: str = ""
    to: str = ""
    using: str = ""

Work = Any

def union_of_intersecting_components(
    a_bimg: ndarray,
    b_bimg: ndarray,
    connectivity = 4
) -> ndarray:
    # find the intersecting labels (labels for components that participate in intersections)
    inter = cv2.bitwise_and(a_bimg, b_bimg)
    n, inter_labeled = cv2.connectedComponents(inter, connectivity)
    _, a_labeled = cv2.connectedComponents(a_bimg, connectivity)
    _, b_labeled = cv2.connectedComponents(b_bimg, connectivity)
    a_intersecting_labels: set[int] = set()
    b_intersecting_labels: set[int] = set()
    for inter_label in range(1, n):
        indices = numpy.where(inter_labeled == inter_label)
        first_row, first_col = indices[0][0], indices[1][0]
        a_intersecting_labels.add(a_labeled[first_row, first_col])
        b_intersecting_labels.add(b_labeled[first_row, first_col])
    # use the intersecting labels to make masks that describe where to bitwise_or
    a_where = numpy.isin(a_labeled, list(a_intersecting_labels))
    b_where = numpy.isin(b_labeled, list(b_intersecting_labels))
    where = a_where | b_where
    return numpy.bitwise_or(a_bimg, b_bimg, out = numpy.zeros_like(inter), where = where)

def killfeed_with_work(image: ndarray) -> tuple[tuple[Kill, ...], Work]:
    # get color segments
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    def morph(segment: ndarray, size: int) -> ndarray:
        kernel = numpy.ones((size, size))
        eroded = cv2.erode(segment, kernel, iterations = 1)
        dilated = cv2.dilate(eroded, kernel, iterations = 2)
        return dilated
    white_seg = morph(cv2.inRange(image, (230, 230, 230, 0), (255, 255, 255, 255)), 3)
    green_seg = morph(cv2.inRange(hsv, (50, 30, 50), (90, 130, 200)), 5)
    red_seg = morph(
        cv2.inRange(hsv, (170, 60, 130), (180, 180, 230)) |
        cv2.inRange(hsv, (0, 60, 130), (10, 180, 230)),
        5
    )
    uoiced = union_of_intersecting_components(
        white_seg,
        union_of_intersecting_components(green_seg, red_seg)
    )
    return (), {
        "white_seg": white_seg,
        "green_seg": green_seg,
        "red_seg": red_seg,
        "uoiced": uoiced,
    }

def killfeed(image: ndarray) -> tuple[Kill, ...]:
    return killfeed_with_work(image)[0]
