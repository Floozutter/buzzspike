import cv2
import numpy
from numpy import ndarray
import functools
from typing import Any, NamedTuple, Iterable

class Kill(NamedTuple):
    by: str = ""
    to: str = ""
    using: str = ""

Work = dict[str, Any]

@functools.cache
def chevron(height: int, width: int, foreground = True) -> ndarray:
    c = cv2.fillConvexPoly(
        numpy.zeros((height, width), dtype = numpy.uint8),
        numpy.array(((0, 0), (width, height // 2), (0, height))),
        255
    )
    return c if foreground else ~c

def detect_chevrons(bimg: ndarray) -> tuple[ndarray, ...]:
    height, width = 30, 18
    t = 0.85
    coefed = cv2.matchTemplate(bimg, chevron(height, width), cv2.TM_CCOEFF_NORMED)
    _, threshed = cv2.threshold(numpy.float32(coefed), t, 255, cv2.THRESH_BINARY)
    n, labeled = cv2.connectedComponents(numpy.uint8(threshed))
    component_masks = (
        numpy.array(numpy.where(labeled == label, 255, 0), dtype = numpy.uint8)
        for label in range(1, n)
    )
    max_locs = (cv2.minMaxLoc(coefed, mask = mask)[3] for mask in component_masks)
    return tuple(
        numpy.array((x, y, width, height))
        for x, y in max_locs
    )

def keep_inverted_chevrons(chevrons: Iterable[ndarray], bimg: ndarray) -> tuple[ndarray, ...]:
    def predicate(c: ndarray) -> bool:
        x, y, width, height = c
        view = bimg[y: y + height, x: x + width]
        coef = cv2.matchTemplate(view, chevron(height, width, False), cv2.TM_CCOEFF_NORMED)
        return coef[0, 0] >= 0.25
    return tuple(filter(predicate, chevrons))

def killfeed_with_work(image: ndarray) -> tuple[tuple[Kill, ...], Work]:
    # get color segments
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green_segment = cv2.inRange(hsv, (60, 45, 50), (100, 150, 220))
    red_segment = numpy.bitwise_or(
        cv2.inRange(hsv, (170, 100, 170), (180, 180, 250)),
        cv2.inRange(hsv, (  0, 100, 170), ( 10, 180, 250))
    )
    # get chevrons using template matches
    green_chevrons = keep_inverted_chevrons(detect_chevrons(green_segment), red_segment)
    red_chevrons = keep_inverted_chevrons(detect_chevrons(red_segment), green_segment)
    return (), {
        "green_segment": green_segment,
        "green_chevrons": green_chevrons,
        "red_segment": red_segment,
        "red_chevrons": red_chevrons,
    }

def killfeed(image: ndarray) -> tuple[Kill, ...]:
    return killfeed_with_work(image)[0]

""" deprecated """

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
