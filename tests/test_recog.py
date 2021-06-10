from bzsp import recog
import cv2
import numpy
from numpy.typing import ArrayLike

def read_mask(path: str) -> ArrayLike:
    image = cv2.imread(path)
    grayscaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshed = cv2.threshold(grayscaled, 1, 255, cv2.THRESH_BINARY)
    return threshed

def test_uoic() -> None:
    mask_a = read_mask("tests/test_recog/uoic_a.png")
    mask_b = read_mask("tests/test_recog/uoic_b.png")
    uoiced = recog.union_of_intersecting_components(mask_a, mask_b)
    n, _, _, _ = cv2.connectedComponentsWithStats(uoic, 4, cv2.CV_32S)
    assert n == 2
