from bzsp import recog
import cv2
import numpy
from numpy import ndarray

def read_bimg(path: str) -> ndarray:
    image = cv2.imread(path)
    grayscaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshed = cv2.threshold(grayscaled, 1, 255, cv2.THRESH_BINARY)
    return threshed

def test_uoic() -> None:
    a_bimg = read_mask("tests/test_recog/uoic_a.png")
    b_bimg = read_mask("tests/test_recog/uoic_b.png")
    uoiced = recog.union_of_intersecting_components(a_bimg, b_bimg)
    n, _, _, _ = cv2.connectedComponentsWithStats(uoiced, 4, cv2.CV_32S)
    cv2.imshow("uoiced", uoiced)
    cv2.waitKey(0)
    assert n == 3
