from bzsp import recog
import cv2
import numpy
from numpy import ndarray

def read_bimg(path: str) -> ndarray:
    image = cv2.imread(path)
    grayscaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshed = cv2.threshold(grayscaled, 1, 255, cv2.THRESH_BINARY)
    return threshed

class TestUoic:
    def test_ab(self) -> None:
        a_bimg = read_bimg("tests/test_recog/uoic_a.png")
        b_bimg = read_bimg("tests/test_recog/uoic_b.png")
        uoiced = recog.union_of_intersecting_components(a_bimg, b_bimg)
        n, _, _, _ = cv2.connectedComponentsWithStats(uoiced, 4)
        cv2.imshow("TestUoic.test_ab: uoiced", uoiced)
        cv2.waitKey(0)
        assert n == 3
    def test_cd(self) -> None:
        c_bimg = read_bimg("tests/test_recog/uoic_c.png")
        d_bimg = read_bimg("tests/test_recog/uoic_d.png")
        uoiced = recog.union_of_intersecting_components(c_bimg, d_bimg)
        n, _, _, _ = cv2.connectedComponentsWithStats(uoiced, 4)
        cv2.imshow("TestUoic.test_cd: uoiced", uoiced)
        cv2.waitKey(0)
        assert n == 2
