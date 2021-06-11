from . import recog
import cv2
import numpy
from numpy import ndarray

def handle_frame(frame: ndarray) -> None:
    _, work = recog.killfeed_with_work(frame)
    show = frame.copy()
    uninteresting = numpy.where(work["uoiced"] == 0)
    show[uninteresting] //= 8
    cv2.imshow("bzst", show)
