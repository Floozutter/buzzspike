from . import recog
import cv2
import numpy
from numpy import ndarray
from typing import Iterator

def handle_source(source: Iterator[ndarray], delay: int) -> None:
    maskname = "uoiced"
    for frame in source:
        # process frame
        _, work = recog.killfeed_with_work(frame)
        # show work
        show = frame.copy()
        if maskname:
            uninteresting = numpy.where(work[maskname] == 0)
            show[uninteresting] //= 8
        cv2.imshow("bzst", show)
        # handle input
        keycode = cv2.waitKey(delay) & 0xFF
        if keycode == 27:
            cv2.destroyAllWindows()
            break
        elif keycode == ord("n"):
            maskname = ""
        elif keycode == ord("w"):
            maskname = "white_seg"
        elif keycode == ord("g"):
            maskname = "green_seg"
        elif keycode == ord("r"):
            maskname = "red_seg"
        elif keycode == ord("u"):
            maskname = "uoiced"
