from . import recog
import cv2
import numpy
from numpy import ndarray
from typing import Iterable

def handle_source(source: Iterable[ndarray], delay: int) -> None:
    maskname = ""
    show_green_chevrons = True
    show_green_boxes = True
    show_red_chevrons = True
    show_red_boxes = True
    for frame in source:
        # process frame
        _, work = recog.killfeed_with_work(frame)
        # show work
        demo = frame.copy()
        if maskname:
            demo[numpy.where(work[maskname] == 0)] //= 8
        if show_green_chevrons:
            for x, y, w, h in work["green_chevrons"]:
                cv2.rectangle(demo, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if show_green_boxes:
            for x, y, w, h in work["green_boxes"]:
                cv2.rectangle(demo, (x, y), (x + w, y + h), (0, 122, 0), 2)
        if show_red_chevrons:
            for x, y, w, h in work["red_chevrons"]:
                cv2.rectangle(demo, (x, y), (x + w, y + h), (0, 0, 255), 2)
        if show_red_boxes:
            for x, y, w, h in work["red_boxes"]:
                cv2.rectangle(demo, (x, y), (x + w, y + h), (0, 0, 122), 2)
        cv2.imshow("bzst", demo)
        # handle input
        keycode = cv2.waitKey(delay) & 0xFF
        if keycode == 27:
            cv2.destroyAllWindows()
            break
        elif keycode == ord("n"):
            maskname = ""
        elif keycode == ord("g"):
            maskname = "green_segment"
        elif keycode == ord("h"):
            show_green_chevrons = not show_green_chevrons
        elif keycode == ord("j"):
            show_green_boxes = not show_green_boxes
        elif keycode == ord("r"):
            maskname = "red_segment"
        elif keycode == ord("t"):
            show_red_chevrons = not show_red_chevrons
        elif keycode == ord("y"):
            show_red_boxes = not show_red_boxes