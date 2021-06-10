from . import recog
import mss
import cv2
import numpy

def main() -> None:
    monitor_number = 1
    with mss.mss() as sct:
        monitor = sct.monitors[monitor_number]
        region = {
            "top": monitor["top"],
            "left": monitor["left"] + monitor["width"] // 2,
            "width": monitor["width"] // 2,
            "height": monitor["height"] // 2,
            "mon": monitor_number,
        }
        while True:
            image = numpy.array(sct.grab(region))
            _, work = recog.killfeed_with_work(image)
            show = cv2.bitwise_and(image, image, mask = work["uoiced"])
            cv2.imshow("bzst", show)
            if cv2.waitKey(5) & 0xFF == 27:
                cv2.destroyAllWindows()
                return

if __name__ == "__main__":
    main()
