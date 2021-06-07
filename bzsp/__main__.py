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
            img = numpy.array(sct.grab(region))
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, (70, 0, 0), (90, 255, 255))
            result = cv2.bitwise_and(img, img, mask = mask)
            cv2.imshow("bzst", result)
            if cv2.waitKey(5) & 0xFF == 27:
                cv2.destroyAllWindows()
                return

if __name__ == "__main__":
    main()
