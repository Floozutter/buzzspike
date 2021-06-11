from .core import handle_frame
import argparse
import mss
import cv2
import numpy

def parse_args() -> tuple[int, int]:
    parser = argparse.ArgumentParser(
        description = "buzzspike from a screen!",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--screen", type = int, default = 1, help = "monitor number")
    parser.add_argument("--delay", type = int, default = 33, help = "in milliseconds")
    args = parser.parse_args()
    return args.screen, args.delay

def main(screen: int, delay: int) -> None:
    with mss.mss() as sct:
        monitor = sct.monitors[screen]
        region = {
            "top": monitor["top"],
            "left": monitor["left"] + monitor["width"] // 2,
            "width": monitor["width"] // 2,
            "height": monitor["height"] // 2,
            "mon": screen,
        }
        while True:
            handle_frame(numpy.array(sct.grab(region)))
            if cv2.waitKey(delay) & 0xFF == 27:
                cv2.destroyAllWindows()
                return

if __name__ == "__main__":
    main(*parse_args())
