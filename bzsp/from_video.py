from . import core
import argparse
import cv2
from numpy import ndarray
from typing import Iterator

def parse_args() -> tuple[str]:
    parser = argparse.ArgumentParser(
        description = "buzzspike from a video!",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("path", type = str, help = "to video file")
    args = parser.parse_args()
    return (args.path,)

def main(path: str) -> None:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("error: couldn't open video capture!")
        return
    delay = int(1000 // cap.get(cv2.CAP_PROP_FPS))
    def it() -> Iterator[ndarray]:
        while cap.isOpened():
            got, frame = cap.read()
            if not got:
                break
            yield frame
    core.handle_source(it(), delay)

if __name__ == "__main__":
    main(*parse_args())
