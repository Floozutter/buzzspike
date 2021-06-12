from . import core
import argparse
import cv2
import itertools

def parse_args() -> tuple[str]:
    parser = argparse.ArgumentParser(
        description = "buzzspike from an image!",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("path", type = str, help = "to image file")
    args = parser.parse_args()
    return (args.path,)

def main(path: str) -> None:
    image = cv2.imread(path)
    core.handle_source(itertools.repeat(image), 0)

if __name__ == "__main__":
    main(*parse_args())
