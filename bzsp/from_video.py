from .core import handle_frame
import argparse
import cv2

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
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 // fps)
    print(delay)
    while cap.isOpened():
        got, frame = cap.read()
        handle_frame(frame)
        if (not got) or (cv2.waitKey(delay) & 0xFF == 27):
            cv2.destroyAllWindows()
            return

if __name__ == "__main__":
    main(*parse_args())
