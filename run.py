import cv2


def is_user_wants_quit():
    return cv2.waitKey(1) & 0xFF == ord('q')


def show_frame(frame):
    cv2.imshow('Video', frame)


def draw_sqare(frame, color):
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)


def get_cascades():
    cascade_paths = {
        "Face front": 'haarcascades/haarcascade_frontalface_default.xml',
        "Face profile": 'haarcascades/haarcascade_profileface.xml',
        # "Smile": 'haarcascades/haarcascade_smile.xml',
        "Eyes": 'haarcascades/haarcascade_eye.xml',
        # "Full body": 'haarcascades/haarcascade_fullbody.xml',
        # "Cat face": 'haarcascades/haarcascade_frontalcatface_extended.xml',
    }
    cascades = [
        cv2.CascadeClassifier(cascade_path)
        for title, cascade_path in cascade_paths.items()
    ]
    return cascades


if __name__ == "__main__":
    cascades = get_cascades()
    video_capture = cv2.VideoCapture(0)
    while True:
        if not video_capture.isOpened():
            print("Couldn't find your webcam... Sorry :c")
        _, webcam_frame = video_capture.read()
        gray_frame = cv2.cvtColor(webcam_frame, cv2.COLOR_BGR2GRAY)
        captures = [cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        ) for cascade in cascades]
        for capture in captures:
            for (x, y, w, h) in capture:
                draw_sqare(webcam_frame, (255, 0, 0))
        show_frame(webcam_frame)

        if is_user_wants_quit():
            break
    video_capture.release()
    cv2.destroyAllWindows()
