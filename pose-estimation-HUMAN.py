import posecamera
import cv2
import time

detector = posecamera.pose_tracker.PoseTracker()
# image = cv2.imread("data/input.jpg")
video_stream = cv2.VideoCapture(0)

while True:
    ret, frame = video_stream.read()
    pose_output = detector(frame)
    data = pose_output.keypoints.items()

    for key, (y, x, confidence) in data:
        cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)

    cv2.imshow('video stream', frame)
    c = cv2.waitKey(1)
    if c == 27:
        break

video_stream.release()
cv2.destroyAllWindows()

# cv2.imshow("Human Pose Estimator", image)
# cv2.waitKey(0)