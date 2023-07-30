from keras.models import load_model
import cv2
import numpy as np
from random import choice
import pyttsx3

REV_CLASS_MAP = {
    0: "yes",
    1: "no",
    2: "alldone",
    3: "please"
}


def mapper(val):
    return REV_CLASS_MAP[val]

engine = pyttsx3.init()
model = load_model("rock-paper-scissors-model.h5")

cap = cv2.VideoCapture(0)

prev_move = None

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # rectangle for user to play
    cv2.rectangle(frame, (100, 100), (500, 500), (255, 255, 255), 2)
    # rectangle for computer to play
    cv2.rectangle(frame, (800, 100), (1200, 500), (255, 255, 255), 2)

    # extract the region of image within the user rectangle
    roi = frame[100:500, 100:500]
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (300, 300))

    # predict the move made
    pred = model.predict(np.array([img]))
    move_code = np.argmax(pred[0])
    user_move_name = mapper(move_code)

    print(user_move_name)


    if cv2.waitKey(1) & 0xFF == ord(' '):
        engine.say(user_move_name)
        engine.runAndWait()


    cv2.imshow("Rock Paper Scissors", frame)

    k = cv2.waitKey(10)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()