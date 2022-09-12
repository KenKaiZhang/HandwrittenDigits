import cv2

cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)

WIDTH = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
HEIGHT = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

while True:
    ret, frame = cap.read()
    frame_copy = frame.copy()

    bbox_size = (60,60)
    bbox = [(int(WIDTH//2 - bbox_size[0]//2), int(HEIGHT//2 - bbox_size[1]//2)),
             (int(WIDTH//2 + bbox_size[0]//2), int(HEIGHT//2 + bbox_size[1]//2))]

    img = frame[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (180,60))

    cv2.imshow('image', img)
    cv2.imshow('frame', frame_copy)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
