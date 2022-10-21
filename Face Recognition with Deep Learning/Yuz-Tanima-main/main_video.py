import cv2
from simple_facerec import SimpleFacerec

# Yüzleri bir klasörden kodlama
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

# Kamerayı Yükle
cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()

    # Yüzleri Algıla ve kim olduğu belirle
    face_locations, face_names = sfr.detect_known_faces(frame)
    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
        # kamera ekranında bulduğu yüzün kime ait olduğunu yazar
        cv2.putText(frame, name,(x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        # kamera ekranında bulduğu yüzün etrafında bir kare çizer
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
