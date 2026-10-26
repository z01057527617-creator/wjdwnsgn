
import cv2
import os
import urllib.request

# 얼굴 검출기 XML 파일 준비
CASCADE_FILE = "haarcascade_frontalface_default.xml"
CASCADE_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/4.x/"
    "data/haarcascades/haarcascade_frontalface_default.xml"
)

if not os.path.exists(CASCADE_FILE):
    print("얼굴 검출기 파일을 내려받는 중...")
    urllib.request.urlretrieve(CASCADE_URL, CASCADE_FILE)

face_cascade = cv2.CascadeClassifier(CASCADE_FILE)

if face_cascade.empty():
    print("얼굴 검출기를 불러올 수 없습니다.")
    raise SystemExit

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    raise SystemExit

WIN = "Sheriff Robot View"
WARN_RATIO = 0.35

while True:
    success, img = cap.read()

    if not success:
        break

    img = cv2.flip(img, 1)
    H, W = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(60, 60)
    )

    # 왼쪽 위 좌표 표시 기본값
    coord_text = [
        "X: ---",
        "Y: ---",
        "Z: NO TARGET"
    ]

    for (x, y, w, h) in faces:
        cx = x + w // 2
        cy = y + h // 2

        # X, Y: 화면 중심을 기준으로 정규화한 위치
        X = (cx - W / 2) / (W / 2)
        Y = (cy - H / 2) / (H / 2)

        # Z: 얼굴 크기를 이용한 상대 거리 추정값
        # 값이 클수록 멀리 있는 것으로 추정
        Z = 100 * 200 / w

        # 가까이 접근했는지 판단
        if w / W > WARN_RATIO:
            color = (0, 0, 255)
            target_text = "WARNING! TOO CLOSE!"
        else:
            color = (0, 255, 0)
            target_text = "TARGET LOCKED"

        # 얼굴 테두리
        cv2.rectangle(
            img, (x, y), (x + w, y + h), color, 3
        )

        # 십자 조준선
        cv2.line(
            img, (cx - 20, cy), (cx + 20, cy), color, 2
        )
        cv2.line(
            img, (cx, cy - 20), (cx, cy + 20), color, 2
        )

        # 얼굴 위 또는 아래에 상태 표시
        ty = y - 10 if y > 30 else y + h + 25
        cv2.putText(
            img, target_text, (x, ty),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
        )

        # 좌표 표시 내용 갱신
        coord_text = [
            f"X: {X:+.2f}",
            f"Y: {Y:+.2f}",
            f"Z: {Z:.1f} (relative)"
        ]

        # 얼굴 하나만 추적
        break

    # 왼쪽 위에 검은색 정보창 표시
    cv2.rectangle(img, (10, 10), (300, 125), (0, 0, 0), -1)

    cv2.putText(
        img, "ROBOT COORDINATES", (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
    )

    for i, line in enumerate(coord_text):
        cv2.putText(
            img, line, (20, 62 + i * 23),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0, 255, 0), 2
        )

    cv2.imshow(WIN, img)

    key = cv2.waitKey(1) & 0xFF

    if key in (ord('q'), 27):
        break

    if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
