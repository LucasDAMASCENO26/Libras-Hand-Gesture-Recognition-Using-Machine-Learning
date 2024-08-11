import cv2
import numpy as np
import joblib
import mediapipe as mp
from sklearn.preprocessing import LabelEncoder

# Inicializa o mediapipe para detecção de mãos
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Carrega o modelo treinado
model = joblib.load('random_forest_model.pkl')

# Carrega o LabelEncoder usado no treinamento
labels = np.load('labels.npy')
label_encoder = LabelEncoder()
label_encoder.fit(labels)  # Ajusta o LabelEncoder aos rótulos originais

# Inicializa a captura de vídeo
video = cv2.VideoCapture(0)
hands = mp_hands.Hands(max_num_hands=1)

while True:
    success, img = video.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    hands_points = results.multi_hand_landmarks

    h, w, _ = img.shape

    if hands_points:
        for points in hands_points:
            # Obtém os limites da mão para desenhar o retângulo
            x_min, x_max = w, 0
            y_min, y_max = h, 0
            for lm in points.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min, x_max = min(x, x_min), max(x, x_max)
                y_min, y_max = min(y, y_min), max(y, y_max)

            # Desenha o retângulo ao redor da mão
            cv2.rectangle(img, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), (0, 255, 255), 2)

            # Desenha os pontos da mão
            mp_drawing.draw_landmarks(img, points, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))

            landmarks = []
            for lm in points.landmark:
                landmarks.append([lm.x, lm.y])

            # Converte os landmarks para a entrada do modelo
            landmarks = np.array(landmarks).flatten().reshape(1, -1)

            # Prediz a letra usando o modelo
            prediction = model.predict(landmarks)
            letter = label_encoder.inverse_transform(prediction)[0]

            # Exibe a letra prevista com uma fonte mais bonita
            cv2.putText(img, f'Letra: {letter}', (50, 100), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 0, 0), 5, cv2.LINE_AA)

    cv2.imshow("Reconhecimento de Letra", img)
    if cv2.waitKey(1) & 0xFF == 27:  # Sai do loop se a tecla 'Esc' for pressionada
        break

video.release()
cv2.destroyAllWindows()
