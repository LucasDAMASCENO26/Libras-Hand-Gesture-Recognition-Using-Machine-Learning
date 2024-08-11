import os
import numpy as np
import cv2
import mediapipe as mp

# Inicializa o mediapipe para detecção de mãos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

data_dir = 'data_set'  # Caminho para a pasta com as subpastas 'test' e 'training'
categories = list('ABCDEFGILMNOPQRSTUV')  # Ignora as letras H, J, K, W, X, Y, Z
data = []
labels = []

for category in categories:
    folder_path = os.path.join(data_dir, 'training', category)
    if not os.path.exists(folder_path):
        continue  # Pula se a pasta não existir
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Processa a imagem para encontrar as mãos
        results = hands.process(img_rgb)
        hands_points = results.multi_hand_landmarks

        if hands_points:
            for points in hands_points:
                landmarks = []
                for lm in points.landmark:
                    landmarks.append([lm.x, lm.y])
                data.append(landmarks)
                labels.append(category)

hands.close()

# Converte listas para numpy arrays e salva em arquivos .npy
data = np.array(data)
labels = np.array(labels)
np.save('landmarks.npy', data)
np.save('labels.npy', labels)
