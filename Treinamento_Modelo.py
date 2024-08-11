import numpy as np  # Adiciona essa linha para importar numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib  # Certifique-se de importar o joblib também, caso ainda não tenha feito

# Carrega os dados e os rótulos
data = np.load('landmarks.npy')
labels = np.load('labels.npy')

# Redimensiona os dados para 2D, pois o Random Forest espera uma entrada de 2D
n_samples, n_features = data.shape[0], data.shape[1] * data.shape[2]
data = data.reshape(n_samples, n_features)

# Codifica os rótulos (letras) como números
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Divide os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(data, labels_encoded, test_size=0.2, random_state=42)

# Treina o modelo Random Forest
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Avalia o modelo
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo: {accuracy:.2f}')

# Salva o modelo treinado
joblib.dump(model, 'random_forest_model.pkl')
