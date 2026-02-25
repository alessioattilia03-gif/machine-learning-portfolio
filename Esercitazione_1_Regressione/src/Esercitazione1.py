import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# ============ CONFIGURAZIONE ============
import os

# Percorso relativo: punta alla cartella 'data' situata al livello superiore rispetto a 'src'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "..", "data", "ele-2.txt")

RANDOM_STATE = 7
NEURONS_RANGE = range(5, 51, 5)

# Parametri rete neurale
LEARNING_RATE = 0.01
TOLERANCE = 1e-4
MAX_ITERATIONS = 1000


# ============ LETTURA DATASET ============
# Caricamento del file CSV; le righe con valori NaN vengono rimosse
data = np.genfromtxt(DATASET_PATH, delimiter=",", invalid_raise=False)
data = data[~np.isnan(data).any(axis=1)]

if data.size == 0:
    raise ValueError("Il file non contiene righe valide")

# Tutte le colonne tranne l'ultima sono feature; l'ultima è il target
X_original = data[:, :-1].copy()
y = data[:, -1]

print(f"Dataset caricato: {X_original.shape[0]} campioni, {X_original.shape[1]} features")

# ============ NORMALIZZAZIONE ============
# Standardizzazione delle feature (media 0, deviazione standard 1)
scaler = StandardScaler()
X = scaler.fit_transform(X_original)

# ============ SPLIT TRAIN/TEST ============
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE
)

# ============ TRAINING MULTI-MODEL ============
# Addestramento di un modello MLP per ciascuna configurazione di neuroni
mse_train, mse_test = [], []

for n in NEURONS_RANGE:
    model = MLPRegressor(
        hidden_layer_sizes=(n,),
        activation='logistic',
        max_iter=MAX_ITERATIONS,
        random_state=RANDOM_STATE,
        learning_rate_init=LEARNING_RATE,
        tol=TOLERANCE,
        early_stopping=False,
    )
    model.fit(X_train, y_train)

    # Calcolo del MSE: media degli errori quadratici tra valori reali e predetti
    # MSE = (1/N) * Σ(y_reale - y_predetto)²
    mse_train.append(mean_squared_error(y_train, model.predict(X_train)))
    mse_test.append(mean_squared_error(y_test, model.predict(X_test)))

# ============ RISULTATI ============
print("\n" + "=" * 50)
print(f"{'Neuroni':>8} | {'MSE Train':>12} | {'MSE Test':>12}")
print("=" * 50)
for n, mse_tr, mse_te in zip(NEURONS_RANGE, mse_train, mse_test):
    print(f"{n:>8} | {mse_tr:>12.4f} | {mse_te:>12.4f}")

# ============ MODELLO OTTIMALE ============
# Selezione del modello con MSE minimo sul test set
min_idx = np.argmin(mse_test)
print("\n" + "=" * 50)
print("MODELLO OTTIMALE")
print("=" * 50)
print(f"Neuroni: {list(NEURONS_RANGE)[min_idx]}")
print(f"MSE Test: {mse_test[min_idx]:.4f}")
print(f"MSE Train: {mse_train[min_idx]:.4f}")
print(f"Gap (overfitting): {abs(mse_test[min_idx] - mse_train[min_idx]):.4f}")

# ============ GRAFICO ============
# Confronto visivo tra errore di training e test al variare dei neuroni
plt.figure(figsize=(12, 6))
plt.plot(NEURONS_RANGE, mse_train, 'o-', color='#2ecc71',
         label='Training MSE', linewidth=2, markersize=8)
plt.plot(NEURONS_RANGE, mse_test, 's--', color='#e74c3c',
         label='Test MSE', linewidth=2, markersize=8)

plt.xlabel("Numero di neuroni nello strato nascosto", fontsize=12)
plt.ylabel("Mean Squared Error (MSE)", fontsize=12)
plt.title("Validazione del modello: Training vs Test MSE", fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()