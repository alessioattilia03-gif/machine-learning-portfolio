# ================= LIBRERIE =================
import numpy as np
import os
import glob
import csv
import random

import warnings
warnings.filterwarnings("ignore")

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# ================= CONFIGURAZIONE =================
# Percorso relativo: punta alla cartella contenente i file CSV del dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
cartella_input = os.path.join(BASE_DIR, "..", "data", "irisSingleCl")
SEED = 42

random.seed(SEED)
np.random.seed(SEED)


# ================= SCRITTURA FILE ARFF =================
def scrivi_arff(percorso_file, nome_relazione, attributi, dati, classi_possibili):
    with open(percorso_file, 'w') as f:
        f.write(f"@RELATION {nome_relazione} \n\n")
        for att in attributi:
            f.write(f"@ATTRIBUTE {att} REAL\n")
        f.write(f"@ATTRIBUTE class {classi_possibili}\n\n")
        f.write(f"@DATA\n")
        for riga in dati:
            riga_str = ",".join(map(str, riga))
            f.write(f"{riga_str}\n")


# ================= MAIN =================
def main():
    print("--- START ---\n")

    # Ricerca di tutti i file CSV nella cartella di input
    percorso_ricerca = os.path.join(cartella_input, "*.csv")
    file_csv = glob.glob(percorso_ricerca)

    if not file_csv:
        print(f"Errore: nessun file .csv trovato in {cartella_input}")
        return

    dataset_originali = {}
    nomi_attributi = []

    print(">>> STEP 1: caricamento e creazione ARFF")

    for file_path in file_csv:
        nome_classe = os.path.basename(file_path).replace(".csv", "")
        dataset_originali[nome_classe] = []

        with open(file_path, 'r') as f:
            reader = csv.reader(f)

            primo_giro = True
            for row in reader:
                if not row:
                    continue

                # Inferisce i nomi degli attributi dalla prima riga valida
                if primo_giro and not nomi_attributi:
                    num_colonne = len(row) - 1
                    nomi_attributi = [f"att_{i}" for i in range(num_colonne)]
                    primo_giro = False

                features = [float(x) for x in row[:-1]]
                dataset_originali[nome_classe].append(features + [nome_classe])

        print(f"   - Caricato {nome_classe}: {len(dataset_originali[nome_classe])} istanze")

    dataset_binari_memoria = {}
    nomi_classi = list(dataset_originali.keys())

    # ================= ONE vs ALL =================
    for target_class in nomi_classi:
        dataset_temporaneo = []
        for classe_corrente, dati in dataset_originali.items():
            for riga in dati:
                features = riga[:-1]
                nuova_label = "positive" if classe_corrente == target_class else "negative"
                dataset_temporaneo.append(features + [nuova_label])

        random.shuffle(dataset_temporaneo)
        dataset_binari_memoria[target_class] = dataset_temporaneo

        nome_arff = os.path.join(cartella_input, f"binary_{target_class}.arff")
        scrivi_arff(nome_arff, f"OneVsAll_{target_class}", nomi_attributi, dataset_temporaneo, "{positive,negative}")

    # ================= CLASSIFICAZIONE BINARIA =================
    print("\n>>> FASE 2: Reti Neurali Binarie (con Scaling)")

    # Pipeline: standardizzazione dei dati seguita dalla rete neurale
    mlp_pipeline = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(20,), max_iter=800, random_state=SEED)
    )

    for target_class, dati in dataset_binari_memoria.items():
        X = np.array([r[:-1] for r in dati])
        y = np.array([r[-1] for r in dati])

        # StratifiedKFold preserva la proporzione delle classi in ogni fold
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        y_pred = cross_val_predict(mlp_pipeline, X, y, cv=cv)

        print(f"\n--- Modello Binario: {target_class} ---")
        print(classification_report(y, y_pred, target_names=["negative", "positive"]))

    # ================= CLASSIFICAZIONE MULTI-CLASSE =================
    print("\n>>> FASE 3: Rete Neurale Multi-Classe")

    dataset_totale = []
    for dati in dataset_originali.values():
        dataset_totale.extend(dati)
    random.shuffle(dataset_totale)

    X_multi = np.array([r[:-1] for r in dataset_totale])
    y_multi = np.array([r[-1] for r in dataset_totale])

    y_pred_multi = cross_val_predict(mlp_pipeline, X_multi, y_multi, cv=cv)

    print("\n[REPORT GLOBALE MULTI-CLASSE]")
    print(classification_report(y_multi, y_pred_multi))


if __name__ == "__main__":
    main()