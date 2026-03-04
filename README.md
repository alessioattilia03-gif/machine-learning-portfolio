# Portfolio Machine Learning: Regressione e Classificazione
Questo repository documenta lo sviluppo di modelli predittivi basati su reti neurali Multi-Layer Perceptron (MLP) per l'analisi di dataset tecnici.

## 📝 Consegne degli Esercizi

### Esercitazione 1: Regressione MLP
**Obiettivo:** Implementare un modello di regressione per la predizione del consumo energetico basato sul dataset `ele-2.txt`.
- **Task:** Valutare l'impatto della complessità della rete variando il numero di neuroni nello strato nascosto (range 5-50, step 5).
- **Requisiti:** - Standardizzazione delle feature tramite `StandardScaler`.
  - Divisione del dataset (70% training, 30% test).
  - Metrica di valutazione: Mean Squared Error (MSE).

### Esercitazione 2: Classificazione Multiclasse
**Obiettivo:** Sviluppare un classificatore per un dataset multiclasse sbilanciato.
- **Task:** Ottimizzare la topologia della rete per massimizzare la capacità di generalizzazione.
- **Requisiti:**
  - Implementazione della **Stratified K-Fold Cross-Validation** per gestire lo sbilanciamento delle classi.
  - Target di performance: Accuratezza superiore a 0.95.

## 📊 Risultati e Metriche
I modelli sono stati validati monitorando la convergenza della funzione di costo.

### Regressione
L'errore è stato calcolato come:
$$MSE = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

### Classificazione
Raggiunta un'accuratezza finale di **0.97** su test set, confermando l'efficacia della standardizzazione nel prevenire il Vanishing Gradient.

## 🛠️ Requisiti Tecnici
Il progetto è configurato per **Python 3.12**.
Per installare le dipendenze:
```bash
pip install -r requirements.txt
```

## 🌐 Navigazione Rapida
- **[Vai al Sito Web Portfolio](https://alessioattilia03-gif.github.io)**
- **[Scarica il mio CV](https://alessioattilia03-gif.github.io/cv_professional.pdf)**
