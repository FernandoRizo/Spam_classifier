# Clasificador de spam/ham

Clasificador de spam/ham para correos y SMS entrenado con TF-IDF (unigramas+bigramas) y un MLP (Keras/TensorFlow). Incluye limpieza robusta de texto (normaliza acentos, corrige leetspeak y obfuscaciones típicas), soporte multifuente (EN, ES, y datos propios), evaluación con métricas estándar y demo en vivo (Notebook).


## Características
- Vectorización: TextVectorization con output_mode="tf_idf", ngrams=2.

- Modelo: Dense(128) + Dropout + Dense(64) + Dense(1, sigmoid).

- Datasets: 
```bash
ENRON (EN): bvk/ENRON-spam, SMS (EN): ucirvine/sms_spam
```
```bash
Hugging face (ES): softecapps/spam_ham_spanish
```
```bash
 Propios (Drive)
```

- Limpieza: normaliza Unicode, URLs/EMAIL, números largos, símbolos a letras (€→e, 1→i, 3→e, …), etc.

- Umbral óptimo por F1 (configurable).

- Modelo guardado en un único archivo: mlp_spam_tf.keras (incluye la capa de vectorización).
## Requisitos

- tensorflow>=2.15
- datasets
- pandas
- numpy
- scikit-learn



## Instalar

```bash
pip install -U tensorflow datasets pandas numpy scikit-learn gradio
```

