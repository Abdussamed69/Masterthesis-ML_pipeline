#2025-10-01 Author:             Abdussamed Korkmaz 
#                               Master Electrical Systems Engineering 
#                               University of Heilbronn, Germany
#2025-10-01 Description:        predict_app.py 
# Inference + Soft-Voting über mehrere Kanäle (Excel-Input -> Klassen + Wahrscheinlichkeiten)
# Erwartung je Kanal: Excel-Datei mit Zeilen=Messungen, Spalten=Features (keine Labels)
#********************************************************************************************************************************
#%% Import necessary libraries
from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
import joblib
from pathlib import Path
import numpy as np
import pandas as pd

#%% Funktionen definieren
def load_model_and_scaler(model_root, channel):
    """
    Lädt Modell und Scaler für einen gegebenen Kanal.
    """
    model_path = Path(model_root) / channel / "LogisticRegression_model.joblib"
    scaler_path = Path(model_root) / channel / "LogisticRegression_scaler.joblib"

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    return model, scaler

@dataclass(frozen=True)
class ChannelConfig:
    name: str
    model_path: Path
    scaler_path: Path
    input_excel: Path
    weight: float = 1.0

def load_X_from_excel(filepath):
    """
    Lädt eine Excel-Datei mit:
    - Zeilen = Messungen
    - Spalten = Features
    - KEINE Label-Spalte
    - KEINE sonstigen Spalten
    """
    df = pd.read_excel(filepath)
    X = df.values
    return X

def predict_channel_proba(X_channel, model, scaler):
    if X_channel.shape[1] != scaler.mean_.shape[0]:
        raise ValueError(
            f"Feature mismatch: expected {scaler.mean_.shape[0]}, got {X_channel.shape[1]}"
        )
    X_scaled = scaler.transform(X_channel)
    proba = model.predict_proba(X_scaled)
    return proba

def soft_voting(probabilities, weights=None):
    """
    probabilities: dict[channel] -> np.array (n_samples, n_classes)
    weights: dict[channel] -> float
    Alle Messtabellen müssen miteinander korrellieren, sonst mathematisch sinnloss
    Alle Messtabellen müssen excat diesselbe Spalten und Zeilenzahl haben!
    Numpy kann keine Matrizenmultiplikation mit unterschiedlichen Dimensionen durchführen
    """
    channels = list(probabilities.keys())

    # --- NEU: Konsistenzprüfung ---
    n_samples_set = {probabilities[ch].shape[0] for ch in channels}
    if len(n_samples_set) != 1:
        raise ValueError(
            "Soft Voting requires identical number of samples per channel. "
            f"Got sample counts: { {ch: probabilities[ch].shape[0] for ch in channels} }"
        )

    n_samples, n_classes = probabilities[channels[0]].shape

    if weights is None:
        weights = {ch: 1.0 for ch in channels}

    combined_proba = np.zeros((n_samples, n_classes))

    for ch in channels:
        combined_proba += weights[ch] * probabilities[ch]

    combined_proba /= sum(weights.values())
    return combined_proba

def final_prediction(combined_proba, class_labels):
    """
    Gibt finale Klasse + Konfidenz zurück.
    """
    class_idx = np.argmax(combined_proba, axis=1)
    confidence = np.max(combined_proba, axis=1)

    predicted_labels = [class_labels[i] for i in class_idx]
    return predicted_labels, confidence

#%% Hauptprogramm
if __name__ == "__main__":

    MODEL_ROOT = Path(r"C:\Users\korkm\Desktop\5.Sem (Master-Thesis)\Thesis\ML\models")
    INPUT_ROOT = Path(r"C:\Users\korkm\Desktop\5.Sem (Master-Thesis)\Thesis\ML\new_measurements")

    CHANNELS = ["325nm", "365nm", "275nm", "lamp_VIS"]

    # Excel-Input laden
    X_new = {
        "325nm": load_X_from_excel(INPUT_ROOT / "new_measurements_325nm.xlsx"),
        "365nm": load_X_from_excel(INPUT_ROOT / "new_measurements_365nm.xlsx"),
        "275nm": load_X_from_excel(INPUT_ROOT / "new_measurements_275nm.xlsx"),
        "lamp_VIS": load_X_from_excel(INPUT_ROOT / "new_measurements_lamp_VIS.xlsx"),
    }

    probabilities = {}

    # Referenzklassen einmal laden
    model_ref, _ = load_model_and_scaler(MODEL_ROOT, CHANNELS[0])
    class_labels = model_ref.classes_.tolist()

    for ch in CHANNELS:
        model, scaler = load_model_and_scaler(MODEL_ROOT, ch)
        probabilities[ch] = predict_channel_proba(X_new[ch], model, scaler)

    sample_counts = {ch: probabilities[ch].shape[0] for ch in probabilities}
#*******************************************************************************************************************    
    #Probabilistische Einzelmodell-Klassifikation
    for ch in CHANNELS:
            preds, conf = final_prediction(probabilities[ch], class_labels)
            df_out = pd.DataFrame({"predicted_label": preds, "confidence": conf})
            out_path = INPUT_ROOT / f"predictions_{ch}.xlsx"
            df_out.to_excel(out_path, index=False)
            print(f"Saved per-channel predictions for {ch} -> {out_path} (n_samples={sample_counts[ch]})")
#*******************************************************************************************************************