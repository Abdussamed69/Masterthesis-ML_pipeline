#2025-10-01 Author:             Abdussamed Korkmaz 
#                               Master Electrical Systems Engineering 
#                               University of Heilbronn, Germany
#2025-10-01 Description:        ML model data preprocessing script
#                               This script loads data from Excel files, removes unnecessary columns,
#                               normalizes the data using Z-score scaling, and saves the processed data back to Excel files.
#2025-12-07 Description:        It also includes functionality for training and evaluating a KNN classifier.
#                               Additionally, it provides visualizations for exploratory data analysis (EDA).
#                               The script is modular, with functions for loading data, normalizing data,
#                               and performing machine learning tasks.
#********************************************************************************************************************************
#%% Import necessary libraries
import os
import gc
import sys
# Force UTF-8 for stdout/stderr on Windows to avoid UnicodeDecodeError
# when subprocess/third-party libraries write characters outside cp1252.
os.environ.setdefault("PYTHONUTF8", "1")
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    # reconfigure may not exist on very old Python versions; ignore then
    pass
# Ensure we can check sklearn estimator types
from sklearn.base import BaseEstimator
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from pandas.plotting import scatter_matrix
import seaborn as sns
import joblib
from lazypredict.Supervised import LazyClassifier
from sklearn.ensemble import RandomForestClassifier
# Klassifikator-Metriken importieren
from sklearn.neighbors import KNeighborsClassifier
# linear SVM importieren
from sklearn.svm import LinearSVC
# Logistische Regression importieren
from sklearn.linear_model import LogisticRegression
# Metriken für Logistische Regression importieren
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, log_loss, balanced_accuracy_score  

#%% Funktionen definieren
def load_data_drop_SourceTable(input_dir: Union[str, Path]):
    """
    Load data from a Excel (.xlsx) file into a pandas DataFrame.
    Removes the "Source" column if it exists
    :param input_dir: Pfad zum Ordner mit .xlsx-Dateien
    :param output_dir: Ziel-Ordner zum Speichern der bereinigten Excel-Datei
    """
    file_paths = list(Path(input_dir).glob("*.xlsx"))
    if not file_paths:
        raise FileNotFoundError("Keine .xlsx-Dateien im angegebenen Verzeichnis gefunden.")

    for file_path in file_paths:
        try:
            df_features = pd.read_excel(file_path)
            print(f"Erfolgreich beim Lesen von {file_path.name}")
        except Exception as e:
            print(f"Fehler beim Lesen von {file_path.name}: {e}")
    # Spalte "Source" entfernen, wenn so eine Spalte existiert
    if "Source" in df_features.columns:
        df_features = df_features.drop("Source", axis=1) 
        # axis=1 für Spaltenoperationen, wenn axis=0, dann für Zeilenoperationen
        print('"Source"-Spalte wurde entfernt.')
    else:
        raise ValueError('"Source"-Spalte nicht gefunden, keine Entfernung vorgenommen.')

    return df_features

def normalize_data(input_dir: Union[str, Path], output_file_name: str = "classification_matrix_normalized.xlsx"):
    """
    Normalize the DataFrame using Z-score scaling (StandardScaler).
    :param input_dir: Pfad zum Ordner mit .xlsx-Dateien
    :param output_file_name: Name der Ausgabedatei für die normalisierten Daten
    """
    file_paths = list(Path(input_dir).glob("*.xlsx"))
    if not file_paths:
        raise FileNotFoundError("Keine .xlsx-Dateien im angegebenen Verzeichnis gefunden.")

    for file_path in file_paths:
        try:
            df = pd.read_excel(file_path)
            print(f"Erfolgreich beim Lesen von {file_path.name}")
        except Exception as e:
            print(f"Fehler beim Lesen von {file_path.name}: {e}")

    # Labelspalte extrahieren, damit nur noch numerische Daten übrig bleiben
    target = df["OilType"]
    features = df.drop("OilType", axis=1)

    # Z-Score Normalisierung der numerischen Daten
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Labelspalte wieder hinzufügen
    df_scaled = pd.DataFrame(scaled_features, columns=features.columns)
    # Erstellen eines neuen DataFrames mit den skalierten Daten
    # features.columns behält die ursprünglichen Spaltennamen bei
    df_scaled["OilType"] = target 
    # Hinzufügen der Labelspalte zurück zum DataFrame
    # target enthält die ursprünglichen Labels

    # Speichern der normalisierten DataFrame inklusive Labelspalte in einer neuen Excel-Datei
    output_file = Path(input_dir) / output_file_name
    df_scaled.to_excel(output_file, index=False, header=True)
    print(f"Z-Score-normalisierte Datei gespeichert unter: {output_file}")

def run_ml_pipeline(input_dir: Union[str, Path], clf: str = "classifier"):
    # Daten laden und "Source"-Spalte entfernen
    file_paths = list(Path(input_dir).glob("*.xlsx"))
    if not file_paths:
        raise FileNotFoundError("Keine .xlsx-Dateien im angegebenen Verzeichnis gefunden.")
    df = load_data_drop_SourceTable(input_dir)

    # Daten in Features und Targets aufteilen
    # Labelspalte "OilType" extrahieren und in y speichern, Features in X speichern
    X, y = df.drop("OilType", axis=1), df["OilType"] #axis=1 für Spaltenoperationen
    #X, y = df(return_X_y=True, drop="OilType")

    # Trainings- und Testdaten aufteilen
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # Stratify sorgt dafür, dass die Klassenverteilung in Trainings- und Testdaten gleich bleibt
    # test_size=0.2 bedeutet, dass 20% der Daten für den Test verwendet werden
    # random_state=42 sorgt für Reproduzierbarkeit der Aufteilung
    # ohne stratify könnte es passieren, dass eine Klasse im Testset fehlt, wenn die Daten ungleich verteilt sind
    # stratifyshufflessplit die Daten so, dass die Klassenverteilung in beiden Sets gleich bleibt
    # ohne random_state würde bei jedem Lauf eine andere Aufteilung erfolgen, was die Vergleichbarkeit erschwert
    # andererseits kann random_state weggelassen werden, wenn keine Reproduzierbarkeit benötigt wird,
    # was zu unterschiedlichen Ergebnissen bei verschiedenen Läufen führen kann

    # Z-Score Normalisierung durchführen mit "StandardScaler"
    scaler = StandardScaler()
    scaled_X_train = scaler.fit_transform(X_train)
    #print("scaled_X_train:", scaled_X_train)
    scaled_X_test = scaler.transform(X_test) # es wird nur transform auf Testdaten angewendet
    # fit_transform auf Trainingsdaten anwenden, um Mittelwert und Standardabweichung zu berechnen und zu transformieren
    # transform auf Testdaten anwenden, um die gleichen Mittelwert- und Standardabweichungswerte zu verwenden, 
    # da das Modell sonst "in die Zukunft schaut" --> wir haben die Testdaten nicht gesehen (dürfen wir auch nicht)
    #print("scaled_X_test",scaled_X_test)
    # Labels für Plots unabhängig vom Klassifikator verwenden
    labels = np.unique(y_train) #unique() gibt die eindeutigen Klassenlabels zurück
    #print("eindeutige labels:", labels)

    if clf == "KNN":
        # KNN-Klassifikator trainieren und evaluieren
        # 1. Klassifikator definieren
        clf = KNeighborsClassifier(n_neighbors=5)   # k=5
        # 2. Cross-Validation (nur Trainingsdaten!)
        cv = StratifiedKFold(n_splits=5)
        cv_scores = cross_val_score(clf, scaled_X_train, y_train, cv=cv)
        # 3. Nach CV: Modell auf Trainingsdaten fitten
        clf.fit(scaled_X_train, y_train)
        # 4. Vorhersage auf Testdaten (noch unberührt)
        y_pred = clf.predict(scaled_X_test)
        # 5. Evaluation des Modells
        # man kann auch , scoring='accuracy' oder scoring='precision' hinzufügen, um die Metrik zu spezifizieren
        print("CV Score (Accuracy)/Kreuzvalidierungs-Genauigkeiten:", cv_scores)
        print("mean CV Score (Accuracy)/Durchschnittliche Kreuzvalidierungs-Genauigkeit:", np.mean(cv_scores))
        print("KNN-Klassifikator Genauigkeit:", clf.score(scaled_X_test, y_test)) # Accuracy-Score/Genauigkeit des Modells auf Testdaten
        print("Konfusionsmatrix:\n", confusion_matrix(y_test, y_pred))
        print("Accuracy-Score:", accuracy_score(y_test, y_pred))
        print("Precision-Score:", precision_score(y_test, y_pred, average='weighted'))
        print("Recall-Score:", recall_score(y_test, y_pred, average='weighted'))
        print("F1-Score:", f1_score(y_test, y_pred, average='weighted'))
        print("Klassifikationsbericht:\n", classification_report(y_test, y_pred))
        cm_knn = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(20, 12))
        ax = sns.heatmap(
            cm_knn,
            annot=True,
            fmt='d',
            cmap='Greens',
            xticklabels=labels,
            yticklabels=labels,
            annot_kws={"size": 18, "weight": 'bold'}   # Zahlen in den Zellen
        )

        ax.set_title("Konfusionsmatrix - KNN", fontsize=22)
        ax.set_xlabel("Predicted Label", fontsize=18)
        ax.set_ylabel("True Label", fontsize=18)

        # Tick-Labels separat skalieren
        ax.tick_params(axis='x', labelsize=14, rotation=90)
        ax.tick_params(axis='y', labelsize=14, rotation=0)

        plt.tight_layout()
        plt.savefig(Path(input_dir) / "confusion_matrix_KNN.png", dpi=300)
        #plt.show()
        plt.close() # Schließt die Abbildung, um RAM-Speicher zu sparen

        # Einzelne Vorhersage mit dem trainierten Klassifikator
        single_instance = scaled_X_test[0].reshape(1, -1)  # Einzelne Probe für Vorhersage
        # reshape(1, -1) wandelt das Array in eine 2D-Form um, die von scikit-learn erwartet wird
        print("Einzelne Probe für Vorhersage:", single_instance)
        single_prediction = clf.predict(single_instance)
        print("Vorhersage für einzelne Probe:", single_prediction)

    elif clf == "LazyClassifier":
        # LazyPredict verwenden für schnellen Vergleich verschiedener Modelle
        # 1. LazyClassifier definieren
        clf = LazyClassifier(verbose=0, ignore_warnings=True)
        # 2. Einfacher Fit auf Trainings- und Testdaten (LazyClassifier macht KEIN CV)
        models, predictions = clf.fit(scaled_X_train, scaled_X_test, y_train, y_test)
        # 3. Ausgabe der Modellübersicht
        #print("LazyClassifier Vorhersagen:\n", predictions)
        print("LazyClassifier Modellübersicht:\n", models)

    elif clf == "RandomForest":
        # Random Forest Classifier trainieren und evaluieren
        param_grid = {
            'n_estimators': [10, 50, 100],
            'max_depth': [None],
            'min_samples_split': [2, 5, 10],}
        # Bedeutung der Hyperparameter:
        # n_estimators: Anzahl der Bäume im Wald, also wie viele Entscheidungsbäume im Ensemble trainiert werden
        # n_estimators zu klein -> Modell kann nicht genug Muster lernen --> underfitting
        # n_estimators zu groß -> Modell wird sehr komplex und rechenintensiv --> overfitting
        # n_estimators=1 bedeutet, dass nur ein einzelner Baum verwendet wird, was im Grunde genommen einem einfachen Entscheidungsbaum entspricht

        # max_depth: Maximale Tiefe jedes Baumes, also mit Tiefe meint man die Anzahl der Entscheidungen von der Wurzel bis zum Blatt
        # max_depth zu klein -> Bäume können nicht genug Muster lernen --> underfitting
        # max_depth zu groß -> Bäume werden sehr komplex und rechenintensiv --> overfitting

        # min_samples_split: Minimale Anzahl von Proben, die erforderlich sind, um einen Knoten zu teilen
        # min_samples_split zu klein -> Bäume werden sehr komplex und rechenintensiv --> overfitting
        # min_samples_split zu groß -> Bäume können nicht genug Muster lernen --> underfitting
        clf = RandomForestClassifier(random_state=42) 
        grid = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=0)
        # max_depth=None bedeutet, dass die Bäume so tief wie möglich wachsen dürfen
        # n_jobs=-1 bedeutet, dass alle verfügbaren CPU-Kerne für die Berechnung genutzt werden
        # verbose=2 gibt detaillierte Informationen über den Fortschritt der Suche aus
        # verbose kann auf 0 gesetzt werden, wenn keine Ausgabe gewünscht ist
        # verbose=8
        grid.fit(scaled_X_train, y_train)
        # Get the best estimator from the grid search and make predictions
        print("best parameters:\n", grid.best_params_)
        best_clf = grid.best_estimator_
        print("best classifyer:\n", best_clf)
        y_pred = best_clf.predict(scaled_X_test)
        print("Random Forest Klassifikator Genauigkeit:", best_clf.score(scaled_X_test, y_test))
        print("Accuracy-Score:", accuracy_score(y_test, y_pred))
        print("Precision-Score:", precision_score(y_test, y_pred, average='weighted'))
        print("Recall-Score:", recall_score(y_test, y_pred, average='weighted'))
        print("F1-Score:", f1_score(y_test, y_pred, average='weighted'))
        print("Klassifikationsbericht:\n", classification_report(y_test, y_pred))
        print("Konfusionsmatrix:\n", confusion_matrix(y_test, y_pred))
        cm_rf = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(20, 12))
        ax = sns.heatmap(
            cm_rf,
            annot=True,
            fmt='d',
            cmap='Oranges',
            xticklabels=labels,
            yticklabels=labels,
            annot_kws={"size": 18, "weight": 'bold'}   # Zahlen in den Zellen
        )

        ax.set_title("Konfusionsmatrix - RandomForest", fontsize=22)
        ax.set_xlabel("Predicted Label", fontsize=18)
        ax.set_ylabel("True Label", fontsize=18)

        # Tick-Labels separat skalieren
        ax.tick_params(axis='x', labelsize=14, rotation=90)
        ax.tick_params(axis='y', labelsize=14, rotation=0)

        plt.tight_layout()
        plt.savefig(Path(input_dir) / "confusion_matrix_RandomForest.png", dpi=300)
        #plt.show()
        plt.close() # Schließt die Abbildung, um RAM-Speicher zu sparen

        # Ensure caller receives the trained estimator
        clf = best_clf

    elif clf == "LogisticRegression":
        """
            Logistische Regression mit L2-Regularisierung.
            Begründung:
            - Spektraldaten sind hochdimensional und stark korreliert (benachbarte Wellenlängen).
            - Eine L1-Regularisierung wurde bewusst nicht eingesetzt, da diese bei stark
            korrelierten spektralen Merkmalen zu instabiler Feature-Selektion führen kann.
            - L2-Regularisierung gewährleistet stabile Koeffizienten und eine physikalisch
            interpretierbare Gewichtung der spektralen Merkmale.
            - Logistic Regression liefert zusätzlich Klassenwahrscheinlichkeiten, die
            für eine spätere Entscheidungsfusion (High-Level Data Fusion) genutzt werden können.
        """
        print("\nTrainiere Logistic Regression mit GridSearchCV ...\n")
        # Lade Wellenlängenachse für spätere Plots der spektralen Relevanz über Koeffizienten
        wavelengths = np.load(Path(input_dir) / "VIS_wavelengths.npy")
        mask = (wavelengths >= 440) & (wavelengths <= 800) # Bereich 440-800 nm
        wavelengths = wavelengths[mask]

        """„Aufgrund der klaren linearen Trennbarkeit wurde der Hyperparameterraum bewusst eingeschränkt, 
            um Overfitting und unnötige Rechenlast zu vermeiden.“"""
        param_grid = {
            "C": [0.1, 1, 10],
            "penalty": ["l2"],          # bewusst nur L2
            "solver": ["lbfgs"],        # stabil für multiklassige Probleme
            "class_weight": [None, "balanced"]
        }
        base_clf = LogisticRegression(
            max_iter=2000,
            multi_class="auto",
            random_state=42
        )

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        grid = GridSearchCV(
            estimator=base_clf,
            param_grid=param_grid,
            scoring="accuracy",
            cv=cv,
            n_jobs=-1
        )

        grid.fit(scaled_X_train, y_train)
        best_clf = grid.best_estimator_
        print("best classifyer:\n", best_clf)

        y_pred = best_clf.predict(scaled_X_test)
        print("Beste Parameter (Logistic Regression):", grid.best_params_)
        print("CV-Score (mean):", grid.best_score_)
        print("Test-Accuracy:", accuracy_score(y_test, y_pred))
        print("Klassifikationsbericht:\n", classification_report(y_test, y_pred))
        print("Konfusionsmatrix:\n", confusion_matrix(y_test, y_pred))
        cm_lr = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(20, 12))
        ax = sns.heatmap(
            cm_lr,
            annot=True,
            fmt='d',
            cmap='Purples',
            xticklabels=labels,
            yticklabels=labels,
            annot_kws={"size": 18, "weight": 'bold'}   # Zahlen in den Zellen
        )

        ax.set_title("Konfusionsmatrix - Logistische Regression", fontsize=22)
        ax.set_xlabel("Predicted Label", fontsize=18)
        ax.set_ylabel("True Label", fontsize=18)

        # Tick-Labels separat skalieren
        ax.tick_params(axis='x', labelsize=14, rotation=90)
        ax.tick_params(axis='y', labelsize=14, rotation=0)

        plt.tight_layout()
        plt.savefig(Path(input_dir) / "confusion_matrix_LogisticRegression.png", dpi=300)
        #plt.show()
        plt.close() # Schließt die Abbildung, um RAM-Speicher zu sparen

        # Spektrale Relevanz für jede Klasse plotten und speichern
        plt.figure(figsize=(12, 6))
        for class_name in best_clf.classes_:
            plot_logreg_spectral_importance(
                clf=best_clf,
                wavelengths=wavelengths,
                class_name=class_name,
                channel_name=Path(input_dir).name,
                save_path=Path(input_dir) / f"spectral_importance_LogReg_{class_name}.png"
            )
        # Ensure caller receives the trained estimator
        clf = best_clf
    
    elif clf == "LinearSVC":
        """
        Linear Support Vector Classifier als linearer Margin-basierter Klassifikator.

        Begründung:
        - LinearSVC eignet sich besonders für hochdimensionale Merkmalsräume
        mit klarer (annähernd) linearer Trennbarkeit.
        - Liefert robuste Entscheidungsgrenzen, jedoch keine Klassenwahrscheinlichkeiten.
        - Wird daher primär als Vergleichsmodell zur Logistic Regression eingesetzt.
        """

        """„Aufgrund der klaren linearen Trennbarkeit wurde der Hyperparameterraum bewusst eingeschränkt, 
            um Overfitting und unnötige Rechenlast zu vermeiden.“"""
        param_grid = {
            "C": [0.1, 1, 10], # Regularisierungsparameter
            # c zu klein -> starke Regularisierung -> einfaches Modell --> underfitting
            # c zu groß -> schwache Regularisierung -> komplexes Modell --> overfitting
            "class_weight": [None, "balanced"]
        }
        base_clf = LinearSVC(
            max_iter=5000,
            dual=False,       # effizienter bei n_samples > n_features
            random_state=42
        )

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        grid = GridSearchCV(
            estimator=base_clf,
            param_grid=param_grid,
            scoring="accuracy",
            cv=cv,
            n_jobs=-1
        )

        grid.fit(scaled_X_train, y_train)
        clf = grid.best_estimator_

        y_pred = clf.predict(scaled_X_test)
        print("Beste Parameter (LinearSVC):", grid.best_params_)
        print("CV-Score (mean):", grid.best_score_)
        print("Test-Accuracy:", accuracy_score(y_test, y_pred))
        print("Klassifikationsbericht:\n", classification_report(y_test, y_pred))
        print("Konfusionsmatrix:\n", confusion_matrix(y_test, y_pred))
        # balanced accuracy score berücksichtigt ungleiche Klassenverteilungen besser 
        cm_linsvc = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(20, 12))
        ax = sns.heatmap(
            cm_linsvc,
            annot=True,
            fmt='d',
            cmap='Reds',
            xticklabels=labels,
            yticklabels=labels,
            annot_kws={"size": 18, "weight": 'bold'}   # Zahlen in den Zellen
        )

        ax.set_title("Konfusionsmatrix - LinearSVC", fontsize=22)
        ax.set_xlabel("Predicted Label", fontsize=18)
        ax.set_ylabel("True Label", fontsize=18)

        # Tick-Labels separat skalieren
        ax.tick_params(axis='x', labelsize=14, rotation=90)
        ax.tick_params(axis='y', labelsize=14, rotation=0)

        plt.tight_layout()
        plt.savefig(Path(input_dir) / "confusion_matrix_LinearSVC.png", dpi=300)
        #plt.show()
        plt.close() # Schließt die Abbildung, um RAM-Speicher zu sparen

    """ Klassenverteilung visualisieren    
    2025-12-07: Hinzugefügt zur Überprüfung der Aufteilung der Klassenverteilung
    # Trainingsset
    # Klassenverteilung im Trainings- und Testset visualisieren
    plt.figure(figsize=(20, 12))
    x = y_train.value_counts().index
    y = y_train.value_counts().values
    plt.bar(x, y)
    plt.title("Klassenverteilung im Trainingsset")
    plt.xlabel("OilType")
    plt.ylabel("Anzahl der Proben")
    plt.xticks(rotation=90)  # <-- X-Achsen-Labels drehen für bessere Lesbarkeit
    plt.tight_layout()       # wichtig gegen Abschneiden
    plt.savefig(Path(input_dir) / "class_distribution_train.png")
    plt.close() # RAM-Speicher sparen

    # Testset
    plt.figure(figsize=(20, 12))
    x = y_test.value_counts().index
    y = y_test.value_counts().values
    plt.bar(x, y)
    plt.title("Klassenverteilung im Testset")
    plt.xlabel("OilType")
    plt.ylabel("Anzahl der Proben")
    plt.xticks(rotation=90)  # <-- X-Achsen-Labels drehen für bessere Lesbarkeit
    plt.tight_layout() # wichtig gegen Abschneiden
    plt.savefig(Path(input_dir) / "class_distribution_test.png")
    plt.close() # RAM-Speicher sparen
    """

    """ andere Metriken und Visualisierungen
    ACHTUNG: Sehr Rechenintensiv bei großen Datensätzen
    # Metriken und Visualisierungen zur Überprüfung der Normalisierung und Aufteilung
    # Überprüfung der Aufteilung der Klassenverteilung mit barcharts

    # --> Vorsicht: Diese Visualisierungen sind nur für kleine Datensätze geeignet, 
    # da sehr Rechenintensiv bei großen Datensätzen --> RAM kann abstürzen
    # Mindestens 32GB RAM empfohlen bei großen Datensätzen
    # Explorative Datenanalyse (EDA) - Optional
    # Verteilung der Merkmale anzeigen
    df.hist(figsize=(20,12))  # Optional: Histogramm der normalisierten Daten anzeigen
    plt.show()
    plt.tight_layout()

    df.describe()  # Optional: Statistische Zusammenfassung der normalisierten Daten anzeigen
    print(df.describe())

    df.info()  # Optional: Informationen über den DataFrame anzeigen
    print(df.info())

    # Korrelationsmatrix anzeigen
    #df_float = df.select_dtypes(include=[np.number])  # Nur numerische Spalten auswählen
    df_float = df.drop("OilType", axis=1)  # "OilType"-Spalte entfernen für Korrelationsmatrix
    corr_matrix = df_float.corr()
    print(corr_matrix)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Korrelationsmatrix der normalisierten Daten")  
    plt.show()

    # Scatter-Matrix anzeigen
    scatter_matrix(df, alpha=0.2, figsize=(12, 12), diagonal='kde')
    plt.suptitle("Scatter-Matrix der normalisierten Daten")   
    plt.show()

    # Paarweise Beziehungen anzeigen
    sns.pairplot(df, hue="OilType")   
    plt.suptitle("Paarweise Beziehungen der normalisierten Daten")
    plt.show()

    # Boxplot anzeigen
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df.drop("OilType", axis=1))
    plt.title("Boxplot der normalisierten Daten")
    plt.show()
    """

    return clf, scaler

def save_model(model, scaler, channel_name, model_name):
    # Modell und Scaler speichern
    # Struktur des Modell-Ordners wird so aufgebaut:
    """
    models/
    │
    ├── 325nm/
    │   ├── LogisticRegression_model.joblib
    │   ├── LogisticRegression_scaler.joblib
    │
    ├── 365nm/
    │   ├── LogisticRegression_model.joblib
    │   ├── LogisticRegression_scaler.joblib
    │
    ├── 275nm/
    │   ├── LogisticRegression_model.joblib
    │   ├── LogisticRegression_scaler.joblib
    │
    ├── lamp_VIS/
    │   ├── LogisticRegression_model.joblib
    │   ├── LogisticRegression_scaler.joblib
    """
    model_dir = Path("models") / channel_name
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"{model_name}_model.joblib"
    scaler_path = model_dir / f"{model_name}_scaler.joblib"

    # Validate that `model` is a scikit-learn estimator (not a plain string)
    # Accept any object that is an instance of sklearn.base.BaseEstimator
    if isinstance(model, str):
        raise TypeError(f"Expected scikit-learn estimator for 'model', got str: {model!r}")

    if not isinstance(model, BaseEstimator):
        # Fallback: require at least a `predict` method for basic estimator behavior
        if not hasattr(model, "predict"):
            raise TypeError("Provided 'model' does not appear to be a scikit-learn estimator (missing 'predict').")

    # Validate scaler-like object has a transform method
    if scaler is not None and not hasattr(scaler, "transform"):
        raise TypeError("Provided 'scaler' does not appear to be a scikit-learn scaler (missing 'transform').")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"Modell gespeichert unter: {model_path}")
    print(f"Scaler gespeichert unter: {scaler_path}")


def plot_logreg_spectral_importance(clf, wavelengths, class_name, channel_name=None, save_path=None):
    """
    Plottet die absolute spektrale Relevanz (|Koeffizienten|)
    der logistischen Regression für eine gegebene Klasse.

    Parameters
    ----------
    clf : trained LogisticRegression
        Trainiertes LogReg-Modell
    wavelengths : np.ndarray
        Wellenlängenachse (nm)
    class_name : str
        Zielklasse (muss in clf.classes_ enthalten sein)
    channel_name : str, optional
        Name des Kanals (z.B. '325nm')
    save_path : Path or str, optional
        Speicherpfad für die Abbildung
    """

    if class_name not in clf.classes_:
        raise ValueError(f"Klasse '{class_name}' nicht im Modell gefunden.")

    class_idx = list(clf.classes_).index(class_name)
    coef = clf.coef_[class_idx]

    #plt.figure(figsize=(12, 6))
    plt.plot(wavelengths, np.abs(coef), linewidth=2)

    title = f"Spektrale Relevanz (LogReg) – Klasse: {class_name}"
    if channel_name:
        title += f" | Kanal: {channel_name}"

    plt.title(title)
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("|Logistic Regression Coefficient|")
    plt.grid(True)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        #plt.close()
    else:
        plt.show()


#%% Hauptprogramm
if __name__ == "__main__":
    # Channel-Schleife im Hauptprogramm + Memory Cleanup
    # Definition der Kanäle und ihrer Pfade
    channels = {
    "325nm": Path(r"C:\Users\korkm\Desktop\5.Sem (Master-Thesis)\Thesis\ML\325nm"),
    "365nm": Path(r"C:\Users\korkm\Desktop\5.Sem (Master-Thesis)\Thesis\ML\365nm"),
    "275nm": Path(r"C:\Users\korkm\Desktop\5.Sem (Master-Thesis)\Thesis\ML\275nm"),
    "lamp_VIS": Path(r"C:\Users\korkm\Desktop\5.Sem (Master-Thesis)\Thesis\ML\lamp\VIS"),
    }
    # ML-Pipeline für jeden Kanal und Klassifikator ausführen
    # ACHTUNG: Sehr rechenintensiv!!!
    # ACHTUNG: LazyClassifier ist viel zu RAM-intensiv, nutze es nicht in der Schleife
    #classifiers = ["LogisticRegression", "LinearSVC", "RandomForest"]
    classifiers = ["LogisticRegression", "LinearSVC", "RandomForest"]
    for clf_name in classifiers:
        for channel_name, channel_path in channels.items():
            print(f"\nTraining {clf_name} für Channel {channel_name}\n")
            clf, scaler = run_ml_pipeline(
                input_dir=channel_path,
                clf=clf_name
            )
            
            # Modell + Scaler speichern
            save_model(
                model=clf,
                scaler=scaler,
                channel_name=channel_name,
                model_name=clf_name
            )
            # RAM freigeben
            del clf, scaler
            gc.collect()
            """
            Nach jedem Trainings- und Speichervorgang werden Modellobjekte explizit
            gelöscht und der Garbage Collector aufgerufen, um den Speicherverbrauch
            bei iterativen Modelltrainings über mehrere Kanäle zu minimieren.
            """

#*******************************************************************************************************************
# Modell und zugehöriger StandardScaler werden gespeichert,
# um eine konsistente Vorverarbeitung und reproduzierbare
# Klassifikation neuer Messdaten ohne erneutes Training zu ermöglichen.
#*******************************************************************************************************************
# Data Fusion Applikation erfolgt in einem separaten Skript
#*******************************************************************************************************************