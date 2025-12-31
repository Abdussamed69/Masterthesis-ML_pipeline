#2025-08-17 Author:             Abdussamed Korkmaz 
#                               Master Electrical Systems Engineering 
#                               University of Heilbronn, Germany
#2025-08-17 Description:        Preprocessing script for fluorescence- & reflection-data 
#*******************************************************************************************************************
# Diesses Skript liest alle Excel-Dateien ein und plottet diese, 
# **auschließlich basierend** auf die vom Skript "Fluorem_v2.py" erstellten Excel-Dateien.
# Dabei wird eine Feature-Matrix erstellt, die in einem separaten Skript für die ML-Modellierung verwendet werden kann.
#*******************************************************************************************************************
# Wichtig: Es wird davon ausgegangen, dass die Excel-Dateien eine bestimmte Struktur haben:
# 20250912-194600_P.224-990056_LIDL_extra virgin olive oil.xlsx
#        ↑           ↑          ↑            ↑
#    Timestamp | FLUOREM-Nr. | OilSource | OilType
#*******************************************************************************************************************

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from scipy.signal import savgol_filter

def preprocessing_fluor(input_dir: Union[str, Path], output_dir: Union[str, Path], sheet_name: str):
    """Preprocessing der Fluoreszenz-Daten.
    Liest alle Excel-Dateien im angegebenen Verzeichnis ein"""
    
    print("Starte Preprocessing_fluor...", flush=True)

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)  
              
    # Alle Excel-Dateien lesen
    file_paths = list(Path(input_dir).glob("*.xlsx"))
    if not file_paths:
        raise FileNotFoundError("Keine .xlsx-Dateien im angegebenen Verzeichnis gefunden.")

    # Messwerte lesen, sowie Labels und Ölmarke/-hersteller aus Dateinamen extrahieren
    observations = []
    labels = []
    sources = []
    for file_path in file_paths:
        try:
            # Messwerte lesen
            data = pd.read_excel(file_path, sheet_name=sheet_name)
            observations.append(data)
            # Labels und Ölmarke/-hersteller aus Dateinamen extrahieren (alles nach dem letzten "_")
            name = file_path.stem  # Dateiname ohne .xlsx
            parts = name.split("_")
            if len(parts) >= 2:
                labels.append(parts[-1])  # letzter Teil = Label --> Klassifizierung
                sources.append(parts[-2])  # vorletzter Teil = Ölmarke/-hersteller
            else:
                labels.append("Unknown")
                sources.append("Unknown")
        except Exception as e:
            print(f"Fehler beim Lesen von {file_path.name}: {e}")

    # Wellenlängen extrahieren
    wavelengths = observations[0].iloc[:, 1].values  # Spalte 2 = Wellenlänge
    # Wavelengths separat speichern (für späteren Plot)
    np.save(output_dir / "wavelengths.npy", wavelengths)

    # Feature-Matrix initialisieren (NaN), maximale Anzahl an Zeilen = maxLength
    #max_length = max(len(df) for df in observations)
    max_length = len(observations[0])  # Annahme: alle Dateien haben gleiche Länge
    feature_matrix = np.full(                       
        (len(observations), max_length),  # Form: (rows: Anzahl_Spektren, columns: max_Anzahl_Wellenlängen)
        np.nan                            # Initialwert: NaN
    )
    # Preprocessing für jeden Datensatz
    for i, df in enumerate(observations):
        s_values = df.iloc[:, 2].values     # Spalte 3 = Intensität        
        if np.any(pd.isna(s_values)):       # NaN auffüllen (interpolieren)
            s_values = pd.Series(s_values).interpolate().bfill().ffill().values

        # In feature_matrix einfügen    
        feature_matrix[i, :len(s_values)] = s_values
        # Wellenlängenbereich reduzieren für Training und Plot (VIS: 440-800nm, NIR: 1050-1850nm)
        if sheet_name == 'NIR':
            mask = (wavelengths >= 1050) & (wavelengths <= 1850)
        elif sheet_name == 'VIS':   
            mask = (wavelengths >= 440) & (wavelengths <= 800)
        else:
            mask = np.ones_like(wavelengths, dtype=bool)  # kein Maskieren, alle Werte behalten
        # Mask anwenden
        feature_matrix[i, ~mask] = np.nan  # Werte außerhalb des Bereichs auf NaN setzen 


    # Tabelle + Labels erstellen
    df_features = pd.DataFrame(feature_matrix)
    df_features["OilType"] = pd.Categorical(labels) # Spalte "Oiltype" einfügen, als kategorische Variable
    df_features["Source"] = pd.Categorical(sources) # Spalte "Source" einfügen, als kategorische Variable

    # Speichern als Excel-Datei mit Dateinamen basierend auf Anregungsquelle
    try:
        # Nur einmal aus der ersten Datei den Wert holen
        info_data = pd.read_excel(file_paths[0], sheet_name='Info')
        Anregungs_Wellenlaenge = str(info_data.iloc[2, 2])
    except Exception as e:
        print(f"Fehler beim Auslesen der Wellenlänge aus {file_paths[0].name}: {e}")
        Anregungs_Wellenlaenge = "unknown"

    matrix_title = f"{sheet_name}-Fluoreszenz-Emissionsspektren Anregung@{Anregungs_Wellenlaenge}.xlsx"
    output_file = output_dir / matrix_title
    df_features.to_excel(output_file, index=False)
    print(f"Preprocessing_fluor abgeschlossen. Datei gespeichert unter: {output_file}")
    
    return matrix_title, df_features

def preprocessing_reflectance(input_dir: Union[str, Path], output_dir: Union[str, Path], sheet_name: str, apply_smoothing: bool = True):
    """Preprocessing der Reflektanz-Daten.
    Liest alle Excel-Dateien im angegebenen Verzeichnis ein, berechnet die Reflektanz (falls Referenz vorhanden),
    erstellt eine Feature-Matrix und speichert diese als Excel-Datei."""
    
    print("Starte Preprocessing_reflectance...", flush=True)
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    # Ordner erstellen, falls nicht vorhanden
    output_dir.mkdir(parents=True, exist_ok=True)
    
    #Referenzdatei zur Berechnung der Reflektanz finden
    # Referenzdatei (optional)
    ref_files = sorted(input_dir.glob("*_reference.xlsx"))
    ref_values = None
    if len(ref_files) == 0:
        print("Keine Referenzdatei mit '_reference.xlsx' gefunden. Reflektion wird ohne Referenz berechnet.", flush=True)
    elif len(ref_files) > 1:
        raise ValueError("Mehrere Referenzdateien gefunden. Bitte nur eine im Ordner lassen.")
    else:
        ref_df = pd.read_excel(ref_files[0], sheet_name=sheet_name)
        if ref_df.shape[1] < 3:
            raise ValueError(f"Referenzsheet hat weniger als 3 Spalten: {ref_files[0].name}")
        # Referenz-Werte (digital counts) extrahieren
        ref_values = ref_df.iloc[:, 2].values # 3.Spalte in Excel = digital Counts
        print(f"Referenzdatei verwendet: {ref_files[0].name}", flush=True)
        # Referenzwerte separat speichern (für späteren Plot)
        np.save(output_dir / "ref_values.npy", ref_values)

    # Alle Excel-Dateien lesen
    file_paths = list(Path(input_dir).glob("*.xlsx"))
    if not file_paths:
        raise FileNotFoundError("Keine .xlsx-Dateien im angegebenen Verzeichnis gefunden.")
    # exclude reference file(s) from data file list
    if len(ref_files) == 1:
        data_files = [p for p in file_paths if p != ref_files[0]]
    else:
        data_files = file_paths.copy()
    if not data_files:
        raise ValueError("Keine Messdateien gefunden.")
    
    # Messwerte lesen, sowie Labels und Ölmarke/-hersteller aus Dateinamen extrahieren (alles nach dem letzten "_")
    observations = []
    labels = []
    sources = []
    for file_path in data_files:
        try:
            # Messwerte lesen
            data = pd.read_excel(file_path, sheet_name=sheet_name)
            observations.append(data)
            # Labels und Ölmarke/-hersteller aus Dateinamen extrahieren (alles nach dem letzten "_")
            name = file_path.stem  # Dateiname ohne .xlsx
            parts = name.split("_")
            if len(parts) >= 2:
                labels.append(parts[-1])  # letzter Teil = Label
                sources.append(parts[-2])  # vorletzter Teil = Ölmarke/-hersteller
            else:
                labels.append("Unknown")
                sources.append("Unknown")
        except Exception as e:
            print(f"Fehler beim Lesen von {file_path.name}: {e}")

    # Wellenlängen extrahieren
    wavelengths = observations[0].iloc[:, 1].values  # Spalte 2 = Wellenlänge
    # Wavelengths separat speichern (für späteren Plot)
    np.save(output_dir / "wavelengths.npy", wavelengths)

    # Feature-Matrix initialisieren (NaN), maximale Anzahl an Zeilen = maxLength
    #max_length = max(len(df) for df in observations)
    max_length = len(observations[0])  # Annahme: alle Dateien haben gleiche Länge
    feature_matrix = np.full((len(observations), max_length), np.nan)

    # Preprocessing für jeden Datensatz
    for i, df in enumerate(observations):
        s_values = df.iloc[:, 2].values     # Spalte 3 = Intensität        
        if np.any(pd.isna(s_values)):       # NaN auffüllen (interpolieren)
            s_values = pd.Series(s_values).interpolate().bfill().ffill().values

        if len(ref_files) > 0:              # Nur wenn Referenzdatei existiert und passt 
            if len(s_values) != len(ref_values):    
                raise ValueError(f"Unterschiedliche Länge in Datei: {data_files[i].name}")       
            # *****Reflektanz berechnen: R(λ) = S(λ) / Ref(λ)*****
            # Schutz gegen zu kleine Referenzwerte
            min_ref = 10  # damit keine zu kleinen Werte in der Referenz sind --> Robuste Kurve
            ref_values_safe = np.where(ref_values < min_ref, min_ref, ref_values)
            r_values = s_values / ref_values_safe
            # Glättung der Reflektanzkurve mit Savitzky-Golay-Filter
            if apply_smoothing:
                # Parameter
                window_length = 11  # ungerade Zahl, z. B. 11, 15, 21
                polyorder = 2       # Grad des Polynoms
                # Glättung anwenden
                r_values_smooth = savgol_filter(r_values, window_length=window_length, polyorder=polyorder)
        else:
            raise ValueError("Keine Referenzdatei gefunden. Reflektanz kann nicht berechnet werden.")

        # In feature_matrix einfügen
        if apply_smoothing:
            # verwendete geglättete Werte
            feature_matrix[i, :len(r_values_smooth)] = r_values_smooth
        else:   
            # Fallback auf unveränderte Reflektanzwerte     
            feature_matrix[i, :len(r_values)] = r_values
        # Wellenlängenbereich reduzieren für Training und Plot (VIS: 440-800nm, NIR: 1050-1850nm)
        if sheet_name == 'NIR':
            mask = (wavelengths >= 1050) & (wavelengths <= 1850)
        elif sheet_name == 'VIS':   
            mask = (wavelengths >= 440) & (wavelengths <= 800) 
        else:
            mask = np.ones_like(wavelengths, dtype=bool)  # kein Maskieren   
        # Mask anwenden
        feature_matrix[i, ~mask] = np.nan  # Werte außerhalb des Bereichs auf NaN setzen

    # Tabelle + Labels erstellen
    df_features = pd.DataFrame(feature_matrix)
    df_features["OilType"] = pd.Categorical(labels) # Spalte "Oiltype" einfügen, als kategorische Variable
    df_features["Source"] = pd.Categorical(sources) # Spalte "Source" einfügen, als kategorische Variable

    # Speichern als Excel-Datei mit Dateinamen basierend auf Anregungsquelle
    try:
        # Nur einmal aus der ersten Datei den Wert holen
        info_data = pd.read_excel(data_files[0], sheet_name='Info')
        Anregungs_Wellenlaenge = str(info_data.iloc[2, 2])
    except Exception as e:
        print(f"Fehler beim Auslesen der Wellenlänge aus {data_files[0].name}: {e}")
        Anregungs_Wellenlaenge = "unknown"

    matrix_title = f"{sheet_name}-Reflexionsspektren Anregung@{Anregungs_Wellenlaenge}.xlsx"
    output_file = output_dir / matrix_title
    df_features.to_excel(output_file, index=False)
    print(f"Preprocessing_reflectance abgeschlossen. Datei gespeichert unter: {output_file}")

    return matrix_title, df_features

def get_nipy_spectral_colors(n):
    cmap = plt.get_cmap("nipy_spectral")    # große Colormap mit vielen Farben
    return [cmap(i / n) for i in range(n)]  # geeignet für wissenschaftliche Plots

def plot_matrix(matrix: pd.DataFrame, title: str, output_dir: Path = None, figsize=(20,12), excludeOiltype: Union[str, list, tuple, set] = "data_1", excludeSource: Union[str, list, tuple, set] = "data_2"):
    print("Starte Plotting...", flush=True)
    #***Daten vorbereiten für Plot*** 
    # Wellenlängen für die Werte der x-Achse des Plots laden
    wavelengths = None
    if output_dir is not None:
        if (output_dir / "wavelengths.npy").exists():
            wavelengths = np.load(output_dir / "wavelengths.npy")
            print(f"Wellenlängen aus 'wavelengths.npy' geladen.\n", flush=True)
        elif (output_dir / "VIS_wavelengths.npy").exists():
            wavelengths = np.load(output_dir / "VIS_wavelengths.npy")
            print(f"Wellenlängen aus 'VIS_wavelengths.npy' geladen.\n", flush=True)
        elif (output_dir / "NIR_wavelengths.npy").exists():
            wavelengths = np.load(output_dir / "NIR_wavelengths.npy")
            print(f"Wellenlängen aus 'NIR_wavelengths.npy' geladen.\n", flush=True)
        else:
            print(f"Wellenlängen konnten nicht geladen werden. Fallback auf Pixelindex.\n", flush=True)

    #%% Ausschließen bestimmter Öltypen oder Quellen, falls angegeben
    # Zuerst Öltypen ausschließen, falls angegeben
    if excludeOiltype is None:
        exclude_set = set()
    elif isinstance(excludeOiltype, (list, tuple, set)):
        exclude_set = set(excludeOiltype)
    else:
        raise TypeError("exclude must be None, list, tuple, or set") # entweder leer lassen oder Liste/Tupel/Set übergeben
    # Maske für Zeilen, die nicht ausgeschlossen werden sollen
    mask = ~matrix["OilType"].isin(exclude_set)

    # Danach Ölmarken/-hersteller ausschließen, falls angegeben
    if excludeSource is None:
        exclude_set = set()
    elif isinstance(excludeSource, (list, tuple, set)):
        exclude_set = set(excludeSource)
    else:
        raise TypeError("exclude must be None, list, tuple, or set") # entweder leer lassen oder Liste/Tupel/Set übergeben
    # mask erweitern/aktualisieren mit mask &= ~...
    mask &= ~matrix["Source"].isin(excludeSource)
    # Gefilterte Matrix erstellen
    matrix = matrix.loc[mask].reset_index(drop=True)

    #%% Daten für Plot extrahieren
    features = matrix.drop(columns=["OilType","Source"]).T   #erst Ölklassen und Ölmarken entfernen, danach transponieren für Plot
    if wavelengths is None:
        wavelengths = np.arange(features.shape[0])  # Wenn wavelengths-Datei nicht vorhanden, generiere Pixel-Indizes nach Anzahl der Zeilen von features
    labels = matrix["OilType"].values  #Ölklassen extrahieren für Legende
    sources = matrix["Source"].values  #Ölmarke/-hersteller extrahieren für Farbcodierung
    unique_sources = np.unique(sources)# eindeutige Ölmarken/-hersteller für die Legende und Farbcodierung      

    #%% Farben für Ölmarken/-hersteller definieren
    # Jeder Ölmarke Farbe zuweisen
    #colors = list(mcolors.TABLEAU_COLORS.values()) # 10 verschiedene kräftige Farben
    colors = get_nipy_spectral_colors(len(unique_sources))  # größere Colormap für mehr Farben, falls mehr Ölmarken vorhanden
    color_map = {source: colors[i] for i, source in enumerate(sorted(unique_sources))}
    
    #%% Plot erstellen
    plt.figure(figsize=figsize)
    # Alle Proben plotten, farblich nach Ölmarke/-hersteller, linienstil nach Ölklasse
    seen = set() # um doppelte Legenden-Einträge zu vermeiden in dieser for-Schleife
    # Die Listen-Datentyp set() speichert Elemente ohne Duplikate; ansonsten wie eine normale Liste
    for i in range(features.shape[1]):
        oil_type = labels[i]
        source = sources[i]
        color = color_map[source]
        key = (oil_type, source)
        if key not in seen:
            label = f"{oil_type} ({source})"
            seen.add(key)
        else:
            label = None  # kein Label für doppelte Einträge
        plt.plot(wavelengths, features.iloc[:, i], color=color,
                alpha=0.8 if label else 0.5, label=label)
        
    # Plot-Details
    plt.xlabel("Wavelength [nm]")
    if "Fluoreszenz" in title:
        plt.xlabel("Wavelength [nm]", fontsize=20)
        plt.ylabel("Intensity (raw values/counts)", fontsize=20)
        if "@325nm" in title:
            plotTitle = "Fluorescence Emission Spectrum @325nm"
        elif "@365nm" in title:
            plotTitle = "Fluorescence Emission Spectrum @365nm"
        elif "@275nm" in title:
            plotTitle = "Fluorescence Emission Spectrum @275nm"
        else:
            plotTitle = "Fluorescence Emission Spectrum"
        plt.title(plotTitle, fontsize=24)
        # x-Achse je nach VIS/NIR einschränken
        if "VIS" in title:
            plt.xlim(440, 800)  # VIS im Bereich 440-800nm
        elif "NIR" in title:
            plt.xlim(1000, 1850) # NIR im Bereich 1000-1900nm
        else:
            plt.xlim(np.min(wavelengths), np.max(wavelengths)) # gesamter Wellenlängenbereich
    elif "Reflexion" or "lamp" or "average" in title:
        plt.xlabel("Wavelength [nm]", fontsize=20)
        plt.ylabel("Reflectance", fontsize=20)
        if "NIR" in title:
            plotTitle = "NIR spectrum"
        elif "VIS" in title:
            plotTitle = "VIS spectrum"
        else:
            plotTitle = "spectrum"
        plt.title(plotTitle, fontsize=24)
        plt.ylabel("Reflectance")
        # Wenn Reflektanz berechnet wurde (was mit kleinen y-Werte erfassbar ist), y-Achse entsprechend setzen
        plt.ylim(0, 2)  # Reflektanz üblicherweise zwischen 0 und 2
        if "VIS" in title:
            plt.xlim(440, 800)  # VIS im Bereich 440-800nm
        elif "NIR" in title:
            plt.xlim(1050, 1850) # NIR im Bereich 1000-1850nm
        else:
            plt.xlim(np.min(wavelengths), np.max(wavelengths)) # gesamter Wellenlängenbereich
    else:
        plt.ylabel("Intensity (Counts)", fontsize=20)
        plt.xlabel("Wavelength [nm]", fontsize=20)
        plt.xlim(np.min(wavelengths), np.max(wavelengths)) # gesamter Wellenlängenbereich
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tick_params(axis='both', labelsize=16)

    # Legende:
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=18)
    # Plot speichern
    save_basename = Path(title).stem if title else "matrix_plot"
    save_name = f"{save_basename}_plot.png"
    if output_dir is not None:
        plt.savefig(output_dir / save_name, bbox_inches='tight')
        #bbox_inches='tight' sorgt dafür, dass die Legende nicht abgeschnitten wird
    else:
        plt.savefig("matrix_plot", bbox_inches='tight')
    plt.close()

    print(f"Plotting abgeschlossen. Datei gespeichert unter: {output_dir}")

if __name__ == "__main__":
    #%% Reflectance
    # Reflectance VIS
    #input_dir = Path(r"C:\Users\korkm\Desktop\5.Sem (Master-Thesis)\Thesis\Messungen\lamp\Zeliva\meas5")
    #output_dir = Path(r"C:\Users\korkm\Desktop\5.Sem (Master-Thesis)\Thesis\Messungen\lamp\Zeliva\meas5\preprocessed_VIS\Savitzky-Golay-Filter")
    # preprocessing_reflectance durchführen und den erzeugten Dateinamen erhalten
    #matrix_title, matrix = preprocessing_reflectance(input_dir, output_dir, sheet_name='VIS', apply_smoothing=True)
    #matrix_stem = Path(matrix_title).stem  # Nur der Name ohne .xlsx
    #plot_matrix(matrix=matrix, title=matrix_stem, output_dir=output_dir, figsize=(20,12), 
    #            excludeOiltype=["reference"], 
    #            excludeSource=[])

    # Reflectance NIR
    #input_dir = Path(r"C:\Users\korkm\Desktop\5.Sem (Master-Thesis)\Thesis\Messungen\lamp\80 refined - 20 extra native\new\meas4")
    #output_dir = Path(r"C:\Users\korkm\Desktop\5.Sem (Master-Thesis)\Thesis\Messungen\lamp\Zeliva\meas5\preprocessed_NIR\Savitzky-Golay-Filter")
    # preprocessing_reflectance durchführen und den erzeugten Dateinamen erhalten
    #matrix_title, matrix = preprocessing_reflectance(input_dir, output_dir, sheet_name='NIR', apply_smoothing=True)
    #matrix_stem = Path(matrix_title).stem  # Nur der Name ohne .xlsx
    #plot_matrix(matrix=matrix, title=matrix_stem, output_dir=output_dir, figsize=(20,12), 
    #            excludeOiltype=["reference"], 
    #            excludeSource=[])
    #%% Fluorescence
    # preprocessing_fluor
    #input_dir = Path(r"C:\Users\korkm\Desktop\5.Sem (Master-Thesis)\Thesis\Messungen\lamp\lamp_allData\VIS")
    #output_dir = Path(r"C:\Users\korkm\Desktop\5.Sem (Master-Thesis)\Thesis\Messungen\lamp\lamp_allData_matrix\VIS\Vegetable oils\fix")

    #matrix_title, matrix = preprocessing_fluor(input_dir, output_dir, sheet_name='VIS')
    #matrix_title, _ = preprocessing_fluor(input_dir, output_dir, sheet_name='NIR')
    #%% Matrix plotten
    # Datei laden und plotten (keine zweite Ausführung von preprocessing)
    # 'matrix' kommt bereits von preprocessing_fluor, kein erneutes Lesen nötig
    # (falls du aus einer separaten Sitzung startest, dann stattdessen pd.read_excel verwenden)
    #matrix = pd.read_excel(output_dir / "lamp_VIS_classification_matrix.xlsx")
    #matrix_stem = Path(matrix_title).stem  # Nur der Name ohne .xlsx
    #plot_matrix(matrix=matrix, title="lamp_VIS_classification_matrix.xlsx", output_dir=output_dir, figsize=(20,12), 
    #            excludeOiltype=["reference","extra virgin olive oil","refined olive oil","refined olive-pomace oil","olive-pomace oil","blend","out-of-class"], 
    #            excludeSource=[])
    
    #excludeOiltype=["reference","extra virgin olive oil","refined olive oil","refined olive-pomace oil","olive-pomace oil","sunflower oil","rapeseed oil","out-of-class"], --> Nur Blends
    #excludeOiltype=["reference","extra virgin olive oil","refined olive oil","refined olive-pomace oil","olive-pomace oil","blend","out-of-class"], --> Nur pflanlzliche Öle
    #excludeOiltype=["reference","extra virgin olive oil","refined olive oil","refined olive-pomace oil","olive-pomace oil","blend","sunflower oil","rapeseed oil"], --> Nur out-of-class Öle
    #excludeOiltype=["reference","blend","sunflower oil","rapeseed oil","out-of-class"], --> Nur pure Olivenöle

    #excludeSource=[] --> Alle Quellen/Marken anzeigen
    # Blends 1
    #excludeSource=["V2-20pct refined–80pct extra native","V2-50pct refined–50pct extra native","V2-80pct refined–20pct extra native","V3-20pct refined–80pct extra native","V3-50pct refined–50pct extra native","V3-80pct refined–20pct extra native"]
    # Blends 2
    #excludeSource=["20pct refined–80pct extra native","50pct refined–50pct extra native","80pct refined–20pct extra native","V3-20pct refined–80pct extra native","V3-50pct refined–50pct extra native","V3-80pct refined–20pct extra native"]
    # Blends 3
    #excludeSource=["20pct refined–80pct extra native","50pct refined–50pct extra native","80pct refined–20pct extra native","V2-20pct refined–80pct extra native","V2-50pct refined–50pct extra native","V2-80pct refined–20pct extra native"]

    output_dir_VIS = Path(r"C:\Users\korkm\Desktop\5.Sem (Master-Thesis)\Thesis\Messungen\lamp\lamp_Savitzky-Golay-Filter_allData_matrix\VIS")
    output_dir_NIR = Path(r"C:\Users\korkm\Desktop\5.Sem (Master-Thesis)\Thesis\Messungen\lamp\lamp_Savitzky-Golay-Filter_allData_matrix\NIR")

    matrix = pd.read_excel(output_dir_VIS / "lamp_VIS_classification_matrix.xlsx")
    plot_matrix(matrix=matrix, title="lamp_VIS_classification_matrix.xlsx", output_dir=output_dir_VIS, figsize=(20,12), 
                excludeOiltype=["reference","extra virgin olive oil","refined olive oil","refined olive-pomace oil","olive-pomace oil","blend","out-of-class"], 
                excludeSource=[])
    
    matrix = pd.read_excel(output_dir_NIR / "lamp_NIR_classification_matrix.xlsx")
    plot_matrix(matrix=matrix, title="lamp_NIR_classification_matrix.xlsx", output_dir=output_dir_NIR, figsize=(20,12), 
                excludeOiltype=["reference","extra virgin olive oil","refined olive oil","refined olive-pomace oil","olive-pomace oil","blend","out-of-class"], 
                excludeSource=[])
#*******************************************************************************************************************
# ML-Modellierung erfolgt im separaten Skript "ML_pipeline.py"
#*******************************************************************************************************************