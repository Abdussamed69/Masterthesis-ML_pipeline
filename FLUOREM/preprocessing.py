#2025-08-17 Author:             Abdussamed Korkmaz 
#                               Master Electrical Systems Engineering 
#                               University of Heilbronn, Germany
#2025-08-17 Description:        Preprocessing script for fluorescence- & reflection-data 
#*******************************************************************************************************************
#This script reads and plots all Excel files,
# **exclusively based** on the Excel files created by the script "Fluorem_v2.py".
# A feature matrix is created which can be used in a separate script for machine learning modeling.
#*******************************************************************************************************************
# Important: It is assumed that the Excel files have a specific structure:
# 20250912-194600_P.224-990056_LIDL_extra virgin olive oil.xlsx
#        ↑           ↑          ↑            ↑
#    Timestamp | Device-No. | OilSource | OilType
#*******************************************************************************************************************

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from scipy.signal import savgol_filter

def preprocessing_fluor(input_dir: Union[str, Path], output_dir: Union[str, Path], sheet_name: str):
    """Preprocessing of fluorescence data.
    Reads all Excel files in the specified directory."""
    
    print("Starting preprocessing_fluor...", flush=True)

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)  
              
    # Read all Excel files
    file_paths = list(Path(input_dir).glob("*.xlsx"))
    if not file_paths:
        raise FileNotFoundError("No .xlsx files were found in the specified directory.")

    # Read measured values ​​and extract labels and oil brand/manufacturer from file names
    observations = []
    labels = []
    sources = []
    for file_path in file_paths:
        try:
            # Read measured values
            data = pd.read_excel(file_path, sheet_name=sheet_name)
            observations.append(data)
            # extract labels and oil brand/manufacturer from file names (everything after the last "_")
            name = file_path.stem  # Filename without .xlsx
            parts = name.split("_")
            if len(parts) >= 2:
                labels.append(parts[-1])  # last part = label --> classification
                sources.append(parts[-2])  # penultimate/second from the last part = oil brand/manufacturer
            else:
                labels.append("Unknown")
                sources.append("Unknown")
        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")

    # Extract wavelengths
    wavelengths = observations[0].iloc[:, 1].values  # Column 2 = Wavelength
    # Save wavelengths separately (for later plotting)
    np.save(output_dir / "wavelengths.npy", wavelengths)

    # Initialize feature matrix (NaN), maximum number of rows = maxLength
    # max_length = max(len(df) for df in observations)
    max_length = len(observations[0])  # Assumption: all files have the same length
    feature_matrix = np.full(                       
        (len(observations), max_length),  # Form: (rows: number of spectra, columns: max number of wavelengths)
        np.nan                            # Initial value: NaN
    )
    # Preprocessing for each data set
    for i, df in enumerate(observations):
        s_values = df.iloc[:, 2].values     # Column 3 = Intensity        
        if np.any(pd.isna(s_values)):       # Fill up (interpolate) NaN
            s_values = pd.Series(s_values).interpolate().bfill().ffill().values

        # Insert into feature_matrix    
        feature_matrix[i, :len(s_values)] = s_values
        # Reduce wavelength range for training and plotting (VIS: 440-800nm, NIR: 1050-1850nm)
        if sheet_name == 'NIR':
            mask = (wavelengths >= 1050) & (wavelengths <= 1850)
        elif sheet_name == 'VIS':   
            mask = (wavelengths >= 440) & (wavelengths <= 800)
        else:
            mask = np.ones_like(wavelengths, dtype=bool)  # no masking, keep all values
        # Apply mask
        feature_matrix[i, ~mask] = np.nan  # Set values outside the range to NaN


    # Create table + labels
    df_features = pd.DataFrame(feature_matrix)
    df_features["OilType"] = pd.Categorical(labels) # Insert the column "Oiltype" as a categorical variable.
    df_features["Source"] = pd.Categorical(sources) # Insert the column "Source" as a categorical variable.

    # Save as Excel file with filename based on suggestion source
    try:
        # Retrieve the value from the first file only once
        info_data = pd.read_excel(file_paths[0], sheet_name='Info')
        Anregungs_Wellenlaenge = str(info_data.iloc[2, 2])
    except Exception as e:
        print(f"Error reading wavelength from {file_paths[0].name}: {e}")
        Anregungs_Wellenlaenge = "unknown"

    matrix_title = f"{sheet_name}-Fluorescence-Emission-Spectra_Excitation@{Anregungs_Wellenlaenge}.xlsx"
    output_file = output_dir / matrix_title
    df_features.to_excel(output_file, index=False)
    print(f"preprocessing_fluor completed. File saved to: {output_file}")
    
    return matrix_title, df_features

def preprocessing_reflectance(input_dir: Union[str, Path], output_dir: Union[str, Path], sheet_name: str, apply_smoothing: bool = True):
    """Preprocessing of reflectance data.
    Reads all Excel files in the specified directory, calculates the reflectance (if a reference exists),
    creates a feature matrix, and saves it as an Excel file."""
    
    print("Starting preprocessing_reflectance...", flush=True)
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    # Create a folder if one does not already exist.
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find reference file for calculating reflectance
    # reference file (optional)
    ref_files = sorted(input_dir.glob("*_reference.xlsx"))
    ref_values = None
    if len(ref_files) == 0:
        print("No reference file with '_reference.xlsx' was found. Reflection is calculated without a reference.", flush=True)
    elif len(ref_files) > 1:
        raise ValueError("Multiple reference files found. Please leave only one in the folder.")
    else:
        ref_df = pd.read_excel(ref_files[0], sheet_name=sheet_name)
        if ref_df.shape[1] < 3:
            raise ValueError(f"Reference sheet has less than 3 columns: {ref_files[0].name}")
        # Extract reference values ​​(digital counts)
        ref_values = ref_df.iloc[:, 2].values # 3rd column in Excel = digital counts
        print(f"Reference file used: {ref_files[0].name}", flush=True)
        # Save reference values separately (for later plotting)
        np.save(output_dir / "ref_values.npy", ref_values)

    # Read all Excel files
    file_paths = list(Path(input_dir).glob("*.xlsx"))
    if not file_paths:
        raise FileNotFoundError("No .xlsx files found in the specified directory.")
    # exclude reference file(s) from data file list
    if len(ref_files) == 1:
        data_files = [p for p in file_paths if p != ref_files[0]]
    else:
        data_files = file_paths.copy()
    if not data_files:
        raise ValueError("No data files found.")
    
    # Read measured values, and extract labels and oil brand/manufacturer from file names
    observations = []
    labels = []
    sources = []
    for file_path in data_files:
        try:
            # Read measured values
            data = pd.read_excel(file_path, sheet_name=sheet_name)
            observations.append(data)
            # extract labels and oil brand/manufacturer from file names (everything after the last "_")
            name = file_path.stem  # Filename without .xlsx
            parts = name.split("_")
            if len(parts) >= 2:
                labels.append(parts[-1])  # last part = label --> classification
                sources.append(parts[-2])  # penultimate/second last part = oil brand/manufacturer
            else:
                labels.append("Unknown")
                sources.append("Unknown")
        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")

    # Extract wavelengths
    wavelengths = observations[0].iloc[:, 1].values  # Column 2 = Wavelength
    # Save wavelengths separately (for later plotting)
    np.save(output_dir / "wavelengths.npy", wavelengths)

    # Initialize feature matrix (NaN), maximum number of rows = max_length
    #max_length = max(len(df) for df in observations)
    max_length = len(observations[0])  # assumption: all files have the same length
    feature_matrix = np.full((len(observations), max_length), np.nan)

    # Preprocessing for each dataset
    for i, df in enumerate(observations):
        s_values = df.iloc[:, 2].values     # Column 3 = intensity
        if np.any(pd.isna(s_values)):       # Fill NaNs (interpolate)
            s_values = pd.Series(s_values).interpolate().bfill().ffill().values

        if len(ref_files) > 0:              
            # Only if a reference file exists and matches
            if len(s_values) != len(ref_values):    
                raise ValueError(f"Mismatched length in file: {data_files[i].name}")       
            # *****Calculate reflectance: R(λ) = S(λ) / Ref(λ)*****
            # Protect against very small reference values
            min_ref = 10  # avoid too small reference values -> robust curve
            ref_values_safe = np.where(ref_values < min_ref, min_ref, ref_values)
            r_values = s_values / ref_values_safe
            # Smoothing of the reflectance curve with Savitzky-Golay filter
            if apply_smoothing:
                # Parameter
                window_length = 11  # odd number, e.g. 11, 15, 21
                polyorder = 2       # degree of the polynomial
                # Apply filter
                r_values_smooth = savgol_filter(r_values, window_length=window_length, polyorder=polyorder)
        else:
            raise ValueError("No reference file found. Reflectance cannot be calculated.")

        # Insert into feature_matrix
        if apply_smoothing:
            # use smoothed values
            feature_matrix[i, :len(r_values_smooth)] = r_values_smooth
        else:
            # fallback to unsmoothed reflectance values
            feature_matrix[i, :len(r_values)] = r_values
        # Reduce wavelength range for training and plotting (VIS: 440-800nm, NIR: 1050-1850nm)
        if sheet_name == 'NIR':
            mask = (wavelengths >= 1050) & (wavelengths <= 1850)
        elif sheet_name == 'VIS':   
            mask = (wavelengths >= 440) & (wavelengths <= 800) 
        else:
            mask = np.ones_like(wavelengths, dtype=bool)  # no masking
        # Apply mask
        feature_matrix[i, ~mask] = np.nan  # Set values outside the range to NaN

    # Create table + labels
    df_features = pd.DataFrame(feature_matrix)
    df_features["OilType"] = pd.Categorical(labels) # Insert the "OilType" column as a categorical variable
    df_features["Source"] = pd.Categorical(sources) # Insert the "Source" column as a categorical variable

    # Save as Excel file with filename based on excitation source
    try:
        # Retrieve the value from the first file only once
        info_data = pd.read_excel(data_files[0], sheet_name='Info')
        Anregungs_Wellenlaenge = str(info_data.iloc[2, 2])
    except Exception as e:
        print(f"Error reading wavelength from {data_files[0].name}: {e}")
        Anregungs_Wellenlaenge = "unknown"

    matrix_title = f"{sheet_name}-Reflexionsspektren Anregung@{Anregungs_Wellenlaenge}.xlsx"
    output_file = output_dir / matrix_title
    df_features.to_excel(output_file, index=False)
    print(f"preprocessing_reflectance completed. File saved to: {output_file}")

    return matrix_title, df_features

def get_nipy_spectral_colors(n):
    cmap = plt.get_cmap("nipy_spectral")    # large colormap with many colors
    return [cmap(i / n) for i in range(n)]  # suitable for scientific plots

def plot_matrix(matrix: pd.DataFrame, title: str, output_dir: Path = None, figsize=(20,12), excludeOiltype: Union[str, list, tuple, set] = "data_1", excludeSource: Union[str, list, tuple, set] = "data_2"):
    print("Starting plotting...", flush=True)
    #***Prepare data for plotting*** 
    # Load wavelengths for the x-axis values
    wavelengths = None
    if output_dir is not None:
        if (output_dir / "wavelengths.npy").exists():
            wavelengths = np.load(output_dir / "wavelengths.npy")
            print(f"Wavelengths loaded from 'wavelengths.npy'.\n", flush=True)
        elif (output_dir / "VIS_wavelengths.npy").exists():
            wavelengths = np.load(output_dir / "VIS_wavelengths.npy")
            print(f"Wavelengths loaded from 'VIS_wavelengths.npy'.\n", flush=True)
        elif (output_dir / "NIR_wavelengths.npy").exists():
            wavelengths = np.load(output_dir / "NIR_wavelengths.npy")
            print(f"Wavelengths loaded from 'NIR_wavelengths.npy'.\n", flush=True)
        else:
            print(f"Wavelengths could not be loaded. Falling back to pixel index.\n", flush=True)

    #%% Exclude specific oil types or sources if specified
    # First exclude oil types if provided
    if excludeOiltype is None:
        exclude_set = set()
    elif isinstance(excludeOiltype, (list, tuple, set)):
        exclude_set = set(excludeOiltype)
    else:
        raise TypeError("exclude must be None, list, tuple, or set") # either leave empty or pass a list/tuple/set
    # Mask for rows that should not be excluded
    mask = ~matrix["OilType"].isin(exclude_set)

    # Then exclude oil brands/manufacturers if provided
    if excludeSource is None:
        exclude_set = set()
    elif isinstance(excludeSource, (list, tuple, set)):
        exclude_set = set(excludeSource)
    else:
        raise TypeError("exclude must be None, list, tuple, or set") # either leave empty or pass a list/tuple/set
    # update mask with mask &= ~...
    mask &= ~matrix["Source"].isin(excludeSource)
    # Create filtered matrix
    matrix = matrix.loc[mask].reset_index(drop=True)

    #%% Extract data for plotting
    features = matrix.drop(columns=["OilType","Source"]).T   # remove oil types and brands, then transpose for plotting
    if wavelengths is None:
        wavelengths = np.arange(features.shape[0])  # If wavelengths file is missing, generate pixel indices by number of rows in features
    labels = matrix["OilType"].values  # extract oil types for legend
    sources = matrix["Source"].values  # extract oil brands/manufacturers for color coding
    unique_sources = np.unique(sources)  # unique brands for legend and color mapping      

    #%% Define colors for oil brands/manufacturers
    # Assign a color to each brand
    #colors = list(mcolors.TABLEAU_COLORS.values()) # 10 distinct strong colors
    colors = get_nipy_spectral_colors(len(unique_sources))  # larger colormap for more brands
    color_map = {source: colors[i] for i, source in enumerate(sorted(unique_sources))}
    
    #%% Create plot
    plt.figure(figsize=figsize)
    # Plot all samples, colored by brand/manufacturer, line style by oil class
    seen = set() # to avoid duplicate legend entries in this loop
    # The set() type stores elements without duplicates; otherwise behaves like a list
    for i in range(features.shape[1]):
        oil_type = labels[i]
        source = sources[i]
        color = color_map[source]
        key = (oil_type, source)
        if key not in seen:
            label = f"{oil_type} ({source})"
            seen.add(key)
        else:
            label = None  # no label for duplicate entries
        plt.plot(wavelengths, features.iloc[:, i], color=color,
                alpha=0.8 if label else 0.5, label=label)
        
    # Plot details
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
        # limit x-axis depending on VIS/NIR
        if "VIS" in title:
            plt.xlim(440, 800)  # VIS range 440-800nm
        elif "NIR" in title:
            plt.xlim(1000, 1850) # NIR range 1000-1900nm
        else:
            plt.xlim(np.min(wavelengths), np.max(wavelengths)) # entire wavelength range
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
        # If reflectance was calculated (detectable by small y-values), set y-axis accordingly
        plt.ylim(0, 2)  # Reflectance typically between 0 and 2
        if "VIS" in title:
            plt.xlim(440, 800)  # VIS range 440-800nm
        elif "NIR" in title:
            plt.xlim(1050, 1850) # NIR range 1000-1850nm
        else:
            plt.xlim(np.min(wavelengths), np.max(wavelengths)) # full wavelength range
    else:
        plt.ylabel("Intensity (Counts)", fontsize=20)
        plt.xlabel("Wavelength [nm]", fontsize=20)
        plt.xlim(np.min(wavelengths), np.max(wavelengths)) # full wavelength range
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tick_params(axis='both', labelsize=16)

    # Legend:
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=18)
    # Plot speichern
    save_basename = Path(title).stem if title else "matrix_plot"
    save_name = f"{save_basename}_plot.png"
    if output_dir is not None:
        plt.savefig(output_dir / save_name, bbox_inches='tight')
        # bbox_inches='tight' ensures the legend is not cut off
    else:
        plt.savefig("matrix_plot", bbox_inches='tight')
    plt.close()

    print(f"Plotting completed. File saved to: {output_dir}")

if __name__ == "__main__":
    #%% Reflectance examples
    # Reflectance VIS example (uncomment and set paths to run)
    #input_dir = Path(r"C:\Users\korkm\Desktop\5.Sem (Master-Thesis)\Thesis\Messungen\lamp\Zeliva\meas5")
    #output_dir = Path(r"C:\Users\korkm\Desktop\5.Sem (Master-Thesis)\Thesis\Messungen\lamp\Zeliva\meas5\preprocessed_VIS\Savitzky-Golay-Filter")
    # Run preprocessing_reflectance and get the generated filename
    #matrix_title, matrix = preprocessing_reflectance(input_dir, output_dir, sheet_name='VIS', apply_smoothing=True)
    #matrix_stem = Path(matrix_title).stem  # name without .xlsx
    #plot_matrix(matrix=matrix, title=matrix_stem, output_dir=output_dir, figsize=(20,12), 
    #            excludeOiltype=["reference"], 
    #            excludeSource=[])

    # Reflectance NIR example
    #input_dir = Path(r"C:\Users\korkm\Desktop\5.Sem (Master-Thesis)\Thesis\Messungen\lamp\80 refined - 20 extra native\new\meas4")
    #output_dir = Path(r"C:\Users\korkm\Desktop\5.Sem (Master-Thesis)\Thesis\Messungen\lamp\Zeliva\meas5\preprocessed_NIR\Savitzky-Golay-Filter")
    # Run preprocessing_reflectance and get the generated filename
    #matrix_title, matrix = preprocessing_reflectance(input_dir, output_dir, sheet_name='NIR', apply_smoothing=True)
    #matrix_stem = Path(matrix_title).stem  # name without .xlsx
    #plot_matrix(matrix=matrix, title=matrix_stem, output_dir=output_dir, figsize=(20,12), 
    #            excludeOiltype=["reference"], 
    #            excludeSource=[])
    #%% Fluorescence examples
    # Run preprocessing_fluor
    #input_dir = Path(r"C:\Users\korkm\Desktop\5.Sem (Master-Thesis)\Thesis\Messungen\lamp\lamp_allData\VIS")
    #output_dir = Path(r"C:\Users\korkm\Desktop\5.Sem (Master-Thesis)\Thesis\Messungen\lamp\lamp_allData_matrix\VIS\Vegetable oils\fix")

    #matrix_title, matrix = preprocessing_fluor(input_dir, output_dir, sheet_name='VIS')
    #matrix_title, _ = preprocessing_fluor(input_dir, output_dir, sheet_name='NIR')
    #%% Plot a saved matrix
    # Load file and plot (do not run preprocessing again)
    # 'matrix' may already come from preprocessing_fluor; otherwise use pd.read_excel
    #matrix = pd.read_excel(output_dir / "lamp_VIS_classification_matrix.xlsx")
    #matrix_stem = Path(matrix_title).stem  # name without .xlsx
    #plot_matrix(matrix=matrix, title="lamp_VIS_classification_matrix.xlsx", output_dir=output_dir, figsize=(20,12), 
    #            excludeOiltype=["reference","extra virgin olive oil","refined olive oil","refined olive-pomace oil","olive-pomace oil","blend","out-of-class"], 
    #            excludeSource=[])
    
    # Examples for excludeOiltype filters (German comments converted):
    #excludeOiltype=["reference","extra virgin olive oil","refined olive oil","refined olive-pomace oil","olive-pomace oil","sunflower oil","rapeseed oil","out-of-class"], --> Only blends
    #excludeOiltype=["reference","extra virgin olive oil","refined olive oil","refined olive-pomace oil","olive-pomace oil","blend","out-of-class"], --> Only vegetable oils
    #excludeOiltype=["reference","extra virgin olive oil","refined olive oil","refined olive-pomace oil","olive-pomace oil","blend","sunflower oil","rapeseed oil"], --> Only out-of-class oils
    #excludeOiltype=["reference","blend","sunflower oil","rapeseed oil","out-of-class"], --> Only pure olive oils

    #excludeSource=[] --> Show all sources/brands
    # Blend examples
    #excludeSource=["V2-20pct refined–80pct extra native","V2-50pct refined–50pct extra native","V2-80pct refined–20pct extra native","V3-20pct refined–80pct extra native","V3-50pct refined–50pct extra native","V3-80pct refined–20pct extra native"]
    #excludeSource=["20pct refined–80pct extra native","50pct refined–50pct extra native","80pct refined–20pct extra native","V3-20pct refined–80pct extra native","V3-50pct refined–50pct extra native","V3-80pct refined–20pct extra native"]
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