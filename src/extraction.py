import pandas as pd
import numpy as np
from pathlib import Path

# Load all JSON files in the folder as DataFrames, keyed by filename stem
def load_all_data(data_folder: str = "data_raw") -> dict:
    """
    Load every JSON file in the given folder into a dict of DataFrames.
    Keys are file stems (e.g. 'Patients', 'Consultation', 'tKERATO', …).
    """
    base_path = Path(data_folder)
    dfs = {}
    for file_path in base_path.glob("*.json"):
        dfs[file_path.stem] = pd.read_json(file_path)
    print(f"[LOAD] {len(dfs)} file(s) loaded: {sorted(dfs.keys())}")
    return dfs

# Normalize any ID value to a string for comparison
def normalize_id(value) -> str | None:
    """
    Convert any ID value to a plain string for safe cross-file comparison.
    Handles float, int, string, None/NaN.
    """
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    try:
        return str(int(float(str(value).strip())))
    except (ValueError, OverflowError):
        s = str(value).strip()
        return s if s else None

# Clean DataFrame: drop empty columns, replace null-like strings with NaN
def clean_df(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Drop fully-empty columns and normalize null-like strings (only on string columns).
    """
    if df is None or df.empty:
        return None
    str_cols = df.select_dtypes(include="object").columns
    if len(str_cols) > 0:
        df = df.copy()
        df[str_cols] = df[str_cols].replace(
            [r"^\s*$", "NaN", "nan", "null", "None", ""],
            np.nan,
            regex=True,
        )
    df = df.dropna(axis=1, how="all")
    return df if not df.empty else None

# Build the complete patient record by linking all relevant data
def get_full_patient_record(dfs: dict, patient_name: str) -> dict | None:
    """
    Build the complete medical file for a patient by following all ID links across the loaded JSON files.
    Returns a dict of { section_name: DataFrame } or None if not found.
    """
    
    # Find patient in Patients.json
    df_patients = dfs.get("Patients")
    if df_patients is None:
        print("[ERROR] 'Patients' not found in loaded data.")
        return None
    import unicodedata
    
    # Combine and normalize name and prename, remove accents
    if "NOM" in df_patients.columns:
        nom_col = df_patients["NOM"].fillna("").astype(str)
    else:
        nom_col = pd.Series("", index=df_patients.index)
    if "Prénom" in df_patients.columns:
        prenom_col = df_patients["Prénom"]
    elif "PRENOM" in df_patients.columns:
        prenom_col = df_patients["PRENOM"]
    elif "Prenom" in df_patients.columns:
        prenom_col = df_patients["Prenom"]
    else:
        prenom_col = pd.Series("", index=df_patients.index)
    prenom_col = prenom_col.fillna("").astype(str)
    combined_names = (nom_col + " " + prenom_col).str.lower()
    combined_names = combined_names.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    clean_search = unicodedata.normalize('NFKD', str(patient_name)).encode('ascii', 'ignore').decode('utf-8')
    search_terms = clean_search.lower().split()
    mask = pd.Series(True, index=df_patients.index)
    for term in search_terms:
        mask = mask & combined_names.str.contains(term, regex=False)
    match = df_patients[mask].copy()
    if match.empty:
        print(f"[NOT FOUND] No patient matching '{patient_name}'.")
        return None
    patient_id = normalize_id(match.iloc[0]["Code patient"])
    print(f"[OK] Patient found — Code patient: {patient_id}")
    record = {"identity": clean_df(match)}
    
    # Build doctor ID to name lookup
    doctor_map = {}
    if "person" in dfs:
        doctor_map = dfs["person"].set_index("ID")["Nom+Prénom"].to_dict()
    
    # Direct links by patient ID
    direct_links = {
        "Ag_Rdv":       "Code Patient",
        "Consultation": "Code patient",
        "Documents":    "code patient",
        "tPostIT":      "CodePat",
    }
    for section, id_col in direct_links.items():
        df = dfs.get(section)
        if df is None:
            print(f"[SKIP] '{section}' not found in loaded files.")
            continue
        if id_col not in df.columns:
            print(f"[SKIP] Column '{id_col}' not found in '{section}' (available: {list(df.columns[:5])} …).")
            continue
        mask = df[id_col].apply(normalize_id) == patient_id
        filtered = df[mask].copy()
        if filtered.empty:
            print(f"[EMPTY] '{section}': no rows for patient ID {patient_id}.")
            continue
        
        # Add doctor name if doctor code column exists
        for doc_col in ("Code Docteur", "Code Médecin"):
            if doc_col in filtered.columns:
                filtered["Doctor_Name"] = filtered[doc_col].map(doctor_map)
                break
        cleaned = clean_df(filtered)
        if cleaned is not None:
            record[section] = cleaned
            print(f"[OK] '{section}': {len(cleaned)} row(s) recovered.")
        else:
            print(f"[EMPTY] '{section}': data was all-NaN after cleaning.")
    
    # Indirect links via consultation IDs (for tKERATO, tREFRACTION)
    if "Consultation" not in record:
        print("[WARN] No 'Consultation' section — cannot resolve tKERATO / tREFRACTION.")
        return record
    consult_ids = (
        record["Consultation"]["N° consultation"]
        .apply(normalize_id)
        .dropna()
        .tolist()
    )
    print(f"[INFO] {len(consult_ids)} consultation ID(s) to resolve for tKERATO / tREFRACTION.")
    indirect_links = {
        "tKERATO":     "NumConsult",
        "tREFRACTION": "NumConsult",
    }
    for section, id_col in indirect_links.items():
        df = dfs.get(section)
        if df is None:
            print(f"[SKIP] '{section}' not found in loaded files.")
            continue
        mask = df[id_col].apply(normalize_id).isin(consult_ids)
        filtered = df[mask].copy()
        cleaned = clean_df(filtered)
        if cleaned is not None:
            record[section] = cleaned
            print(f"[OK] '{section}': {len(cleaned)} row(s) recovered.")
        else:
            print(f"[EMPTY] '{section}': no matching rows for this patient's consultations.")
    return record