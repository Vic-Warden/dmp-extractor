"""
medical_summary.py — Cockpit Dashboard v13
==========================================
Ophthalmology patient follow-up dashboard for Streamlit.

Public API:
  render_medical_summary(record)   -> Streamlit render
  analyse_patient(record)          -> dict
  generate_medical_summary(record) -> str (Markdown)
  analyse_parcours_soin(record)    -> dict (alias)

Expected record keys / columns:
  Consultation : N° consultation, Code patient, Date, DOMINANTE,
                 Observation, Ordonnance, AutresPrescriptions, ProchainRDV,
                 TOD, TOG, Doctor_Name
  tREFRACTION  : NumConsult, SphD, SphG, CylD, CylG, AxeD, AxeG, AVDL, AVGL, AVDP, AVGP
  tKERATO      : NumConsult, R1D, R2D, AXE1D, AXE2D, R1G, R2G
  Patients     : Antécédants, Allergies, Traitements, Diagnostic OPH,
                 Important, Téléphone, Adressé par :, DateCreation
  Documents    : DESCRIPTIONS, TEXTE, Date
"""

import re
import pandas as pd
import streamlit as st
from datetime import datetime

try:
    import plotly.graph_objects as go
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False


# Utilities
def _safe_df(record: dict, key: str):
    df = record.get(key)
    return df if (df is not None and not df.empty) else None


def _parse_dates(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, errors="coerce", format="mixed", dayfirst=True)
    except (TypeError, ValueError):
        return pd.to_datetime(series, errors="coerce", dayfirst=True)


def _fmt_date(dt, fmt: str = "%d/%m/%Y") -> str:
    try:
        return pd.Timestamp(dt).strftime(fmt)
    except Exception:
        return "—"


def _val(v, fallback: str = "—") -> str:
    if v is None:
        return fallback
    try:
        if pd.isna(v):
            return fallback
    except (TypeError, ValueError):
        pass
    s = str(v).strip()
    return s if s else fallback


def _is_null_val(v) -> bool:
    if v is None:
        return True
    try:
        return bool(pd.isnull(v))
    except (TypeError, ValueError):
        return False


def _last_consult_date(record: dict) -> str:
    df = _safe_df(record, "Consultation")
    if df is None or "Date" not in df.columns:
        return "—"
    dates = _parse_dates(df["Date"]).dropna()
    return _fmt_date(dates.max()) if not dates.empty else "—"


def _n_consult(record: dict) -> int:
    df = _safe_df(record, "Consultation")
    return len(df) if df is not None else 0


def _sort_consult_desc(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    if "Date" in tmp.columns:
        tmp["_dt"] = _parse_dates(tmp["Date"])
        tmp = tmp.sort_values("_dt", ascending=False)
    return tmp


def _str_id(v) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    try:
        return str(int(float(str(v).strip())))
    except Exception:
        return str(v).strip()


# TEMPORAL PARSING
_RE_DATE_FULL  = re.compile(r'\b(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})\b')
_RE_DATE_MONTH = re.compile(r'\b(\d{1,2}[/\-]\d{2,4})\b')
_RE_YEAR_PAREN = re.compile(r'\((\d{2,4})\)')
_RE_YEAR_BARE  = re.compile(r'\b((?:19|20)\d{2})\b')
_RE_YEAR_SHORT = re.compile(r"(?<!\d)'(\d{2})(?!\d)")

_CURRENT_YEAR = datetime.now().year


def _normalise_year(raw: str) -> str:
    raw = raw.strip()
    if re.fullmatch(r'\d{2}', raw):
        y = int(raw)
        return str(2000 + y) if y < 40 else str(1900 + y)
    if re.fullmatch(r'(?:19|20)\d{2}', raw):
        return raw
    return raw


def _extract_inline_date(text: str) -> str | None:
    if not text or not isinstance(text, str):
        return None
    m = _RE_DATE_FULL.search(text)
    if m:
        raw = m.group(1)
        try:
            dt = pd.to_datetime(raw, dayfirst=True, errors="raise")
            return dt.strftime("%m/%Y")
        except Exception:
            pass
    m = _RE_DATE_MONTH.search(text)
    if m:
        raw = m.group(1)
        parts = re.split(r'[/\-]', raw)
        if len(parts) == 2:
            month_part, year_part = parts
            year_str = _normalise_year(year_part)
            if re.fullmatch(r'(?:19|20)\d{2}', year_str):
                return f"{month_part.zfill(2)}/{year_str}"
    m = _RE_YEAR_PAREN.search(text)
    if m:
        return _normalise_year(m.group(1))
    m = _RE_YEAR_BARE.search(text)
    if m:
        return m.group(1)
    m = _RE_YEAR_SHORT.search(text)
    if m:
        return _normalise_year(m.group(1))
    return None


def _get_date_creation(record: dict) -> str:
    id_df = _safe_df(record, "identity")
    if id_df is None:
        return ""
    row = id_df.iloc[0]
    for col in ("DateCreation", "Date création", "Date creation",
                "date_creation", "DATE_CREATION"):
        raw = row.get(col)
        if raw is not None and not _is_null_val(raw):
            try:
                dt = pd.to_datetime(raw, dayfirst=True, errors="raise")
                return dt.strftime("%d/%m/%Y")
            except Exception:
                v = _val(raw, "")
                if v and v != "—":
                    return v
    return ""


def _items_with_dates(
    raw_text: str,
    fallback_date: str,
    max_items: int = 8,
) -> list[dict]:
    if not raw_text or raw_text == "—":
        return []
    parts = [
        x.strip()
        for x in re.split(r"[,;\n/]", raw_text)
        if x.strip() and len(x.strip()) > 1
    ][:max_items]
    result = []
    for part in parts:
        inline = _extract_inline_date(part)
        label = part
        if inline:
            label = _RE_DATE_FULL.sub("", label)
            label = _RE_DATE_MONTH.sub("", label)
            label = _RE_YEAR_PAREN.sub("", label)
            label = _RE_YEAR_BARE.sub("", label)
            label = _RE_YEAR_SHORT.sub("", label)
            label = re.sub(r'\s+', ' ', label).strip(" ,-;()")
        if not label:
            label = part
        result.append({
            "label": label,
            "date":  inline or fallback_date,
        })
    return result


# DATA EXTRACTION — ACUITÉ VISUELLE (v13 new KPI #1)
# Column candidates for AV (distance / near, with / without correction)
_AV_CANDIDATES = {
    "avdl_sc":  ["AVscOD", "AV sc OD", "AVSCOD", "AVD sc", "AVDsc"],
    "avdl_cc":  ["AVccOD", "AV cc OD", "AVCCOD", "AVD cc", "AVDcc"],
    "avgl_sc":  ["AVscOG", "AV sc OG", "AVSCOG", "AVG sc", "AVGsc"],
    "avgl_cc":  ["AVccOG", "AV cc OG", "AVCCOG", "AVG cc", "AVGcc"],
    # tREFRACTION columns — AVDL = loin OD, AVGL = loin OG (sans correction)
    "refrac_avdl": ["AVDL"],
    "refrac_avgl": ["AVGL"],
    "refrac_avdp": ["AVDP"],
    "refrac_avgp": ["AVGP"],
}


def _find_col(df: pd.DataFrame, candidates: list) -> str | None:
    return next((c for c in candidates if c in df.columns), None)


def _extract_visual_acuity(record: dict) -> dict:
    """
    Extract the most recent AV values (OD + OG, distance + near).
    Sources: Consultation columns first, then tREFRACTION as fallback.
    Returns dict: {od_sc, od_cc, og_sc, og_cc, date, source}
    """
    result = {"od_sc": "", "od_cc": "", "og_sc": "", "og_cc": "",
              "date": "", "source": ""}

    # 1. Try Consultation columns
    df_c = _safe_df(record, "Consultation")
    if df_c is not None:
        tmp = _sort_consult_desc(df_c)
        col_avdl_sc = _find_col(tmp, _AV_CANDIDATES["avdl_sc"])
        col_avdl_cc = _find_col(tmp, _AV_CANDIDATES["avdl_cc"])
        col_avgl_sc = _find_col(tmp, _AV_CANDIDATES["avgl_sc"])
        col_avgl_cc = _find_col(tmp, _AV_CANDIDATES["avgl_cc"])

        for _, row in tmp.iterrows():
            od_sc = _val(row.get(col_avdl_sc), "") if col_avdl_sc else ""
            od_cc = _val(row.get(col_avdl_cc), "") if col_avdl_cc else ""
            og_sc = _val(row.get(col_avgl_sc), "") if col_avgl_sc else ""
            og_cc = _val(row.get(col_avgl_cc), "") if col_avgl_cc else ""
            if any(v and v != "—" for v in (od_sc, od_cc, og_sc, og_cc)):
                dt = _fmt_date(row.get("_dt") or row.get("Date"), "%d/%m/%Y")
                result.update({
                    "od_sc": od_sc if od_sc != "—" else "",
                    "od_cc": od_cc if od_cc != "—" else "",
                    "og_sc": og_sc if og_sc != "—" else "",
                    "og_cc": og_cc if og_cc != "—" else "",
                    "date": dt if dt != "—" else "",
                    "source": "Consultation",
                })
                return result

    # 2. Fallback: tREFRACTION (most recent via linked consultation date)
    df_r = _safe_df(record, "tREFRACTION")
    df_consult = _safe_df(record, "Consultation")
    if df_r is not None:
        # Build date lookup
        date_map: dict[str, pd.Timestamp] = {}
        if df_consult is not None and "N° consultation" in df_consult.columns:
            for _, row in df_consult.iterrows():
                nc = _str_id(row.get("N° consultation"))
                dt = _parse_dates(pd.Series([row.get("Date")])).iloc[0]
                if nc and not _is_null_val(dt):
                    date_map[nc] = dt

        best_dt = None
        best_row = None
        for _, row in df_r.iterrows():
            nc = _str_id(row.get("NumConsult"))
            dt = date_map.get(nc)
            if dt is not None and (best_dt is None or dt > best_dt):
                best_dt = dt
                best_row = row

        if best_row is not None:
            col_avdl = _find_col(df_r, _AV_CANDIDATES["refrac_avdl"])
            col_avgl = _find_col(df_r, _AV_CANDIDATES["refrac_avgl"])
            avdl = _val(best_row.get(col_avdl), "") if col_avdl else ""
            avgl = _val(best_row.get(col_avgl), "") if col_avgl else ""
            if avdl != "—" or avgl != "—":
                result.update({
                    "od_sc": avdl if avdl != "—" else "",
                    "og_sc": avgl if avgl != "—" else "",
                    "date": _fmt_date(best_dt) if best_dt else "",
                    "source": "Réfraction",
                })

    return result


# DATA EXTRACTION — PIO (v13: dedicated block)
def _extract_pio(record: dict) -> dict:
    """
    Return the most recent PIO values.
    Returns dict: {od, og, date, alert} where alert is True if any value > 21.
    """
    result = {"od": "", "og": "", "date": "", "alert": False}
    df = _safe_df(record, "Consultation")
    if df is None or "TOD" not in df.columns:
        return result
    tmp = _sort_consult_desc(df)
    for _, row in tmp.iterrows():
        od = _val(row.get("TOD"), "")
        og = _val(row.get("TOG"), "") if "TOG" in tmp.columns else ""
        if (od and od != "—") or (og and og != "—"):
            dt = _fmt_date(row.get("_dt") or row.get("Date"), "%d/%m/%Y")
            result["od"] = od if od != "—" else ""
            result["og"] = og if og != "—" else ""
            result["date"] = dt if dt != "—" else ""
            # Check hypertony
            for v_str in (od, og):
                try:
                    if float(v_str.replace(",", ".")) > 21:
                        result["alert"] = True
                except (ValueError, AttributeError):
                    pass
            return result
    return result


# kept for backwards compat (generate_medical_summary still calls it)
def _extract_pio_alert(record: dict) -> str:
    pio = _extract_pio(record)
    if not pio["alert"]:
        return ""
    parts = []
    if pio["od"]:
        parts.append(f"PIO OD : {pio['od']} mmHg")
    if pio["og"]:
        parts.append(f"PIO OG : {pio['og']} mmHg")
    return " · ".join(parts) + " — surveillance renforcée recommandée" if parts else ""


def _extract_pio_history(record: dict) -> pd.DataFrame:
    """
    Extract the full chronological history of PIO measurements (TOD / TOG).

    Returns a DataFrame with columns:
        date  (datetime)  — consultation date (NaT rows are dropped)
        od    (float)     — PIO Œil Droit in mmHg  (NaN if missing / unparseable)
        og    (float)     — PIO Œil Gauche in mmHg (NaN if missing / unparseable)

    The DataFrame is sorted ascending by date so the chart reads left → right.
    Returns an empty DataFrame if no usable data is found.
    """
    df = _safe_df(record, "Consultation")
    if df is None:
        return pd.DataFrame(columns=["date", "od", "og"])

    has_tod = "TOD" in df.columns
    has_tog = "TOG" in df.columns
    if not has_tod and not has_tog:
        return pd.DataFrame(columns=["date", "od", "og"])

    tmp = df.copy()
    tmp["_dt"] = _parse_dates(tmp["Date"] if "Date" in tmp.columns else pd.Series(dtype=str))

    def _to_float(series: pd.Series) -> pd.Series:
        return (
            series.astype(str)
            .str.replace(",", ".", regex=False)
            .str.strip()
            .replace({"": float("nan"), "—": float("nan"), "nan": float("nan"),
                      "None": float("nan"), "NaN": float("nan")})
            .pipe(pd.to_numeric, errors="coerce")
        )

    rows = pd.DataFrame({
        "date": tmp["_dt"],
        "od":   _to_float(tmp["TOD"]) if has_tod else float("nan"),
        "og":   _to_float(tmp["TOG"]) if has_tog else float("nan"),
    })

    # Keep only rows where the date is known AND at least one eye has a value
    rows = rows.dropna(subset=["date"])
    rows = rows[rows[["od", "og"]].notna().any(axis=1)]
    rows = rows.sort_values("date").reset_index(drop=True)
    return rows


# DATA EXTRACTION — 360° CARD (other blocks unchanged from v12)
def _col_lookup(row: pd.Series, candidates: list) -> str:
    import unicodedata

    def _norm(s: str) -> str:
        s = unicodedata.normalize("NFC", s).strip().lower()
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"\s*:\s*", ":", s)
        return s

    normed_row = {_norm(str(k)): v for k, v in row.items()}
    for candidate in candidates:
        v = _val(row.get(candidate), "")
        if v and v != "—":
            return v
        v2 = normed_row.get(_norm(candidate), None)
        if v2 is not None:
            v2 = _val(v2, "")
            if v2 and v2 != "—":
                return v2
    return ""


def _extract_important(record: dict) -> tuple[str, list[dict]]:
    id_df = _safe_df(record, "identity")
    if id_df is None:
        return ("", [])
    row = id_df.iloc[0]
    raw = _col_lookup(row, ["Important", "IMPORTANT", "important",
                             "Notes importantes", "Note importante", "Note"])
    if not raw:
        return ("", [])
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    fallback = _get_date_creation(record)
    items = _items_with_dates(text, fallback, max_items=12)
    return (text, items)


def _extract_contact_info(record: dict) -> dict:
    id_df = _safe_df(record, "identity")
    result = {"telephone": "", "adresse_par": ""}
    if id_df is None:
        return result
    row = id_df.iloc[0]
    tel = _col_lookup(row, ["Téléphone", "Telephone", "TELEPHONE",
                             "Téléphone bureau", "Tel", "Mobile"])
    if tel:
        result["telephone"] = tel
    ref = _col_lookup(row, ["Adressé par :", "Adressé par:", "Adressé par",
                             "Adresse par :", "Adresse par", "Référent",
                             "Referent", "Médecin référent"])
    if ref:
        result["adresse_par"] = ref
    return result


def _extract_antecedents(record: dict) -> dict:
    id_df = _safe_df(record, "identity")
    result = {"antecedents": [], "allergies": []}
    if id_df is None:
        return result
    row = id_df.iloc[0]
    fallback = _get_date_creation(record)
    for col in ("Antécédants", "Antecedants", "Antécédents", "ANTECEDANTS"):
        raw = _val(row.get(col), "")
        if raw and raw != "—":
            result["antecedents"] = _items_with_dates(raw, fallback, max_items=8)
            break
    raw = _val(row.get("Allergies"), "")
    if raw and raw != "—":
        result["allergies"] = _items_with_dates(raw, fallback, max_items=5)
    return result


def _extract_traitements(record: dict) -> list:
    """Plain-string list (backwards compat)."""
    id_df = _safe_df(record, "identity")
    if id_df is None:
        return []
    raw = _val(id_df.iloc[0].get("Traitements"), "")
    if raw and raw != "—":
        return [
            x.strip()
            for x in re.split(r"[,;\n/]", raw)
            if x.strip() and len(x.strip()) > 1
        ][:8]
    return []


# Heuristic keywords to classify a treatment as "local" (collyres/eye drops)
_COLLYRE_KEYWORDS = re.compile(
    r'\b(collyre|goutte|gel ophtalmique|timolol|latanoprost|dorzolamide|'
    r'brimonidine|bimatoprost|travoprost|tafluprost|brinzolamide|betaxolol|'
    r'carteolol|azetazolamide|acetazolamide|pilocarpin|tropicamide|'
    r'dexaméthasone|prednisolone|tobramycin|ciprofloxacin|ofloxacin|'
    r'chloramphénicol|fluorescéine|hypromellose|larme|lubrifiant)\b',
    re.IGNORECASE,
)


def _classify_traitement(label: str) -> str:
    """Return 'local' (collyre) or 'systemic'."""
    return "local" if _COLLYRE_KEYWORDS.search(label) else "systemic"


def _extract_traitements_history(record: dict) -> list[dict]:
    """
    Historised treatment items with dates and local/systemic classification.
    Returns list of {label, date, type: 'local'|'systemic'}.
    """
    id_df = _safe_df(record, "identity")
    if id_df is None:
        return []
    raw = _val(id_df.iloc[0].get("Traitements"), "")
    if not raw or raw == "—":
        return []
    fallback = _get_date_creation(record)
    items = _items_with_dates(raw, fallback, max_items=10)

    def _sort_key(item):
        d = item.get("date", "")
        for fmt in ("%d/%m/%Y", "%m/%Y", "%Y"):
            try:
                return pd.Timestamp(datetime.strptime(d, fmt))
            except (ValueError, TypeError):
                pass
        return pd.Timestamp.min

    items_sorted = sorted(items, key=_sort_key, reverse=True)
    for item in items_sorted:
        item["type"] = _classify_traitement(item["label"])
    return items_sorted


def _extract_prescriptions_history(record: dict) -> list[dict]:
    """All consultations with prescriptions, newest first."""
    df = _safe_df(record, "Consultation")
    if df is None:
        return []
    tmp = _sort_consult_desc(df)
    result = []
    for _, row in tmp.iterrows():
        ord_v = _val(row.get("Ordonnance"), "") if "Ordonnance" in tmp.columns else ""
        aut_v = _val(row.get("AutresPrescriptions"), "") if "AutresPrescriptions" in tmp.columns else ""
        if (ord_v and ord_v != "—") or (aut_v and aut_v != "—"):
            dt = _fmt_date(row.get("_dt") or row.get("Date"), "%d/%m/%Y")
            result.append({
                "date":       dt if dt != "—" else "",
                "ordonnance": ord_v if ord_v != "—" else "",
                "autres":     aut_v if aut_v != "—" else "",
            })
    return result


def _extract_prescriptions(record: dict) -> dict:
    """Most recent prescription (backwards compat)."""
    df = _safe_df(record, "Consultation")
    if df is None:
        return {"ordonnance": "", "autres": "", "date": ""}
    tmp = _sort_consult_desc(df)
    for _, row in tmp.iterrows():
        ord_v = _val(row.get("Ordonnance"), "") if "Ordonnance" in tmp.columns else ""
        aut_v = _val(row.get("AutresPrescriptions"), "") if "AutresPrescriptions" in tmp.columns else ""
        if (ord_v and ord_v != "—") or (aut_v and aut_v != "—"):
            dt = _fmt_date(row.get("_dt") or row.get("Date"), "%d/%m/%Y")
            return {
                "ordonnance": ord_v if ord_v != "—" else "",
                "autres":     aut_v if aut_v != "—" else "",
                "date":       dt if dt != "—" else "",
            }
    return {"ordonnance": "", "autres": "", "date": ""}


def _extract_diagnostic(record: dict) -> tuple[str, str]:
    id_df = _safe_df(record, "identity")
    diag_text = ""
    if id_df is not None:
        row = id_df.iloc[0]
        for col in ("Diagnostic OPH", "Diagnostic  OPH", "DiagnosticOPH",
                    "Diagnostic Oph", "Diagnostic"):
            v = _val(row.get(col), "")
            if v and v != "—":
                diag_text = v[:250]
                break
    date_str = _last_consult_date(record)
    return (diag_text, date_str if date_str != "—" else "")


def _extract_plan_suivi(record: dict) -> str:
    df = _safe_df(record, "Consultation")
    if df is None or "ProchainRDV" not in df.columns:
        return ""
    tmp = _sort_consult_desc(df)
    for _, row in tmp.iterrows():
        v = _val(row.get("ProchainRDV"), "")
        if v and v != "—":
            return v[:180]
    return ""


def _extract_motif(record: dict) -> tuple[str, str]:
    """kept for generate_medical_summary backwards compat."""
    df = _safe_df(record, "Consultation")
    if df is None:
        return ("Bilan ophtalmologique général", "")
    tmp = _sort_consult_desc(df)
    if "DOMINANTE" in tmp.columns:
        for _, row in tmp.iterrows():
            v = _val(row.get("DOMINANTE"), "")
            if v and v != "—" and len(v) > 2:
                dt = _fmt_date(row.get("_dt") or row.get("Date"), "%d/%m/%Y")
                return (v[:150], dt if dt != "—" else "")
    return ("Bilan ophtalmologique général", "")


# TABLEAU DES ACTES — EXAM LABEL NORMALISATION
_EXAM_LABEL_MAP: list[tuple[str, list[str], str]] = [
    ("OCT",              ["oct"],                                "img"),
    ("Angiographie",     ["angio", "ffa"],                      "img"),
    ("Rétinographie",    ["rétino", "retino", "fond d",
                          "rétinographie", "retinographie"],    "img"),
    ("Imagerie",         ["imagenet", "imagerie"],              "img"),
    ("Lampe à fente",    ["laf", "lampe à fente",
                          "lampe a fente", "lamp fente"],       "exam"),
    ("Champ visuel",     ["champ visuel", "périmét",
                          "perimetrie", "périmètre"],           "exam"),
    ("Biométrie",        ["biométrie", "biometrie",
                          "iolmaster", "iol master"],           "exam"),
    ("Pachymétrie",      ["pachy"],                             "exam"),
    ("Topo cornéenne",   ["topograph", "topo cor"],             "exam"),
    ("Kératométrie",     ["kérato", "kerato"],                  "exam"),
    ("Laser",            ["laser"],                             "proc"),
    ("Injection IVT",    ["ivt", "injection intra",
                          "injection ivt"],                     "proc"),
]

_BADGE_CSS: dict[str, str] = {
    "img":  "ck-chip ck-chip-img",
    "exam": "ck-chip ck-chip-exam",
    "proc": "ck-chip ck-chip-proc",
}


def _normalize_exam_label(raw: str) -> tuple[str, str]:
    if not raw or raw == "—":
        return ("", "")
    clean = raw.strip().lower()
    clean = re.sub(r'\b(od|og|odg|od/og|oed)\b', '', clean).strip()
    for canonical, patterns, category in _EXAM_LABEL_MAP:
        for pat in patterns:
            if pat in clean:
                return (canonical, category)
    label = raw.strip()
    if len(label) > 36:
        label = label[:36] + "…"
    return (label.title(), "")


# TABLEAU DES ACTES — DATA BUILDER
def _build_actes_rows(record: dict) -> list[dict]:
    consult_df = _safe_df(record, "Consultation")
    ker_df     = _safe_df(record, "tKERATO")
    ref_df     = _safe_df(record, "tREFRACTION")
    docs_df    = _safe_df(record, "Documents")

    nc_has_kerato: set[str] = set()
    if ker_df is not None and "NumConsult" in ker_df.columns:
        nc_has_kerato = {_str_id(v) for v in ker_df["NumConsult"] if _str_id(v)}

    nc_has_refrac: set[str] = set()
    if ref_df is not None and "NumConsult" in ref_df.columns:
        nc_has_refrac = {_str_id(v) for v in ref_df["NumConsult"] if _str_id(v)}

    groups: dict[tuple, dict] = {}

    def _upsert(date_ts, date_str: str, doctor: str,
                motif: str, tech: list[tuple[str, str]]):
        key = (date_str, doctor)
        if key not in groups:
            groups[key] = {
                "date_ts":    date_ts,
                "date_str":   date_str,
                "motif":      motif,
                "tech_actes": [],
                "doctor":     doctor,
            }
        if not groups[key]["motif"] and motif:
            groups[key]["motif"] = motif
        seen_labels = {t[0] for t in groups[key]["tech_actes"]}
        for label, cat in tech:
            if label and label not in seen_labels:
                groups[key]["tech_actes"].append((label, cat))
                seen_labels.add(label)

    if consult_df is not None:
        tmp = _sort_consult_desc(consult_df)
        for _, row in tmp.iterrows():
            dt = row.get("_dt")
            if _is_null_val(dt):
                continue
            date_str = _fmt_date(dt)
            doctor   = _val(row.get("Doctor_Name"), "")
            if doctor == "—":
                doctor = ""
            motif = _val(row.get("DOMINANTE"), "")
            if motif == "—":
                motif = ""
            nc   = _str_id(row.get("N° consultation"))
            tech = []
            if nc and nc in nc_has_kerato:
                tech.append(("Kératométrie", "exam"))
            if nc and nc in nc_has_refrac:
                tech.append(("Réfraction", "exam"))
            _upsert(dt, date_str, doctor, motif, tech)

    if docs_df is not None and "Date" in docs_df.columns:
        for _, row in docs_df.iterrows():
            raw_dt = row.get("Date")
            dts    = _parse_dates(pd.Series([raw_dt]))
            dt     = dts.iloc[0] if not dts.empty else None
            if _is_null_val(dt):
                continue
            date_str = _fmt_date(dt)
            raw_desc = row.get("DESCRIPTIONS") or row.get("Type") or ""
            desc = _val(raw_desc, "")
            if not desc or desc == "—":
                continue
            label, cat = _normalize_exam_label(desc)
            if not label:
                continue
            matched_key = next((k for k in groups if k[0] == date_str), None)
            if matched_key:
                seen = {t[0] for t in groups[matched_key]["tech_actes"]}
                if label not in seen:
                    groups[matched_key]["tech_actes"].append((label, cat))
            else:
                _upsert(dt, date_str, "", "", [(label, cat)])

    return sorted(groups.values(), key=lambda x: x["date_ts"], reverse=True)


# CSS — v13
_CSS = """
<style>
/* Design tokens */
:root {
    --ck-navy:    #1B3A6B;
    --ck-teal:    #0D7A5F;
    --ck-amber:   #D97706;
    --ck-red:     #DC2626;
    --ck-muted:   #6B7280;
    --ck-border:  #E5E7EB;
    --ck-card:    #FFFFFF;
    --ck-soft:    #F8FAFC;
    --ck-text:    #111827;
    --ck-r:       12px;
    --ck-shadow:  0 4px 28px rgba(0,0,0,0.09);
    --ck-font:    'Segoe UI', system-ui, Arial, sans-serif;
}

/* Card shell */
.ck-card {
    background: var(--ck-card);
    border-radius: var(--ck-r);
    box-shadow: var(--ck-shadow);
    border: 1px solid var(--ck-border);
    overflow: hidden;
    margin-bottom: 1.2rem;
    font-family: var(--ck-font);
}

/* Ultra-thin header strip (v13) */
.ck-header-strip {
    display: flex;
    align-items: center;
    gap: 0;
    background: #1B3A6B;
    padding: 6px 18px;
    font-family: var(--ck-font);
    flex-wrap: wrap;
}
.ck-header-strip-item {
    display: flex;
    align-items: center;
    gap: 6px;
    color: rgba(255,255,255,0.92);
    font-size: 0.71rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    padding: 0 14px 0 0;
}
.ck-header-strip-item:not(:last-child)::after {
    content: '·';
    margin-left: 14px;
    opacity: 0.45;
}
.ck-header-strip-label {
    font-size: 0.60rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    opacity: 0.58;
    margin-right: 4px;
}

/* Card header (360° card, Actes card) */
.ck-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 20px;
    background: linear-gradient(135deg, #1B3A6B 0%, #234DA8 100%);
    color: #fff;
    flex-wrap: wrap;
    gap: 6px;
}
.ck-head-title {
    font-size: 0.69rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.12em;
}
.ck-head-meta {
    font-size: 0.67rem;
    opacity: 0.72;
    letter-spacing: 0.04em;
}
.ck-head-contact {
    font-size: 0.65rem;
    opacity: 0.65;
    letter-spacing: 0.03em;
    margin-top: 2px;
    width: 100%;
}
.ck-head-dossier {
    font-size: 0.62rem;
    opacity: 0.58;
    letter-spacing: 0.03em;
    margin-top: 1px;
    width: 100%;
    font-style: italic;
}

/* Important alert banner */
.ck-important-banner {
    background: #FFF7ED;
    border-left: 4px solid #F59E0B;
    border-bottom: 1px solid #FDE68A;
    padding: 12px 22px 12px 18px;
    font-family: var(--ck-font);
}
.ck-important-banner-title {
    font-size: 0.63rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.10em;
    color: #92400E;
    margin-bottom: 5px;
    display: flex;
    align-items: center;
    gap: 6px;
}
.ck-important-banner-body {
    font-size: 0.82rem;
    color: #78350F;
    line-height: 1.65;
    white-space: pre-wrap;
    word-break: break-word;
}

/* Dynamic grid (v13) */
/* Uses flex instead of fixed 3-col grid so empty cells are truly absent. */
.ck-row {
    display: flex;
    border-bottom: 1px solid var(--ck-border);
}
.ck-row:last-of-type {
    border-bottom: none;
}
.ck-cell {
    flex: 1 1 0;
    padding: 16px 20px;
    border-right: 1px solid var(--ck-border);
    min-height: 96px;
    min-width: 0;
}
.ck-cell:last-child { border-right: none; }
.ck-cell-teal  { background: linear-gradient(180deg,#F0FDF4 0%,#FAFAFA 100%); }
.ck-cell-blue  { background: linear-gradient(180deg,#EFF6FF 0%,#FAFAFA 100%); }
.ck-cell-rdv   { background: linear-gradient(180deg,#F5F3FF 0%,#FAFAFA 100%); }
.ck-cell-pio   { background: linear-gradient(180deg,#FFF7ED 0%,#FAFAFA 100%); }
.ck-cell-av    { background: linear-gradient(180deg,#F0F9FF 0%,#FAFAFA 100%); }

/* Section label */
.ck-lbl {
    font-size: 0.58rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.10em;
    color: var(--ck-muted);
    margin-bottom: 9px;
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 2px;
}
.ck-lbl-date {
    font-size: 0.56rem;
    font-weight: 500;
    text-transform: none;
    letter-spacing: 0.02em;
    color: var(--ck-muted);
    font-style: italic;
    margin-left: 4px;
    padding-left: 6px;
    border-left: 1px solid #CBD5E1;
    opacity: 0.80;
}

/* Values */
.ck-val-main {
    font-size: 0.98rem;
    font-weight: 700;
    color: var(--ck-navy);
    line-height: 1.45;
}
.ck-val {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--ck-text);
    line-height: 1.55;
}
.ck-val-muted {
    font-size: 0.79rem;
    color: var(--ck-muted);
    font-style: italic;
    font-weight: 400;
}

/* AV KPI table (v13) */
.ck-av-grid {
    display: grid;
    grid-template-columns: auto 1fr 1fr;
    gap: 4px 10px;
    font-family: var(--ck-font);
    font-size: 0.78rem;
    align-items: center;
    margin-top: 2px;
}
.ck-av-eye-label {
    font-size: 0.60rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--ck-muted);
    padding: 0 6px;
    border-right: 2px solid var(--ck-border);
    text-align: right;
    white-space: nowrap;
}
.ck-av-val {
    font-size: 0.82rem;
    font-weight: 700;
    color: var(--ck-navy);
}
.ck-av-sub {
    font-size: 0.62rem;
    color: var(--ck-muted);
    margin-top: 1px;
    white-space: nowrap;
}
.ck-av-header {
    font-size: 0.58rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--ck-muted);
    text-align: center;
}

/* PIO KPI (v13) */
.ck-pio-grid {
    display: flex;
    gap: 12px;
    margin-top: 4px;
    flex-wrap: wrap;
}
.ck-pio-eye {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    background: #FFF;
    border: 1px solid #FED7AA;
    border-radius: 8px;
    padding: 8px 14px;
    min-width: 68px;
}
.ck-pio-eye.alert {
    border-color: #FCA5A5;
    background: #FEF2F2;
}
.ck-pio-label {
    font-size: 0.58rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: var(--ck-muted);
    margin-bottom: 3px;
}
.ck-pio-val {
    font-size: 1.05rem;
    font-weight: 800;
    color: #92400E;
    line-height: 1.2;
}
.ck-pio-val.alert {
    color: var(--ck-red);
}
.ck-pio-unit {
    font-size: 0.62rem;
    color: var(--ck-muted);
    margin-top: 1px;
}

/* Tag chips */
.ck-tags { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 2px; }
.ck-tag {
    font-size: 0.69rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    border: 1px solid transparent;
    line-height: 1.45;
    display: inline-flex;
    align-items: center;
    gap: 0;
}
.t-navy  { background: #EFF6FF; color: #1E3A8A; border-color: #BFDBFE; }
.t-teal  { background: #ECFDF5; color: #065F46; border-color: #6EE7B7; }
.t-red   { background: #FEF2F2; color: #991B1B; border-color: #FECACA; }
.t-amber { background: #FFFBEB; color: #92400E; border-color: #FCD34D; }
.t-grey  { background: #F9FAFB; color: #374151; border-color: #D1D5DB; }
.ck-tag-date {
    font-size: 0.60rem;
    font-weight: 400;
    opacity: 0.62;
    margin-left: 6px;
    padding-left: 6px;
    border-left: 1px solid currentColor;
    white-space: nowrap;
    letter-spacing: 0.01em;
}

/* Allergy critical banner (v13) */
.ck-allergy-warn {
    display: flex;
    align-items: center;
    gap: 6px;
    background: #FEF2F2;
    border: 1px solid #FECACA;
    border-radius: 6px;
    padding: 5px 10px;
    margin-bottom: 6px;
    font-size: 0.69rem;
    font-weight: 700;
    color: #991B1B;
}

/* Treatment type divider (v13) */
.ck-trt-section {
    font-size: 0.57rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.10em;
    color: var(--ck-muted);
    margin: 8px 0 4px 0;
    padding-bottom: 2px;
    border-bottom: 1px solid var(--ck-border);
}
.ck-trt-section:first-child { margin-top: 0; }

/* Prochain RDV highlight box */
.ck-rdv-box {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    background: #EDE9FE;
    border-radius: 8px;
    padding: 11px 15px;
    border: 1px solid #C4B5FD;
}
.ck-rdv-icon { font-size: 1.15rem; flex-shrink: 0; margin-top: 1px; }
.ck-rdv-text {
    font-size: 0.88rem;
    font-weight: 700;
    color: #4C1D95;
    line-height: 1.45;
}
.ck-rdv-none {
    font-size: 0.79rem;
    color: var(--ck-muted);
    font-style: italic;
}

/* Section number badge */
.ck-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: var(--ck-navy);
    color: #fff;
    font-size: 0.58rem;
    font-weight: 900;
    flex-shrink: 0;
    margin-right: 5px;
}

/* TABLEAU DES ACTES */
.ck-actes-wrap {
    overflow-y: auto;
    max-height: 354px;
    border-radius: 0 0 var(--ck-r) var(--ck-r);
}
.ck-actes-table {
    width: 100%;
    border-collapse: collapse;
    font-family: var(--ck-font);
    table-layout: fixed;
}
.ck-actes-table thead th {
    position: sticky;
    top: 0;
    z-index: 2;
    background: var(--ck-soft);
    font-size: 0.59rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.10em;
    color: var(--ck-muted);
    padding: 10px 20px;
    border-bottom: 2px solid var(--ck-border);
    text-align: left;
    white-space: nowrap;
}
.ck-actes-th-date   { width: 98px; }
.ck-actes-th-doctor { width: 160px; }
.ck-actes-table tbody td {
    padding: 11px 20px;
    border-bottom: 1px solid #F3F4F6;
    vertical-align: top;
}
.ck-actes-table tbody tr:last-child td { border-bottom: none; }
.ck-actes-table tbody tr:hover td {
    background: #F9FBFF;
    transition: background 0.12s;
}
.ck-act-date {
    font-size: 0.82rem;
    font-weight: 700;
    color: var(--ck-navy);
    white-space: nowrap;
    padding-top: 13px !important;
}
.ck-act-motif {
    font-size: 0.84rem;
    font-weight: 600;
    color: var(--ck-text);
    margin-bottom: 6px;
    line-height: 1.38;
}
.ck-act-motif-empty {
    font-size: 0.79rem;
    color: var(--ck-muted);
    font-style: italic;
    margin-bottom: 4px;
}
.ck-act-chips { display: flex; flex-wrap: wrap; gap: 4px; }
.ck-chip {
    font-size: 0.65rem;
    font-weight: 700;
    padding: 2px 9px;
    border-radius: 4px;
    white-space: nowrap;
    line-height: 1.55;
    border: 1px solid transparent;
    letter-spacing: 0.02em;
}
.ck-chip-img  { background: #F5F3FF; color: #4C1D95; border-color: #C4B5FD; }
.ck-chip-exam { background: #ECFDF5; color: #065F46; border-color: #6EE7B7; }
.ck-chip-proc { background: #FFFBEB; color: #92400E; border-color: #FCD34D; }
.ck-chip-grey { background: #F9FAFB; color: #374151; border-color: #D1D5DB; }
.ck-act-doctor {
    font-size: 0.78rem;
    color: var(--ck-muted);
    font-weight: 500;
    padding-top: 13px !important;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.ck-actes-empty {
    padding: 24px 22px;
    font-size: 0.82rem;
    color: var(--ck-muted);
    font-style: italic;
    text-align: center;
}

/* HISTORISED BLOCKS */
.ck-hist {
    max-height: 400px;
    overflow-y: auto;
    padding-right: 2px;
    margin-top: 2px;
}
.ck-hist-row {
    display: flex;
    gap: 10px;
    align-items: baseline;
    padding: 4px 0;
    border-bottom: 1px dashed #EAECF0;
    font-family: var(--ck-font);
}
.ck-hist-row:last-child { border-bottom: none; }
.ck-hist-dt {
    font-size: 0.63rem;
    font-weight: 700;
    color: var(--ck-muted);
    white-space: nowrap;
    min-width: 50px;
    letter-spacing: 0.01em;
    flex-shrink: 0;
}
.ck-hist-txt {
    font-size: 0.81rem;
    color: var(--ck-text);
    line-height: 1.45;
    word-break: break-word;
}
.ck-hist-txt-muted {
    font-size: 0.75rem;
    color: var(--ck-muted);
    margin-top: 2px;
    line-height: 1.4;
    font-style: italic;
}
</style>
"""


# HTML HELPERS
def _escape(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _ck_lbl(num: int, text: str, date_str: str = "") -> str:
    date_mention = (
        f'<span class="ck-lbl-date">En date du {_escape(date_str)}</span>'
        if date_str else ""
    )
    return (
        f'<div class="ck-lbl">'
        f'<span class="ck-num">{num}</span>{text}{date_mention}'
        f'</div>'
    )


def _ck_tags(items, cls: str) -> str:
    if not items:
        return '<span class="ck-val-muted">Aucun renseigné</span>'
    chips = []
    for item in items:
        if isinstance(item, dict):
            label    = _escape(item.get("label", ""))
            date_val = item.get("date", "")
            badge    = (
                f'<span class="ck-tag-date">{_escape(date_val)}</span>'
                if date_val else ""
            )
            chips.append(f'<span class="ck-tag {cls}">{label}{badge}</span>')
        else:
            chips.append(f'<span class="ck-tag {cls}">{_escape(str(item))}</span>')
    return '<div class="ck-tags">' + "".join(chips) + '</div>'


def _ck_mixed_tags(antecedents: list, allergies: list) -> str:
    """Antecedents (navy) + allergies (red) with date micro-badges."""
    has_ant = bool(antecedents)
    has_alg = bool(allergies)
    if not has_ant and not has_alg:
        return '<span class="ck-val-muted">Aucun renseigné</span>'
    html = ""
    # Critical allergy banner if any allergy present
    if has_alg:
        alg_labels = []
        for item in allergies:
            l = item.get("label", "") if isinstance(item, dict) else str(item)
            alg_labels.append(_escape(l))
        html += (
            '<div class="ck-allergy-warn">'
            '⚠ Allergie : '
            + " / ".join(alg_labels)
            + '</div>'
        )
    html += '<div class="ck-tags">'
    for item in antecedents:
        if isinstance(item, dict):
            label    = _escape(item.get("label", ""))
            date_val = item.get("date", "")
            badge    = f'<span class="ck-tag-date">{_escape(date_val)}</span>' if date_val else ""
            html += f'<span class="ck-tag t-navy">{label}{badge}</span>'
        else:
            html += f'<span class="ck-tag t-navy">{_escape(str(item))}</span>'
    for item in allergies:
        if isinstance(item, dict):
            label    = _escape(item.get("label", ""))
            date_val = item.get("date", "")
            badge    = f'<span class="ck-tag-date">{_escape(date_val)}</span>' if date_val else ""
            html += f'<span class="ck-tag t-red">⚠ {label}{badge}</span>'
        else:
            html += f'<span class="ck-tag t-red">⚠ {_escape(str(item))}</span>'
    html += '</div>'
    return html


def _ck_rdv(plan: str) -> str:
    if not plan:
        return '<span class="ck-rdv-none">Non planifié</span>'
    return (
        '<div class="ck-rdv-box">'
        '<span class="ck-rdv-icon">📅</span>'
        f'<span class="ck-rdv-text">{_escape(plan)}</span>'
        '</div>'
    )


def _ck_important_banner(text: str) -> str:
    if not text:
        return ""
    return (
        '<div class="ck-important-banner">'
        '<div class="ck-important-banner-title">'
        '⚠️&nbsp; Note clinique importante'
        '</div>'
        f'<div class="ck-important-banner-body">{_escape(text)}</div>'
        '</div>'
    )


def _ck_hist_block(items: list[dict], mode: str = "traitement") -> str:
    if not items:
        return '<span class="ck-val-muted">Aucun renseigné</span>'
    rows_html = ""
    for item in items:
        if mode == "traitement":
            dt_raw   = item.get("date", "")
            txt      = _escape(item.get("label", ""))
            dt_label = _escape(dt_raw) if dt_raw else "—"
            rows_html += (
                f'<div class="ck-hist-row">'
                f'<span class="ck-hist-dt">{dt_label}</span>'
                f'<span class="ck-hist-txt">{txt}</span>'
                f'</div>'
            )
        else:
            dt_raw   = item.get("date", "")
            dt_label = f"Le {_escape(dt_raw)}" if dt_raw else "—"
            ord_txt  = _escape(item.get("ordonnance", ""))
            aut_txt  = _escape(item.get("autres", ""))
            body     = ord_txt or aut_txt
            sub      = (
                f'<div class="ck-hist-txt-muted">{aut_txt}</div>'
                if ord_txt and aut_txt else ""
            )
            rows_html += (
                f'<div class="ck-hist-row">'
                f'<span class="ck-hist-dt">{dt_label}</span>'
                f'<div>'
                f'<div class="ck-hist-txt">{body}</div>'
                f'{sub}'
                f'</div>'
                f'</div>'
            )
    return f'<div class="ck-hist">{rows_html}</div>'


# v13 AV BLOCK RENDERER
def _ck_av_block(av: dict) -> str:
    """Render the AV KPI grid (OD/OG × sc/cc)."""
    od_sc = av.get("od_sc", "")
    od_cc = av.get("od_cc", "")
    og_sc = av.get("og_sc", "")
    og_cc = av.get("og_cc", "")

    has_any = any([od_sc, od_cc, og_sc, og_cc])
    if not has_any:
        return '<span class="ck-val-muted">Non renseignée</span>'

    def _fmt_av(v: str) -> str:
        return _escape(v) if v else '<span style="color:#9CA3AF">—</span>'

    has_cc = bool(od_cc or og_cc)
    if has_cc:
        html = (
            '<div class="ck-av-grid">'
            # header row
            '<div></div>'
            '<div class="ck-av-header">sc</div>'
            '<div class="ck-av-header">cc</div>'
            # OD row
            f'<div class="ck-av-eye-label">OD</div>'
            f'<div class="ck-av-val">{_fmt_av(od_sc)}</div>'
            f'<div class="ck-av-val">{_fmt_av(od_cc)}</div>'
            # OG row
            f'<div class="ck-av-eye-label">OG</div>'
            f'<div class="ck-av-val">{_fmt_av(og_sc)}</div>'
            f'<div class="ck-av-val">{_fmt_av(og_cc)}</div>'
            '</div>'
        )
    else:
        # Only sc available (e.g. from tREFRACTION)
        html = (
            '<div class="ck-av-grid" style="grid-template-columns:auto 1fr;">'
            '<div></div>'
            '<div class="ck-av-header">sc</div>'
            f'<div class="ck-av-eye-label">OD</div>'
            f'<div class="ck-av-val">{_fmt_av(od_sc)}</div>'
            f'<div class="ck-av-eye-label">OG</div>'
            f'<div class="ck-av-val">{_fmt_av(og_sc)}</div>'
            '</div>'
        )

    src = av.get("source", "")
    src_note = (
        f'<div class="ck-av-sub" style="margin-top:6px;">Source : {_escape(src)}</div>'
        if src else ""
    )
    return html + src_note


# v13 PIO BLOCK RENDERER
def _ck_pio_block(pio: dict) -> str:
    """Render the PIO KPI tiles (OD / OG)."""
    od = pio.get("od", "")
    og = pio.get("og", "")
    alert = pio.get("alert", False)

    if not od and not og:
        return '<span class="ck-val-muted">Non renseignée</span>'

    def _tile(label: str, val: str) -> str:
        is_alert = False
        try:
            if float(val.replace(",", ".")) > 21:
                is_alert = True
        except (ValueError, AttributeError):
            pass
        alert_cls = " alert" if is_alert else ""
        val_cls   = " alert" if is_alert else ""
        icon      = "⚠ " if is_alert else ""
        return (
            f'<div class="ck-pio-eye{alert_cls}">'
            f'<span class="ck-pio-label">{label}</span>'
            f'<span class="ck-pio-val{val_cls}">{icon}{_escape(val) if val else "—"}</span>'
            f'<span class="ck-pio-unit">mmHg</span>'
            f'</div>'
        )

    return (
        f'<div class="ck-pio-grid">'
        + (_tile("OD", od) if od else "")
        + (_tile("OG", og) if og else "")
        + '</div>'
    )


# PIO TEMPORAL CHART — Plotly (v13 enrichissement)
_PIO_NORMAL_LIMIT = 21.0   # mmHg — upper limit of normotension (glaucoma threshold)

_PIO_COLOR_OD        = "#1D4ED8"   # royal blue  — convention OD
_PIO_COLOR_OG        = "#B91C1C"   # crimson red — convention OG
_PIO_COLOR_THRESHOLD = "#B45309"   # dark amber  — threshold line (distinct from OG red)


def _build_pio_fig(
    history: pd.DataFrame,
    show_od: bool = True,
    show_og: bool = True,
) -> "go.Figure":
    """
    PIO temporal evolution chart.

    history : DataFrame[date, od, og] sorted ascending.
    show_od / show_og : toggle traces.

    Design: white bg, OD=#1D4ED8, OG=#B91C1C, no threshold line/shading.
    Dynamic min/max annotations — merged when both eyes share the exact same
    date+value extremum.
    """
    dates     = history["date"]
    od_series = history["od"]
    og_series = history["og"]

    # Y-axis range from visible series only
    visible_vals = []
    if show_od:
        visible_vals.append(od_series)
    if show_og:
        visible_vals.append(og_series)
    all_visible = pd.concat(visible_vals).dropna() if visible_vals else pd.Series(dtype=float)
    y_max = float(all_visible.max()) + 4 if not all_visible.empty else 30
    y_min = max(0.0, float(all_visible.min()) - 4) if not all_visible.empty else 0

    fig = go.Figure()

    # OD trace
    if show_od and od_series.notna().any():
        fig.add_trace(go.Scatter(
            x=dates,
            y=od_series,
            mode="lines+markers",
            name="OD — Œil Droit",
            line=dict(color=_PIO_COLOR_OD, width=2.5),
            marker=dict(size=8, color=_PIO_COLOR_OD,
                        line=dict(width=2, color="#FFFFFF")),
            hovertemplate=(
                "<b style='color:" + _PIO_COLOR_OD + "'>OD</b>"
                " : <b>%{y:.1f} mmHg</b><br>"
                "<span style='color:#6B7280'>%{x|%d/%m/%Y}</span>"
                "<extra></extra>"
            ),
        ))

    # OG trace
    if show_og and og_series.notna().any():
        fig.add_trace(go.Scatter(
            x=dates,
            y=og_series,
            mode="lines+markers",
            name="OG — Œil Gauche",
            line=dict(color=_PIO_COLOR_OG, width=2.5),
            marker=dict(size=8, color=_PIO_COLOR_OG,
                        line=dict(width=2, color="#FFFFFF")),
            hovertemplate=(
                "<b style='color:" + _PIO_COLOR_OG + "'>OG</b>"
                " : <b>%{y:.1f} mmHg</b><br>"
                "<span style='color:#6B7280'>%{x|%d/%m/%Y}</span>"
                "<extra></extra>"
            ),
        ))

    # Dynamic min/max annotations
    def _add_extremum_annotation(
        val: float,
        date: pd.Timestamp,
        label_text: str,
        border_color: str,
        ay: int,
    ) -> None:
        """Add a styled annotation pointing to a single extremum point."""
        date_str = date.strftime("%d/%m/%Y")
        fig.add_annotation(
            x=date,
            y=val,
            text=f"<b>{label_text}</b><br><span style='color:#6B7280;font-size:10px'>{date_str}</span>",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1.5,
            arrowcolor=border_color,
            ax=0,
            ay=ay,
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor=border_color,
            borderwidth=1.5,
            borderpad=4,
            font=dict(size=11, color="#111111"),
        )

    if not all_visible.empty:
        # Build a lookup: (date, eye) → value for visible points
        points: list[dict] = []  # {date, val, eye}
        if show_od:
            for d, v in zip(dates, od_series):
                if pd.notna(v):
                    points.append({"date": d, "val": float(v), "eye": "od"})
        if show_og:
            for d, v in zip(dates, og_series):
                if pd.notna(v):
                    points.append({"date": d, "val": float(v), "eye": "og"})

        global_max = max(p["val"] for p in points)
        global_min = min(p["val"] for p in points)

        def _resolve_annotation(target_val: float, ay: int) -> None:
            """
            Find all visible points matching target_val, determine if OD/OG
            share the same date+value, then add a merged or per-eye annotation.
            """
            matches = [p for p in points if p["val"] == target_val]
            # Group by date
            by_date: dict = {}
            for p in matches:
                key = p["date"]
                by_date.setdefault(key, set()).add(p["eye"])

            # Pick the date with the most eyes involved (prefer merge)
            best_date = max(by_date, key=lambda d: len(by_date[d]))
            eyes_at_best = by_date[best_date]

            if "od" in eyes_at_best and "og" in eyes_at_best:
                # Merged annotation
                label = f"OD = OG = {target_val:.1f} mmHg"
                border = "#6B7280"
            elif "od" in eyes_at_best:
                label = f"OD : {target_val:.1f} mmHg"
                border = _PIO_COLOR_OD
            else:
                label = f"OG : {target_val:.1f} mmHg"
                border = _PIO_COLOR_OG

            _add_extremum_annotation(target_val, best_date, label, border, ay)

        _resolve_annotation(global_max, ay=-40)  # annotation above point
        if global_min != global_max:
            _resolve_annotation(global_min, ay=40)  # annotation below point

    # Layout
    # Explicit colours — Streamlit's dark theme would otherwise override them.
    _FONT_COLOR  = "#111111"
    _GRID_COLOR  = "#E5E7EB"
    _FONT_FAMILY = "'Segoe UI', Arial, sans-serif"

    fig.update_layout(
        height=300,
        margin=dict(l=4, r=12, t=36, b=4),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.04,
            xanchor="left",   x=0,
            font=dict(size=11, color=_FONT_COLOR, family=_FONT_FAMILY),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#D1D5DB",
            borderwidth=1,
        ),
        xaxis=dict(
            title=None,
            showgrid=False,
            showline=True,
            linecolor="#9CA3AF",
            linewidth=1,
            tickformat="%b %Y",
            tickfont=dict(size=10, color=_FONT_COLOR, family=_FONT_FAMILY),
            ticks="outside",
            ticklen=4,
            tickcolor="#9CA3AF",
        ),
        yaxis=dict(
            title=dict(
                text="PIO (mmHg)",
                font=dict(size=10, color=_FONT_COLOR, family=_FONT_FAMILY),
            ),
            tickfont=dict(size=10, color=_FONT_COLOR, family=_FONT_FAMILY),
            showgrid=True,
            gridcolor=_GRID_COLOR,
            gridwidth=1,
            showline=True,
            linecolor="#9CA3AF",
            linewidth=1,
            range=[y_min, y_max],
            zeroline=False,
            ticks="outside",
            ticklen=4,
            tickcolor="#9CA3AF",
        ),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            bordercolor="#D1D5DB",
            font=dict(size=11, color=_FONT_COLOR, family=_FONT_FAMILY),
        ),
        font=dict(family=_FONT_FAMILY, color=_FONT_COLOR),
    )

    return fig


def _render_pio_chart(record: dict) -> None:
    """
    Render the full PIO section:
      1. KPI metrics row  (last value + delta vs previous visit)
      2. st.tabs with 3 views: OD + OG | OD seul | OG seul
         — default tab is always "OD + OG" (first position)

    Falls back gracefully when no PIO data exists.
    """
    history = _extract_pio_history(record)

    # Graceful empty state
    if history.empty:
        st.markdown(
            '<div style="font-size:0.80rem;color:#6B7280;font-style:italic;'
            'padding:8px 0 4px 0;">'
            '📊 Aucune donnée PIO historisée disponible pour ce patient.'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    # KPI metrics — last value vs previous visit
    last = history.iloc[-1]
    prev = history.iloc[-2] if len(history) >= 2 else None

    def _delta(col: str) -> float | None:
        if prev is None:
            return None
        curr_v, prev_v = last[col], prev[col]
        if pd.isna(curr_v) or pd.isna(prev_v):
            return None
        return round(float(curr_v) - float(prev_v), 1)

    def _fmt_val(v) -> str:
        return f"{v:.0f} mmHg" if not pd.isna(v) else "—"

    def _delta_label(d: float | None) -> str | None:
        if d is None:
            return None
        sign = "+" if d > 0 else ""
        return f"{sign}{d:.1f} vs visite préc."

    col_od, col_og, col_spacer = st.columns([1, 1, 2])

    with col_od:
        od_val   = last["od"]
        od_delta = _delta("od")
        od_alert = (not pd.isna(od_val)) and float(od_val) > _PIO_NORMAL_LIMIT
        st.metric(
            label="PIO — Œil Droit (OD)",
            value=_fmt_val(od_val) if not pd.isna(od_val) else "—",
            delta=_delta_label(od_delta),
            delta_color="inverse",   # red when rising, green when falling
            help="Pression intra-oculaire OD. Seuil normal : ≤ 21 mmHg.",
        )
        if od_alert:
            st.markdown(
                '<span style="font-size:0.72rem;color:#DC2626;font-weight:700;">'
                '⚠ Hypertonie OD</span>',
                unsafe_allow_html=True,
            )

    with col_og:
        og_val   = last["og"]
        og_delta = _delta("og")
        og_alert = (not pd.isna(og_val)) and float(og_val) > _PIO_NORMAL_LIMIT
        st.metric(
            label="PIO — Œil Gauche (OG)",
            value=_fmt_val(og_val) if not pd.isna(og_val) else "—",
            delta=_delta_label(og_delta),
            delta_color="inverse",
            help="Pression intra-oculaire OG. Seuil normal : ≤ 21 mmHg.",
        )
        if og_alert:
            st.markdown(
                '<span style="font-size:0.72rem;color:#DC2626;font-weight:700;">'
                '⚠ Hypertonie OG</span>',
                unsafe_allow_html=True,
            )

    # Chart — requires plotly
    if not _PLOTLY_AVAILABLE:
        st.info("Installez plotly (`pip install plotly`) pour afficher le graphique d'évolution.")
        return

    n_pts   = len(history)
    caption = (
        f"{n_pts} mesure(s) disponible(s)  ·  "
        "Ligne pointillée : limite supérieure de la normale (21 mmHg)  ·  "
        "OD = Œil Droit (bleu)  ·  OG = Œil Gauche (rouge)"
    )
    _CHART_CFG = {"displayModeBar": False, "responsive": True}

    # st.tabs — first tab is active by default (Streamlit guarantee)
    tab_both, tab_od, tab_og = st.tabs(["OD + OG", "OD seul", "OG seul"])

    with tab_both:
        st.plotly_chart(
            _build_pio_fig(history, show_od=True, show_og=True),
            use_container_width=True,
            config=_CHART_CFG,
        )
        st.caption(caption)

    with tab_od:
        od_history = history[history["od"].notna()]
        if od_history.empty:
            st.info("Aucune mesure OD disponible.")
        else:
            st.plotly_chart(
                _build_pio_fig(history, show_od=True, show_og=False),
                use_container_width=True,
                config=_CHART_CFG,
            )
            st.caption(caption)

    with tab_og:
        og_history = history[history["og"].notna()]
        if og_history.empty:
            st.info("Aucune mesure OG disponible.")
        else:
            st.plotly_chart(
                _build_pio_fig(history, show_od=False, show_og=True),
                use_container_width=True,
                config=_CHART_CFG,
            )
            st.caption(caption)


# v13 TRAITEMENTS BLOCK (local vs systémique split)
def _ck_traitements_block(trt_history: list[dict]) -> str:
    """
    Render treatments split into Collyres (local) and Systémiques sections.
    Falls back to a single undivided list if no classification applies.
    """
    if not trt_history:
        return '<span class="ck-val-muted">Aucun renseigné</span>'

    local_items    = [i for i in trt_history if i.get("type") == "local"]
    systemic_items = [i for i in trt_history if i.get("type") != "local"]

    has_local    = bool(local_items)
    has_systemic = bool(systemic_items)

    def _render_items(items: list[dict]) -> str:
        rows = ""
        for item in items:
            dt_raw   = item.get("date", "")
            dt_label = _escape(dt_raw) if dt_raw else "—"
            txt      = _escape(item.get("label", ""))
            rows += (
                f'<div class="ck-hist-row">'
                f'<span class="ck-hist-dt">{dt_label}</span>'
                f'<span class="ck-hist-txt">{txt}</span>'
                f'</div>'
            )
        return rows

    html = '<div class="ck-hist">'

    if has_local and has_systemic:
        html += '<div class="ck-trt-section">Collyres / Locaux</div>'
        html += _render_items(local_items)
        html += '<div class="ck-trt-section">Systémiques</div>'
        html += _render_items(systemic_items)
    else:
        # No meaningful split — just render all
        html += _render_items(trt_history)

    html += '</div>'
    return html


# TABLEAU DES ACTES — NATIVE STREAMLIT RENDERER
_ACTES_CSS = """
<style>
/* Tableau des Actes card */
/* No forced background — theme provides white via config.toml */
.ck-actes-wrap-st {
    border-radius: 12px;
    border: 1px solid #E5E7EB;
    box-shadow: 0 4px 28px rgba(0,0,0,0.07);
    overflow: hidden;
    margin-bottom: 1.2rem;
    font-family: 'Segoe UI', system-ui, Arial, sans-serif;
}
.ck-actes-st-header {
    display: flex; align-items: center; justify-content: space-between;
    background: linear-gradient(135deg, #1B3A6B 0%, #234DA8 100%);
    padding: 10px 20px;
}
.ck-actes-st-title {
    font-size: 0.69rem; font-weight: 800; text-transform: uppercase;
    letter-spacing: 0.12em; color: #fff;
}
.ck-actes-st-meta { font-size: 0.67rem; opacity: 0.72; color: #fff; }
.ck-actes-st-body { padding: 0 16px 8px 16px; }
.ck-actes-st-col-hdr {
    font-size: 0.59rem; font-weight: 800; text-transform: uppercase;
    letter-spacing: 0.10em; color: #6B7280;
    padding: 10px 0 6px 0; border-bottom: 2px solid #E5E7EB;
}
.ck-actes-st-date {
    font-size: 0.84rem; font-weight: 700; color: #1B3A6B;
    padding-top: 12px;
}
.ck-actes-st-motif {
    font-size: 0.84rem; font-weight: 600; color: #111827;
    margin-bottom: 4px; padding-top: 10px;
}
.ck-actes-st-motif-empty {
    font-size: 0.79rem; font-style: italic; color: #6B7280;
    margin-bottom: 4px; padding-top: 10px;
}
.ck-actes-st-doctor {
    font-size: 0.78rem; color: #6B7280;
    padding-top: 12px;
}
.ck-actes-st-sep {
    border: none; border-top: 1px solid #E5E7EB; margin: 4px 0;
}
</style>
"""


def _filter_record_by_date(record: dict, date_str: str) -> dict:
    """
    Return a copy of record with each DataFrame filtered to rows matching date_str.
    - Consultation / Documents: matched on their own Date column (formatted DD/MM/YYYY).
    - tKERATO / tREFRACTION: matched via the NumConsult of consultations on that date.
    """
    consult_df = _safe_df(record, "Consultation")

    # Collect NumConsult values for consultations on this date
    matching_nc: set[str] = set()
    if consult_df is not None and "Date" in consult_df.columns:
        tmp = consult_df.copy()
        tmp["_dt_fmt"] = _parse_dates(tmp["Date"]).dt.strftime("%d/%m/%Y")
        for _, row in tmp[tmp["_dt_fmt"] == date_str].iterrows():
            nc = _str_id(row.get("N° consultation"))
            if nc:
                matching_nc.add(nc)

    filtered: dict = {}

    for key, df in record.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            filtered[key] = df
            continue

        if key == "Consultation" and "Date" in df.columns:
            tmp = df.copy()
            tmp["_dt_fmt"] = _parse_dates(tmp["Date"]).dt.strftime("%d/%m/%Y")
            filtered[key] = tmp[tmp["_dt_fmt"] == date_str].drop(columns=["_dt_fmt"])

        elif key in ("tKERATO", "tREFRACTION") and "NumConsult" in df.columns:
            if matching_nc:
                mask = df["NumConsult"].astype(str).isin(matching_nc)
                filtered[key] = df[mask]
            else:
                filtered[key] = df.iloc[0:0]  # empty, same columns

        elif key == "Documents" and "Date" in df.columns:
            tmp = df.copy()
            tmp["_dt_fmt"] = _parse_dates(tmp["Date"]).dt.strftime("%d/%m/%Y")
            filtered[key] = tmp[tmp["_dt_fmt"] == date_str].drop(columns=["_dt_fmt"])

        else:
            filtered[key] = df  # identity (e.g. "identity" DataFrame)

    return filtered


def _render_actes_streamlit(
    rows: list[dict],
    n_total: int,
    record: dict,
    generate_pdf_bytes_fn,
    pdf_available: bool,
    full_name: str,
    dob_str: str,
    patient_id: str,
) -> None:
    """
    Render the Tableau des Actes using native Streamlit widgets.

    - Forced white background (matches the 360° card).
    - Each row has a "Consulter cette journée" download button.
    - PDF contains only data for that specific date.
    """
    st.markdown(_ACTES_CSS, unsafe_allow_html=True)

    last_date = rows[0]["date_str"] if rows else "—"

    # Card wrapper open — white background forced via CSS class
    st.markdown(
        '<div class="ck-actes-wrap-st">'
        '<div class="ck-actes-st-header">'
        f'<span class="ck-actes-st-title">Tableau des Actes</span>'
        f'<span class="ck-actes-st-meta">'
        f'{n_total} visite(s) &nbsp;·&nbsp; Dernier acte : {_escape(last_date)}'
        f'</span></div>'
        '<div class="ck-actes-st-body">',
        unsafe_allow_html=True,
    )

    if not rows:
        st.info("Aucun acte enregistré pour ce patient.")
        st.markdown('</div></div>', unsafe_allow_html=True)
        return

    # Column proportions: Date | Actes réalisés | Praticien | Bouton
    col_w = [1.2, 4, 2, 1.6]

    # Header labels
    h_date, h_actes, h_doctor, h_pdf = st.columns(col_w)
    h_date.markdown('<div class="ck-actes-st-col-hdr">Date</div>', unsafe_allow_html=True)
    h_actes.markdown('<div class="ck-actes-st-col-hdr">Actes réalisés</div>', unsafe_allow_html=True)
    h_doctor.markdown('<div class="ck-actes-st-col-hdr">Praticien</div>', unsafe_allow_html=True)
    h_pdf.markdown('<div class="ck-actes-st-col-hdr">Rapport</div>', unsafe_allow_html=True)

    for i, r in enumerate(rows):
        c_date, c_actes, c_doctor, c_pdf = st.columns(col_w)

        with c_date:
            st.markdown(
                f'<div class="ck-actes-st-date">{_escape(r["date_str"])}</div>',
                unsafe_allow_html=True,
            )

        with c_actes:
            motif_cls  = "ck-actes-st-motif" if r["motif"] else "ck-actes-st-motif-empty"
            motif_text = _escape(r["motif"][:80]) if r["motif"] else "Consultation"
            chips_html = ""
            if r["tech_actes"]:
                chips = [
                    f'<span class="{_BADGE_CSS.get(cat, "ck-chip ck-chip-grey")}">'
                    f'{_escape(label)}</span>'
                    for label, cat in r["tech_actes"]
                ]
                chips_html = f'<div class="ck-act-chips">{"".join(chips)}</div>'
            st.markdown(
                f'<div class="{motif_cls}">{motif_text}</div>{chips_html}',
                unsafe_allow_html=True,
            )

        with c_doctor:
            st.markdown(
                f'<div class="ck-actes-st-doctor">'
                f'{_escape(r["doctor"]) if r["doctor"] else "—"}'
                f'</div>',
                unsafe_allow_html=True,
            )

        with c_pdf:
            date_str  = r["date_str"]
            safe_date = date_str.replace("/", "-")
            safe_name = full_name.replace(" ", "_")
            filename  = f"consultation_{safe_name}_{safe_date}.pdf"
            btn_key   = f"pdf_acte_{i}_{date_str}"

            # Vertical padding to align button with text rows
            st.markdown("<div style='padding-top:8px'>", unsafe_allow_html=True)
            if pdf_available and generate_pdf_bytes_fn is not None:
                record_filtre = _filter_record_by_date(record, date_str)
                pdf_bytes = generate_pdf_bytes_fn(
                    record_filtre, full_name, dob_str, patient_id
                )
                st.download_button(
                    label="Consulter cette journée",
                    data=pdf_bytes,
                    file_name=filename,
                    mime="application/pdf",
                    key=btn_key,
                    help=f"Télécharger le rapport du {date_str}",
                    use_container_width=True,
                )
            else:
                st.button(
                    "Consulter cette journée",
                    key=btn_key,
                    disabled=True,
                    help="fpdf2 non installé — pip install fpdf2",
                    use_container_width=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

        # Row separator
        st.markdown('<hr class="ck-actes-st-sep">', unsafe_allow_html=True)

    # Close card wrapper divs
    st.markdown('</div></div>', unsafe_allow_html=True)


# 360° CARD HTML — v13 DYNAMIC GRID
def _360_card_html(data: dict) -> str:
    ant            = data["antecedents_allergies"]
    plan           = data["plan_suivi"]
    diag           = data["diagnostic"]
    diag_date      = data.get("diagnostic_date", "")
    n_c            = data["n_consult"]
    date           = data["last_consult_date"]
    important      = data.get("important", "")
    contact        = data.get("contact", {"telephone": "", "adresse_par": ""})
    date_creation  = data.get("date_creation", "")
    trt_history    = data.get("traitements_history", [])
    presc_history  = data.get("prescriptions_history", [])
    av             = data.get("visual_acuity", {})
    pio            = data.get("pio", {})

    # Header
    contact_parts = []
    if contact.get("telephone"):
        contact_parts.append(f"📞 {_escape(contact['telephone'])}")
    if contact.get("adresse_par"):
        contact_parts.append(f"Réf. : {_escape(contact['adresse_par'])}")
    contact_line = (
        f'<div class="ck-head-contact">{" &nbsp;·&nbsp; ".join(contact_parts)}</div>'
        if contact_parts else ""
    )
    dossier_line = (
        f'<div class="ck-head-dossier">Dossier ouvert le : {_escape(date_creation)}</div>'
        if date_creation else ""
    )
    header = (
        '<div class="ck-head">'
        '<div>'
        '<span class="ck-head-title">Profil Patient 360°</span>'
        + contact_line + dossier_line +
        '</div>'
        f'<span class="ck-head-meta">Dernière consultation : {date}'
        f' &nbsp;·&nbsp; {n_c} visite(s)</span>'
        '</div>'
    )

    # Important alert banner
    important_html = _ck_important_banner(important)

    # Row 1: AV | PIO | Antécédents & Allergies (dynamic — empty columns omitted)
    row1_cells = []
    # Block 1 — Acuité Visuelle
    av_html = _ck_av_block(av)
    av_date = av.get("date", "")
    row1_cells.append(
        '<div class="ck-cell ck-cell-av">'
        + _ck_lbl(1, "Acuité Visuelle", av_date)
        + av_html
        + '</div>'
    )

    # Block 2 — PIO (only if data available)
    od_val = pio.get("od", "")
    og_val = pio.get("og", "")
    if od_val or og_val:
        pio_date = pio.get("date", "")
        row1_cells.append(
            '<div class="ck-cell ck-cell-pio">'
            + _ck_lbl(2, "PIO", pio_date)
            + _ck_pio_block(pio)
            + '</div>'
        )

    # Block 3 — Antécédents & Allergies
    has_ant = bool(ant["antecedents"])
    has_alg = bool(ant["allergies"])
    if has_ant or has_alg:
        row1_cells.append(
            '<div class="ck-cell">'
            + _ck_lbl(3, "Antécédents &amp; Allergies")
            + _ck_mixed_tags(ant["antecedents"], ant["allergies"])
            + '</div>'
        )

    row1 = (
        '<div class="ck-row">' + "".join(row1_cells) + '</div>'
        if row1_cells else ""
    )

    # Row 2: Traitements | Diagnostic OPH | Plan de suivi (dynamic)
    row2_cells = []
    # Block 4 — Traitements (always rendered, shows "aucun" if empty)
    row2_cells.append(
        '<div class="ck-cell ck-cell-teal">'
        + _ck_lbl(4, "Traitements")
        + _ck_traitements_block(trt_history)
        + '</div>'
    )

    # Block 5 — Diagnostic OPH (only if present)
    if diag:
        diag_html = f'<div class="ck-val">{_escape(diag)}</div>'
        row2_cells.append(
            '<div class="ck-cell ck-cell-blue">'
            + _ck_lbl(5, "Diagnostic OPH", diag_date)
            + diag_html
            + '</div>'
        )

    # Block 6 — Plan de suivi (only if present)
    if plan:
        row2_cells.append(
            '<div class="ck-cell ck-cell-rdv">'
            + _ck_lbl(6, "Plan de suivi")
            + _ck_rdv(plan)
            + '</div>'
        )

    row2 = (
        '<div class="ck-row">' + "".join(row2_cells) + '</div>'
        if row2_cells else ""
    )

    # Row 3: Prescriptions history (only if present)
    row3 = ""
    if presc_history:
        row3 = (
            '<div class="ck-row">'
            '<div class="ck-cell" style="flex:1;">'
            + _ck_lbl(7, "Prescriptions")
            + _ck_hist_block(presc_history, mode="prescription")
            + '</div>'
            '</div>'
        )

    return (
        '<div class="ck-card">'
        + header
        + important_html
        + row1 + row2 + row3
        + '</div>'
    )


# ULTRA-THIN PATIENT HEADER STRIP (v13)
def _patient_header_strip_html(record: dict) -> str:
    """
    Single-line strip: ID Patient | N° consultations | Dernière visite
    Rendered above the Tableau des Actes, inside the expander.
    """
    id_df = _safe_df(record, "identity")
    patient_id = "—"
    if id_df is not None and not id_df.empty:
        raw_id = id_df.iloc[0].get("Code patient")
        if raw_id is not None and not _is_null_val(raw_id):
            patient_id = _escape(str(raw_id).strip())

    n_c       = _n_consult(record)
    last_date = _last_consult_date(record)

    def _item(label: str, value: str) -> str:
        return (
            f'<span class="ck-header-strip-item">'
            f'<span class="ck-header-strip-label">{label}</span>'
            f'{value}'
            f'</span>'
        )

    return (
        '<div class="ck-header-strip">'
        + _item("ID Patient", patient_id)
        + _item("Consultations", str(n_c))
        + _item("Dernière visite", _escape(last_date))
        + '</div>'
    )


# MAIN RENDER
def _render_dashboard(
    record: dict,
    generate_pdf_bytes_fn=None,
    pdf_available: bool = False,
    full_name: str = "",
    dob_str: str = "",
    patient_id: str = "",
) -> None:
    data    = analyse_patient(record)
    actes   = _build_actes_rows(record)
    n_total = data["n_consult"]

    # Patient header strip
    st.markdown(_patient_header_strip_html(record), unsafe_allow_html=True)
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # Zone 1 : Tableau des Actes (native Streamlit — supports PDF buttons)
    _render_actes_streamlit(
        actes, n_total, record,
        generate_pdf_bytes_fn=generate_pdf_bytes_fn,
        pdf_available=pdf_available,
        full_name=full_name,
        dob_str=dob_str,
        patient_id=patient_id,
    )

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # Zone 2 : Profil Patient 360°
    st.markdown(_360_card_html(data), unsafe_allow_html=True)

    # Zone 3 : PIO — Évolution temporelle
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:0.72rem;font-weight:800;color:#1B3A6B;'
        'letter-spacing:0.07em;text-transform:uppercase;'
        'border-bottom:1px solid #E2E8F0;padding-bottom:5px;'
        'margin-bottom:10px;font-family:\'Segoe UI\',sans-serif;">'
        '📈&nbsp; PIO — Évolution temporelle (OD / OG)'
        '</div>',
        unsafe_allow_html=True,
    )
    _render_pio_chart(record)


def render_medical_summary(
    record: dict,
    generate_pdf_bytes_fn=None,
    pdf_available: bool = False,
    full_name: str = "",
    dob_str: str = "",
    patient_id: str = "",
) -> None:
    """Main public entry point — renders the Cockpit Dashboard in Streamlit."""
    # Inject JS to collapse sidebar once a patient is loaded
    # (works in Streamlit ≥ 1.28 where the sidebar toggle button is present)
    _collapse_sidebar_js = """
    <script>
    (function() {
        try {
            const btn = window.parent.document.querySelector(
                'button[data-testid="collapsedControl"], '
                + 'button[aria-label="Close sidebar"], '
                + 'button[title="Close sidebar"]'
            );
            // Only auto-collapse if sidebar is currently expanded
            const sidebar = window.parent.document.querySelector(
                '[data-testid="stSidebar"]'
            );
            if (sidebar) {
                const expanded = sidebar.getAttribute('aria-expanded');
                if (expanded === 'true' && btn) { btn.click(); }
            }
        } catch(e) {}
    })();
    </script>
    """
    st.markdown(_collapse_sidebar_js, unsafe_allow_html=True)

    with st.expander("Synthèse de Consultation", expanded=True):
        st.markdown(_CSS, unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:0.78rem;font-weight:800;color:#1B3A6B;'
            'letter-spacing:0.07em;text-transform:uppercase;'
            'border-bottom:2px solid #1B3A6B;padding-bottom:6px;'
            'margin-bottom:1.2rem;font-family:\'Segoe UI\',sans-serif;">'
            'Tableau de Bord — Suivi Long Terme'
            '</div>',
            unsafe_allow_html=True,
        )
        _render_dashboard(
            record,
            generate_pdf_bytes_fn=generate_pdf_bytes_fn,
            pdf_available=pdf_available,
            full_name=full_name,
            dob_str=dob_str,
            patient_id=patient_id,
        )
        st.caption(
            f"Analyse générée le {datetime.now().strftime('%d/%m/%Y à %H:%M')}  ·  "
            "Données issues du dossier local  ·  "
            "Outil d'aide à la décision — ne remplace pas l'examen clinique."
        )


# PUBLIC ANALYSIS API
def analyse_patient(record: dict) -> dict:
    """
    Extract and return all structured data for the dashboard.
    v13 new keys: visual_acuity, pio (as dict).
    All v12 keys preserved for backwards compatibility.
    """
    if not record:
        return {}

    ant                     = _extract_antecedents(record)
    motif_text, motif_date  = _extract_motif(record)
    diag_text, diag_date    = _extract_diagnostic(record)
    imp_text, _imp_items    = _extract_important(record)

    return {
        # v13 new keys
        "visual_acuity":         _extract_visual_acuity(record),
        "pio":                   _extract_pio(record),

        # Core keys (v12 unchanged)
        "motif":                 motif_text,
        "motif_date":            motif_date,
        "diagnostic":            diag_text,
        "diagnostic_date":       diag_date,
        "antecedents_allergies": ant,
        "traitements":           _extract_traitements(record),
        "prescriptions":         _extract_prescriptions(record),

        # v12 historised keys
        "traitements_history":   _extract_traitements_history(record),
        "prescriptions_history": _extract_prescriptions_history(record),

        # Unchanged
        "plan_suivi":            _extract_plan_suivi(record),
        "pio_alert":             _extract_pio_alert(record),   # kept for compat
        "last_consult_date":     _last_consult_date(record),
        "n_consult":             _n_consult(record),
        "important":             imp_text,
        "important_items":       _imp_items,
        "date_creation":         _get_date_creation(record),
        "contact":               _extract_contact_info(record),

        # Legacy keys
        "antecedents_perso": [
            i["label"] if isinstance(i, dict) else i
            for i in ant["antecedents"]
        ],
        "antecedents_fam": [],
    }


def analyse_parcours_soin(record: dict) -> dict:
    """Backwards-compatible alias."""
    return analyse_patient(record)


# MARKDOWN FALLBACK  (PDF export / text environments)
def generate_medical_summary(record: dict) -> str:
    """Return a structured Markdown string — used for PDF export."""
    if not record:
        return "_Aucune donnée disponible._"

    d     = analyse_patient(record)
    ant   = d["antecedents_allergies"]
    presc = d["prescriptions"]
    imp   = d.get("important", "")
    cont  = d.get("contact", {})
    dc    = d.get("date_creation", "")
    trt_h = d.get("traitements_history", [])
    prs_h = d.get("prescriptions_history", [])
    av    = d.get("visual_acuity", {})
    pio   = d.get("pio", {})

    lines = [
        "## Synthèse de Suivi Ophtalmologique\n",
        f"**Dernière consultation :** {d['last_consult_date']}  ",
        f"**Nombre de consultations :** {d['n_consult']}\n",
    ]

    if dc:
        lines.append(f"**Dossier ouvert le :** {dc}  ")
    if cont.get("telephone"):
        lines.append(f"**Tél. :** {cont['telephone']}  ")
    if cont.get("adresse_par"):
        lines.append(f"**Référent :** {cont['adresse_par']}\n")

    lines.append("---\n")

    if imp:
        lines += ["### ⚠ Note clinique importante", imp, "", "---\n"]

    # AV
    lines.append("### 1. Acuité Visuelle")
    av_date_txt = f" *(en date du {av['date']})*" if av.get("date") else ""
    lines.append(f"*Source : {av.get('source', '?')}*{av_date_txt}")
    if av.get("od_sc") or av.get("od_cc"):
        parts = []
        if av.get("od_sc"):
            parts.append(f"sc {av['od_sc']}")
        if av.get("od_cc"):
            parts.append(f"cc {av['od_cc']}")
        lines.append(f"- OD : {' / '.join(parts)}")
    if av.get("og_sc") or av.get("og_cc"):
        parts = []
        if av.get("og_sc"):
            parts.append(f"sc {av['og_sc']}")
        if av.get("og_cc"):
            parts.append(f"cc {av['og_cc']}")
        lines.append(f"- OG : {' / '.join(parts)}")
    if not any([av.get("od_sc"), av.get("od_cc"), av.get("og_sc"), av.get("og_cc")]):
        lines.append("_Non renseignée_")

    # PIO
    lines.append("\n### 2. PIO (Pression Intra-Oculaire)")
    pio_date_txt = f" *(en date du {pio['date']})*" if pio.get("date") else ""
    if pio.get("od") or pio.get("og"):
        lines.append(f"*Mesure{pio_date_txt}*")
        if pio.get("od"):
            lines.append(f"- OD : {pio['od']} mmHg")
        if pio.get("og"):
            lines.append(f"- OG : {pio['og']} mmHg")
        if pio.get("alert"):
            lines.append("⚠ Hypertonie oculaire détectée — surveillance renforcée recommandée")
    else:
        lines.append("_Non renseignée_")

    # Antécédents
    lines += ["\n### 3. Antécédents & Allergies"]
    for item in ant["antecedents"]:
        if isinstance(item, dict):
            date_txt = f" [{item['date']}]" if item.get("date") else ""
            lines.append(f"- {item['label']}{date_txt}")
        else:
            lines.append(f"- {item}")
    if ant["allergies"]:
        lines.append("**Allergies :**")
        for item in ant["allergies"]:
            if isinstance(item, dict):
                date_txt = f" [{item['date']}]" if item.get("date") else ""
                lines.append(f"- ⚠ {item['label']}{date_txt}")
            else:
                lines.append(f"- ⚠ {item}")
    if not ant["antecedents"] and not ant["allergies"]:
        lines.append("_Aucun renseigné_")

    # Traitements
    lines += ["\n### 4. Traitements"]
    if trt_h:
        local_items    = [i for i in trt_h if i.get("type") == "local"]
        systemic_items = [i for i in trt_h if i.get("type") != "local"]
        if local_items and systemic_items:
            lines.append("**Collyres / Locaux :**")
            for item in local_items:
                dt = f"[{item['date']}] " if item.get("date") else ""
                lines.append(f"- {dt}{item['label']}")
            lines.append("**Systémiques :**")
            for item in systemic_items:
                dt = f"[{item['date']}] " if item.get("date") else ""
                lines.append(f"- {dt}{item['label']}")
        else:
            for item in trt_h:
                dt = f"[{item['date']}] " if item.get("date") else ""
                lines.append(f"- {dt}{item['label']}")
    elif d["traitements"]:
        lines.extend(f"- {i}" for i in d["traitements"])
    else:
        lines.append("_Aucun renseigné_")

    # Diagnostic
    diag_date_txt = f" *(en date du {d['diagnostic_date']})*" if d.get("diagnostic_date") else ""
    lines += [
        f"\n### 5. Diagnostic OPH{diag_date_txt}",
        d["diagnostic"] or "_Non renseigné_",
    ]

    # Plan de suivi
    lines += ["\n### 6. Plan de suivi", d["plan_suivi"] or "_Non planifié_"]

    # Prescriptions
    lines += ["\n### 7. Prescriptions"]
    if prs_h:
        for p in prs_h:
            dt = f"Le {p['date']} : " if p.get("date") else ""
            if p.get("ordonnance"):
                lines.append(f"- {dt}{p['ordonnance']}")
            if p.get("autres"):
                lines.append(f"  *(Autres : {p['autres']})*")
    elif presc.get("ordonnance") or presc.get("autres"):
        if presc.get("date"):
            lines.append(f"*En date du {presc['date']}*")
        if presc["ordonnance"]:
            lines.append(f"**Ordonnance :** {presc['ordonnance']}")
        if presc["autres"]:
            lines.append(f"**Autres :** {presc['autres']}")
    else:
        lines.append("_Aucune prescription renseignée_")

    if d["pio_alert"]:
        lines += ["", "---", f"⚠ **Alerte PIO :** {d['pio_alert']}"]

    return "\n".join(lines)