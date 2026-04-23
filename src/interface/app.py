import streamlit as st
import pandas as pd
import sys

sys.path.append("src")
from extraction import load_all_data, get_full_patient_record

try:
    from fpdf import FPDF
    import fpdf as _fpdf_pkg
    from pathlib import Path
    _FPDF_FONT_DIR = Path(_fpdf_pkg.__file__).parent / "fonts"
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Dossier Patient - Ophtalmologie",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Streamlit UI
st.markdown(
    """
    <style>
        .block-container { padding-top: 1.5rem; }
        button[data-baseweb="tab"] {
            font-size: 0.875rem;
            font-weight: 600;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }
        [data-testid="stSidebar"] { background-color: var(--secondary-background-color); }
        [data-testid="stExpander"] {
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 6px;
            margin-bottom: 0.45rem;
        }
        .field-label {
            font-size: 0.72rem;
            color: var(--text-color);
            opacity: 0.55;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 1px;
        }
        .field-value {
            font-size: 0.95rem;
            font-weight: 600;
            color: var(--text-color);
            margin-bottom: 0.6rem;
        }
        .eye-header {
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            padding: 3px 0 3px 10px;
            margin-bottom: 6px;
            background: transparent !important;
            color: var(--text-color) !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Helper utilities
def is_empty(val) -> bool:
    """Return True for any absent/empty value: None, NaN, NaT, blank string."""
    if val is None:
        return True
    try:
        if pd.isna(val):
            return True
    except (TypeError, ValueError):
        pass
    if isinstance(val, str) and not val.strip():
        return True
    return False

def to_datetime_safe(series: pd.Series) -> pd.Series:
    
    # Use dayfirst=True for DD/MM/YYYY, fallback for older pandas
    try:
        return pd.to_datetime(series, errors="coerce", format="mixed", dayfirst=True)
    except ValueError:
        return pd.to_datetime(series, errors="coerce", dayfirst=True)

def sort_by_date(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = to_datetime_safe(df[date_col])
    return df.sort_values(date_col, ascending=False, na_position="last")

def fmt_date_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    if col in df.columns:
        df[col] = to_datetime_safe(df[col]).dt.strftime("%d/%m/%Y").fillna("")
    return df

def sort_exams_via_consult(
    exam_df: pd.DataFrame,
    consult_df: pd.DataFrame,
    consult_date_col: str = "Date",
    num_consult_col: str = "NumConsult",
    consult_id_col: str = "N° consultation",
    keep_date: bool = False,
) -> pd.DataFrame:
    """
    Sort tKERATO / tREFRACTION rows by their linked consultation date.
    If keep_date=True, the formatted date is kept as '_date_tmp' for display.
    """
    if exam_df is None or exam_df.empty:
        return exam_df
    date_lookup = (
        consult_df[[consult_id_col, consult_date_col]]
        .rename(columns={consult_id_col: num_consult_col, consult_date_col: "_date_tmp"})
        .drop_duplicates(subset=[num_consult_col])
    )
    merged = exam_df.merge(date_lookup, on=num_consult_col, how="left")
    merged["_date_tmp"] = to_datetime_safe(merged["_date_tmp"])
    merged = merged.sort_values("_date_tmp", ascending=False, na_position="last")
    if keep_date:
        merged["_date_tmp"] = merged["_date_tmp"].dt.strftime("%d/%m/%Y").fillna("")
    else:
        merged = merged.drop(columns=["_date_tmp"])
    return merged.reset_index(drop=True)

def classify_od_og(col_name: str) -> str:
    """Classify a column as 'OD', 'OG', or 'other' based on its suffix."""
    name = col_name.strip().upper().replace("_", "").replace(" ", "")
    if name.endswith("OD"):
        return "OD"
    if name.endswith("OG"):
        return "OG"
    return "other"

def clean_row_items(row: pd.Series, exclude_cols: list = None) -> list:
    """Return (col, val) pairs, filtering empty values and excluded columns."""
    exclude = set(exclude_cols or [])
    return [
        (col, val) for col, val in row.items()
        if col not in exclude and not is_empty(val)
    ]

def kv_html(label: str, value) -> str:
    """Render a key-value pair as styled HTML."""
    return (
        f'<p class="field-label">{label}</p>'
        f'<p class="field-value">{value}</p>'
    )

# Rendering helpers
def render_consultation_row(row: pd.Series, exclude_cols: list = None):
    """Render consultation fields in a responsive 3-column grid (empty values omitted)."""
    items = clean_row_items(row, exclude_cols)
    if not items:
        st.caption("Aucune donnée disponible.")
        return
    chunk_size = 3
    for i in range(0, len(items), chunk_size):
        chunk = items[i : i + chunk_size]
        cols = st.columns(chunk_size)
        for j, (col_name, val) in enumerate(chunk):
            with cols[j]:
                st.markdown(kv_html(col_name, val), unsafe_allow_html=True)

def render_exam_row(row: pd.Series, exclude_cols: list = None):
    """
    Render a tKERATO or tREFRACTION row.
    OD columns in left, OG columns in right, others below.
    """
    items       = clean_row_items(row, exclude_cols)
    od_items    = [(k, v) for k, v in items if classify_od_og(k) == "OD"]
    og_items    = [(k, v) for k, v in items if classify_od_og(k) == "OG"]
    other_items = [(k, v) for k, v in items if classify_od_og(k) == "other"]
    if od_items or og_items:
        col_od, col_og = st.columns(2)
        with col_od:
            st.markdown(
                '<p class="eye-header" style="border-left:3px solid #3b82f6;">Oeil droit (OD)</p>',
                unsafe_allow_html=True,
            )
            for k, v in od_items:
                label = k.upper().rstrip("OD").rstrip("_").strip() or k
                st.markdown(kv_html(label, v), unsafe_allow_html=True)
        with col_og:
            st.markdown(
                '<p class="eye-header" style="border-left:3px solid #22c55e;">Oeil gauche (OG)</p>',
                unsafe_allow_html=True,
            )
            for k, v in og_items:
                label = k.upper().rstrip("OG").rstrip("_").strip() or k
                st.markdown(kv_html(label, v), unsafe_allow_html=True)
    if other_items:
        if od_items or og_items:
            st.divider()
        chunk_size = min(len(other_items), 4)
        cols = st.columns(chunk_size)
        for i, (k, v) in enumerate(other_items):
            with cols[i % chunk_size]:
                st.markdown(kv_html(k, v), unsafe_allow_html=True)
    if not od_items and not og_items and not other_items:
        st.caption("Aucune donnée disponible.")

# PDF generation (requires: pip install fpdf2)
def _register_fonts(pdf):
    """
    Try to load DejaVuSansCondensed (bundled with fpdf2) as 'DejaVu'.
    If not found, fallback to Helvetica and return False.
    Returns True if DejaVu loaded, False if fallback used.
    """
    try:
        from pathlib import Path
        import fpdf as _fpdf_pkg
        _FPDF_FONT_DIR = Path(_fpdf_pkg.__file__).parent / "fonts"
        styles = {
            "":   "DejaVuSansCondensed.ttf",
            "B":  "DejaVuSansCondensed-Bold.ttf",
            "I":  "DejaVuSansCondensed-Oblique.ttf",
            "BI": "DejaVuSansCondensed-BoldOblique.ttf",
        }
        for style, filename in styles.items():
            font_path = _FPDF_FONT_DIR / filename
            if not font_path.exists():
                raise FileNotFoundError(f"Font not found: {font_path}")
            pdf.add_font("DejaVu", style=style, fname=str(font_path), uni=True)
        return True
    except Exception:
        return False

def _pdf_section_title(pdf, title: str, font_main="DejaVu", font_bold="B"):
    pdf.set_font(font_main, font_bold, 11)
    pdf.set_fill_color(237, 242, 251)
    pdf.set_text_color(30, 64, 175)
    pdf.cell(0, 8, f"  {title}", ln=True, fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(1)

def _pdf_entry_header(pdf, label: str, font_main="DejaVu", font_bi="BI"):
    pdf.set_font(font_main, font_bi, 9)
    pdf.set_text_color(55, 65, 81)
    pdf.cell(0, 6, label, ln=True)
    pdf.set_text_color(0, 0, 0)

def _pdf_kv(pdf, key: str, val, font_main="DejaVu", font_bold="B"):
    pdf.set_font(font_main, font_bold, 8)
    
    # Replace problematic Unicode characters for PDF compatibility
    safe_key = (
        str(key)
        .replace("œ", "oe").replace("Œ", "OE")
        .replace("—", "-").replace("–", "-")
        .replace("’", "'").replace("‘", "'")
        .replace("“", '"').replace("”", '"')
        .replace("«", '"').replace("»", '"')
        .replace("…", "...")
    )
    safe_val = (
        str(val)
        .replace("œ", "oe").replace("Œ", "OE")
        .replace("—", "-").replace("–", "-")
        .replace("’", "'").replace("‘", "'")
        .replace("“", '"').replace("”", '"')
        .replace("«", '"').replace("»", '"')
        .replace("…", "...")
    )
    pdf.cell(58, 5, safe_key[:40], border=0)
    pdf.set_font(font_main, "", 8)
    pdf.multi_cell(0, 5, safe_val, border=0)
    pdf.set_x(pdf.l_margin)

def generate_pdf_bytes(
    record: dict,
    full_name: str,
    dob_str: str,
    patient_id: str,
) -> bytes:
    """Build a clinical PDF from a record dict; empty fields are silently omitted."""
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_margins(15, 15, 15)
    has_dejavu = _register_fonts(pdf)
    font_main = "DejaVu" if has_dejavu else "Helvetica"
    font_bold = "B"
    font_italic = "I"
    font_bi = "BI" if has_dejavu else "B"
    
    # PDF header
    pdf.set_font(font_main, font_bold, 15)
    pdf.set_text_color(17, 24, 39)
    pdf.cell(0, 9, "Dossier Medical - Ophtalmologie", ln=True, align="C")
    pdf.set_font(font_main, "", 9)
    pdf.set_text_color(75, 85, 99)
    pdf.cell(
        0, 5,
        f"Patient : {full_name}   |   Ne(e) le : {dob_str}   |   ID : {patient_id}",
        ln=True, align="C",
    )
    pdf.set_draw_color(200, 210, 230)
    pdf.line(15, pdf.get_y() + 2, 195, pdf.get_y() + 2)
    pdf.ln(5)
    pdf.set_text_color(0, 0, 0)
    
    # Consultations
    consult_df = record.get("Consultation")
    if consult_df is not None and not consult_df.empty:
        df = sort_by_date(consult_df.copy(), "Date") if "Date" in consult_df.columns else consult_df.copy()
        df = fmt_date_col(df, "Date")
        _pdf_section_title(pdf, "Consultations", font_main, font_bold)
        for _, row in df.iterrows():
            date_val = row.get("Date", "-")
            subtitle = next(
                (str(row[c]) for c in ("DOMINANTE", "Motif", "Motif consultation")
                 if c in row.index and not is_empty(row[c])),
                "",
            )
            _pdf_entry_header(pdf, f"Consultation du {date_val}" + (f"  -  {subtitle}" if subtitle else ""), font_main, font_bi)
            for col, val in row.items():
                if col == "Date" or is_empty(val):
                    continue
                _pdf_kv(pdf, col, val, font_main, font_bold)
            pdf.ln(3)
    
    # Keratometry
    kerato_df = record.get("tKERATO")
    if kerato_df is not None and not kerato_df.empty:
        _pdf_section_title(pdf, "Keratomètrie", font_main, font_bold)
        for _, row in kerato_df.iterrows():
            _pdf_entry_header(pdf, f"Examen - Consultation n° {row.get('NumConsult', '-')}" , font_main, font_bi)
            for col, val in row.items():
                if is_empty(val):
                    continue
                _pdf_kv(pdf, col, val, font_main, font_bold)
            pdf.ln(3)
    
    # Refraction
    refrac_df = record.get("tREFRACTION")
    if refrac_df is not None and not refrac_df.empty:
        _pdf_section_title(pdf, "Refraction", font_main, font_bold)
        for _, row in refrac_df.iterrows():
            _pdf_entry_header(pdf, f"Examen - Consultation n° {row.get('NumConsult', '-')}" , font_main, font_bi)
            for col, val in row.items():
                if is_empty(val):
                    continue
                _pdf_kv(pdf, col, val, font_main, font_bold)
            pdf.ln(3)
    
    # Documents
    docs_df = record.get("Documents")
    if docs_df is not None and not docs_df.empty:
        df = sort_by_date(docs_df.copy(), "Date") if "Date" in docs_df.columns else docs_df.copy()
        df = fmt_date_col(df, "Date")
        _pdf_section_title(pdf, "Documents", font_main, font_bold)
        for _, row in df.iterrows():
            date_val = row.get("Date", "-")
            desc = row.get("DESCRIPTIONS", "") if not is_empty(row.get("DESCRIPTIONS")) else ""
            _pdf_entry_header(pdf, f"Document du {date_val}" + (f"  -  {desc}" if desc else ""), font_main, font_bi)
            for col, val in row.items():
                if col in ("Date", "DESCRIPTIONS") or is_empty(val):
                    continue
                _pdf_kv(pdf, col, val, font_main, font_bold)
            pdf.ln(3)
    return bytes(pdf.output())

# Data initialization
@st.cache_resource(show_spinner="Chargement des données…")
def init_data() -> dict:
    return load_all_data("data_raw")

all_data = init_data()

# Sidebar
with st.sidebar:
    st.markdown("### Recherche patient")
    st.divider()
    search_query = st.text_input(
        "Nom et/ou Prénom du patient",
        placeholder="Ex. : Dupont Jean",
        label_visibility="collapsed",
    )
    st.caption("Vous pouvez chercher par nom, prénom, ou les deux. L'ordre n'a pas d'importance.")

# Main content
if not search_query:
    st.markdown("## Dossier Patient")
    st.info("Saisissez un nom dans la barre latérale pour afficher le dossier médical.")
    st.stop()

with st.spinner("Recherche en cours…"):
    record = get_full_patient_record(all_data, search_query)

if record is None:
    st.error(f"Aucun patient trouvé pour la recherche « {search_query} ».")
    st.stop()

# Identity header
id_df = record.get("identity")
full_name, dob_str, patient_id = "-", "-", "-"
if id_df is not None and not id_df.empty:
    row       = id_df.iloc[0]
    full_name = f"{row.get('NOM', '')} {row.get('Prénom', '')}".strip() or "-"
    dob       = row.get("Date Naissance", None)
    dob_str   = (
        pd.to_datetime(dob, errors="coerce").strftime("%d/%m/%Y")
        if not is_empty(dob) else "-"
    )
    patient_id = str(row.get("Code patient", "-"))

st.markdown(f"## {full_name}")
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Identifiant", patient_id)
col_b.metric("Date de naissance", dob_str)
col_c.metric("Consultations", len(record.get("Consultation", pd.DataFrame())))
col_d.metric("Documents", len(record.get("Documents", pd.DataFrame())))

# Full dossier PDF button
if PDF_AVAILABLE:
    st.divider()
    if st.button("Générer le dossier complet (PDF)", type="secondary"):
        with st.spinner("Génération du PDF…"):
            pdf_bytes = generate_pdf_bytes(record, full_name, dob_str, patient_id)
        st.download_button(
            label="Télécharger le dossier complet",
            data=pdf_bytes,
            file_name=f"dossier_{full_name.replace(' ', '_')}.pdf",
            mime="application/pdf",
            key="dl_full_dossier",
        )

st.divider()

# Tabs
tab_consult, tab_exams, tab_docs = st.tabs([
    "Consultations", "Examens techniques", "Documents"
])

# Consultations tab
with tab_consult:
    consult_df = record.get("Consultation")
    if consult_df is None or consult_df.empty:
        st.info("Aucune consultation enregistrée pour ce patient.")
    else:
        if "Date" in consult_df.columns:
            consult_df = sort_by_date(consult_df, "Date")
            consult_df = fmt_date_col(consult_df, "Date")
        st.caption(f"{len(consult_df)} consultation(s) - ordre chronologique décroissant")
        for idx, row in consult_df.iterrows():
            date_val    = row.get("Date", "")
            date_label  = date_val if not is_empty(date_val) else "Date inconnue"
            subtitle    = next(
                (str(row[c]) for c in ("DOMINANTE", "Motif", "Motif consultation", "Acte", "Type")
                 if c in row.index and not is_empty(row[c])),
                "",
            )
            expander_title = f"Consultation du {date_label}" + (f"  -  {subtitle}" if subtitle else "")
            with st.expander(expander_title, expanded=False):
                if PDF_AVAILABLE:
                    consult_id = row.get("N° consultation")
                    if pd.isna(consult_id):
                        consult_id = row.get("NumConsult")
                    single = {"Consultation": pd.DataFrame([row])}
                    if not pd.isna(consult_id):
                        kerato_df = record.get("tKERATO")
                        if kerato_df is not None and not kerato_df.empty:
                            single["tKERATO"] = kerato_df[kerato_df["NumConsult"].astype(str) == str(consult_id)]
                        refrac_df = record.get("tREFRACTION")
                        if refrac_df is not None and not refrac_df.empty:
                            single["tREFRACTION"] = refrac_df[refrac_df["NumConsult"].astype(str) == str(consult_id)]
                    docs_df = record.get("Documents")
                    if docs_df is not None and not docs_df.empty:
                        if "NumConsult" in docs_df.columns and not pd.isna(consult_id):
                            single["Documents"] = docs_df[docs_df["NumConsult"].astype(str) == str(consult_id)]
                        elif "N° consultation" in docs_df.columns and not pd.isna(consult_id):
                            single["Documents"] = docs_df[docs_df["N° consultation"].astype(str) == str(consult_id)]
                        else:
                            single["Documents"] = docs_df[docs_df["Date"] == row.get("Date")]
                    pdf_c = generate_pdf_bytes(single, full_name, dob_str, patient_id)
                    st.download_button(
                        label="Télécharger cette consultation (PDF)",
                        data=pdf_c,
                        file_name=f"consultation_{date_label.replace('/', '-')}_{full_name.replace(' ', '_')}.pdf",
                        mime="application/pdf",
                        key=f"dl_consult_{idx}",
                    )
                    st.divider()
                render_consultation_row(row, exclude_cols=["Date"])

# Exams tab
with tab_exams:
    consult_raw = record.get("Consultation")
    for section_label, section_key in [
        ("Kératométrie", "tKERATO"),
        ("Réfraction",   "tREFRACTION"),
    ]:
        st.markdown(f"### {section_label}")
        exam_df = record.get(section_key)
        if exam_df is None or exam_df.empty:
            st.info(f"Aucune donnée de {section_label.lower()} pour ce patient.")
        else:
            if consult_raw is not None and not consult_raw.empty:
                exam_df = sort_exams_via_consult(exam_df, consult_raw, keep_date=True)
            st.caption(f"{len(exam_df)} enregistrement(s) - ordre chronologique décroissant")
            for idx, row in exam_df.iterrows():
                date_val    = row.get("_date_tmp", "")
                num_consult = row.get("NumConsult", "-")
                date_label  = f"du {date_val}" if not is_empty(date_val) else f"- Consultation n° {num_consult}"
                expander_title = f"{section_label} {date_label}"
                with st.expander(expander_title, expanded=False):
                    render_exam_row(row, exclude_cols=["_date_tmp"])
        st.divider()

# Documents tab
with tab_docs:
    docs_df = record.get("Documents")
    if docs_df is None or docs_df.empty:
        st.info("Aucun document associé à ce patient.")
    else:
        if "Date" in docs_df.columns:
            docs_df = sort_by_date(docs_df, "Date")
            docs_df = fmt_date_col(docs_df, "Date")
        st.caption(f"{len(docs_df)} document(s) - ordre chronologique décroissant")
        for idx, row in docs_df.iterrows():
            date_val   = row.get("Date", "")
            date_label = date_val if not is_empty(date_val) else "Date inconnue"
            desc       = next(
                (str(row[c]) for c in ("DESCRIPTIONS", "Type", "Libellé")
                 if c in row.index and not is_empty(row[c])),
                "",
            )
            expander_title = f"Document du {date_label}" + (f"  -  {desc}" if desc else "")
            with st.expander(expander_title, expanded=False):
                items = clean_row_items(row, exclude_cols=["Date", "DESCRIPTIONS"])
                if not items:
                    st.caption("Aucune métadonnée supplémentaire disponible.")
                else:
                    chunk_size = 3
                    for i in range(0, len(items), chunk_size):
                        chunk = items[i : i + chunk_size]
                        cols  = st.columns(chunk_size)
                        for j, (col_name, val) in enumerate(chunk):
                            with cols[j]:
                                st.markdown(kv_html(col_name, val), unsafe_allow_html=True)