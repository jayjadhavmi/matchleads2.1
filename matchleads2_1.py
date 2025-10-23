import streamlit as st
import pandas as pd
from io import BytesIO
import torch

# Try importing optional dependencies
try:
    from rapidfuzz import fuzz
    HAVE_RAPIDFUZZ = True
except ImportError:
    try:
        from fuzzywuzzy import fuzz
        HAVE_RAPIDFUZZ = True
    except ImportError:
        HAVE_RAPIDFUZZ = False

try:
    from sentence_transformers import SentenceTransformer, util
    HAVE_TRANSFORMERS = True
except ImportError:
    HAVE_TRANSFORMERS = False

# Page configuration
st.set_page_config(
    page_title="Match Data",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Premium Dark CSS - Exact match to reference
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, sans-serif;
    }
    
    /* Force dark theme */
    .stApp {
        background-color: #0f1419 !important;
    }
    
    .main {
        background-color: #0f1419 !important;
    }
    
    .block-container {
        max-width: 700px;
        padding: 2rem 1rem;
        background-color: #0f1419 !important;
    }
    
    /* Hide Streamlit elements */
    #MainMenu, footer, header {visibility: hidden;}
    
    /* Header */
    .premium-header {
        font-size: 3.5rem;
        font-weight: 700;
        color: #5eead4;
        text-align: center;
        margin: 2rem 0 2.5rem 0;
        letter-spacing: -0.02em;
    }
    
    /* Upload cards */
    .upload-card {
        background: #f5f5f5;
        border-radius: 24px;
        padding: 2rem 2rem 2.5rem 2rem;
        margin-bottom: 1.5rem;
    }
    
    .card-label {
        font-size: 1.125rem;
        font-weight: 600;
        color: #1a1a1a;
        margin-bottom: 1rem;
        display: block;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: #2a2e35 !important;
        border: 2px dashed #4a5568;
        border-radius: 16px;
        padding: 2.5rem 2rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #718096;
        background: #3a3f47 !important;
    }
    
    [data-testid="stFileUploader"] section {
        border: none !important;
        padding: 0 !important;
        background: transparent !important;
    }
    
    [data-testid="stFileUploader"] section > div {
        background: transparent !important;
    }
    
    [data-testid="stFileUploader"] label {
        display: none !important;
    }
    
    [data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] {
        color: #e0e0e0 !important;
    }
    
    .upload-text {
        color: #e0e0e0;
        font-size: 1.125rem;
        font-weight: 500;
        text-align: center;
        margin-bottom: 0.75rem;
    }
    
    .upload-subtext {
        color: #9ca3af;
        font-size: 0.875rem;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    /* Browse button inside uploader */
    [data-testid="stFileUploader"] button {
        background: #2a2e35 !important;
        color: #e0e0e0 !important;
        border: 1px solid #4a5568 !important;
        border-radius: 10px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        margin: 0 auto !important;
        display: block !important;
    }
    
    [data-testid="stFileUploader"] button:hover {
        background: #3a3f47 !important;
        border-color: #718096 !important;
    }
    
    /* Select boxes */
    .stSelectbox, .stMultiSelect {
        margin-bottom: 1rem;
    }
    
    .stSelectbox label, .stMultiSelect label {
        font-size: 1rem;
        font-weight: 600;
        color: #e0e0e0;
        margin-bottom: 0.5rem;
    }
    
    [data-baseweb="select"] > div {
        background-color: #2a2e35 !important;
        border-color: #4a5568 !important;
        color: #e0e0e0 !important;
        border-radius: 10px !important;
    }
    
    [data-baseweb="select"] input {
        color: #e0e0e0 !important;
    }
    
    [data-baseweb="select"] svg {
        fill: #9ca3af !important;
    }
    
    /* Match button */
    .stButton > button {
        width: 100%;
        background: #5eead4 !important;
        color: #0f1419 !important;
        border: none !important;
        padding: 1rem 2rem !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        border-radius: 12px !important;
        cursor: pointer;
        transition: all 0.3s ease;
        margin-top: 1.5rem;
    }
    
    .stButton > button:hover {
        background: #2dd4bf !important;
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(94, 234, 212, 0.3);
    }
    
    /* Download buttons */
    .stDownloadButton > button {
        background: #22c55e !important;
        color: white !important;
        border: none !important;
        font-weight: 600 !important;
        padding: 0.875rem 2rem !important;
        border-radius: 12px !important;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stDownloadButton > button:hover {
        background: #16a34a !important;
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(34, 197, 94, 0.3);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #5eead4;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.75rem;
        font-weight: 600;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .stMetric {
        background: #1a1f26;
        padding: 1.25rem;
        border-radius: 14px;
        border: 1px solid #2a2e35;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #5eead4 0%, #2dd4bf 100%);
        border-radius: 10px;
        height: 8px;
    }
    
    .stProgress > div {
        background: #2a2e35;
        border-radius: 10px;
    }
    
    /* Info section */
    .info-section {
        background: #f5f5f5;
        border-radius: 24px;
        padding: 2rem;
        margin-top: 2rem;
    }
    
    .info-title {
        font-size: 1rem;
        font-weight: 700;
        color: #1a1a1a;
        margin-bottom: 1.25rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .match-types {
        display: grid;
        gap: 1rem;
    }
    
    .match-type {
        background: #fafafa;
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        border: 1px solid #e5e5e5;
    }
    
    .match-type-name {
        font-size: 0.95rem;
        font-weight: 700;
        color: #1a1a1a;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    
    .match-type-desc {
        font-size: 0.875rem;
        color: #737373;
        line-height: 1.5;
    }
    
    /* Alerts */
    .stAlert {
        border-radius: 12px;
        background: #1a1f26;
        border: 1px solid #2a2e35;
        color: #e0e0e0;
    }
    
    .stSuccess {
        background: #14532d !important;
        border-color: #16a34a !important;
        color: #bbf7d0 !important;
    }
    
    .stWarning {
        background: #431407 !important;
        border-color: #ea580c !important;
        color: #fed7aa !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #1a1f26;
        border-radius: 12px;
        font-weight: 600;
        color: #e0e0e0;
        border: 1px solid #2a2e35;
        padding: 1rem 1.25rem;
    }
    
    .streamlit-expanderHeader:hover {
        background: #2a2e35;
        border-color: #4a5568;
    }
    
    /* Dataframe */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        border: 1px solid #2a2e35;
    }
    
    /* Text color fixes */
    p, span, div {
        color: #e0e0e0;
    }
    
    /* Mobile responsive */
    @media (max-width: 768px) {
        .block-container {
            padding: 1.5rem 1rem;
        }
        
        .premium-header {
            font-size: 2.5rem;
            margin: 1.5rem 0 2rem 0;
        }
        
        .upload-card {
            padding: 1.5rem;
            margin-bottom: 1rem;
        }
        
        .card-label {
            font-size: 1rem;
        }
        
        [data-testid="stFileUploader"] {
            padding: 2rem 1.5rem;
        }
        
        .info-section {
            padding: 1.5rem;
        }
        
        .match-type {
            padding: 1.25rem;
        }
    }
    </style>
""", unsafe_allow_html=True)

def get_base_domain(domain):
    """Extract the base domain (without TLD)"""
    if not isinstance(domain, str):
        return str(domain)
    try:
        parts = domain.split('.')
        if len(parts) > 1:
            return parts[0]
        return domain
    except:
        return domain

@st.cache_resource
def load_ai_model():
    """Load the sentence transformer model (cached)"""
    if HAVE_TRANSFORMERS:
        try:
            return SentenceTransformer('all-MiniLM-L6-v2')
        except:
            return None
    return None

def process_matching(df1, df2, match_col1, match_col2, return_columns, progress_bar=None):
    """Main matching function with exact, TLD, fuzzy, and AI matching"""
    # Standardize columns
    df1.columns = df1.columns.str.strip().str.lower()
    df2.columns = df2.columns.str.strip().str.lower()
    
    match_col1 = match_col1.strip().lower()
    match_col2 = match_col2.strip().lower()
    return_columns = [col.strip().lower() for col in return_columns]
    
    # Clean data
    df1[match_col1] = df1[match_col1].astype(str).str.strip().str.lower()
    df2[match_col2] = df2[match_col2].astype(str).str.strip().str.lower()
    
    # Create mapping
    domains1 = df1[match_col1].tolist()
    dict1 = {}
    for _, row in df1.iterrows():
        domain = row[match_col1]
        dict1[domain] = {col: row[col] for col in return_columns if col in df1.columns}
    
    # Base domains for TLD matching
    base_domains = {}
    for domain in domains1:
        base = get_base_domain(domain)
        if base not in base_domains:
            base_domains[base] = []
        base_domains[base].append(domain)
    
    # Load AI model
    model = None
    if HAVE_TRANSFORMERS:
        model = load_ai_model()
        if model:
            embeddings1 = model.encode(domains1, convert_to_tensor=True, show_progress_bar=False)
    
    # Process
    input_domains = df2[match_col2].dropna().tolist()
    results = []
    total = len(input_domains)
    
    for idx, domain in enumerate(input_domains):
        if progress_bar:
            progress_bar.progress((idx + 1) / total)
        
        result_row = {
            'Input': domain,
            'Match Type': 'No Match',
            'Matched': '',
            'Score': ''
        }
        
        # Exact
        if domain in dict1:
            result_row['Match Type'] = 'Exact'
            result_row['Matched'] = domain
            result_row['Score'] = '100%'
            for col in return_columns:
                result_row[col] = dict1[domain].get(col, '')
        
        # TLD
        elif result_row['Match Type'] == 'No Match':
            base = get_base_domain(domain)
            if base in base_domains:
                tld_matches = [d for d in base_domains[base] if d != domain]
                if tld_matches:
                    matched = tld_matches[0]
                    result_row['Match Type'] = 'TLD'
                    result_row['Matched'] = matched
                    result_row['Score'] = '95%'
                    for col in return_columns:
                        result_row[col] = dict1[matched].get(col, '')
        
        # Fuzzy + AI
        if result_row['Match Type'] == 'No Match' and HAVE_RAPIDFUZZ:
            fuzzy_matches = []
            for d in domains1:
                ratio = fuzz.ratio(domain, d)
                if ratio > 85 and d != domain:
                    fuzzy_matches.append((d, ratio))
            
            if fuzzy_matches:
                fuzzy_matches.sort(key=lambda x: x[1], reverse=True)
                
                if model and HAVE_TRANSFORMERS:
                    try:
                        candidates = [m[0] for m in fuzzy_matches[:10]]
                        input_emb = model.encode(domain, convert_to_tensor=True, show_progress_bar=False)
                        candidate_embs = model.encode(candidates, convert_to_tensor=True, show_progress_bar=False)
                        scores = util.cos_sim(input_emb, candidate_embs)[0]
                        best_idx = torch.argmax(scores).item()
                        best_score = scores[best_idx].item()
                        
                        if best_score >= 0.85:
                            matched = candidates[best_idx]
                            result_row['Match Type'] = 'AI'
                            result_row['Matched'] = matched
                            result_row['Score'] = f'{round(best_score * 100, 1)}%'
                            for col in return_columns:
                                result_row[col] = dict1[matched].get(col, '')
                    except:
                        matched = fuzzy_matches[0][0]
                        result_row['Match Type'] = 'Fuzzy'
                        result_row['Matched'] = matched
                        result_row['Score'] = f'{fuzzy_matches[0][1]}%'
                        for col in return_columns:
                            result_row[col] = dict1[matched].get(col, '')
                else:
                    matched = fuzzy_matches[0][0]
                    result_row['Match Type'] = 'Fuzzy'
                    result_row['Matched'] = matched
                    result_row['Score'] = f'{fuzzy_matches[0][1]}%'
                    for col in return_columns:
                        result_row[col] = dict1[matched].get(col, '')
        
        if result_row['Match Type'] == 'No Match':
            for col in return_columns:
                result_row[col] = ''
        
        results.append(result_row)
    
    return pd.DataFrame(results)

def main():
    st.markdown('<h1 class="premium-header">Match Data</h1>', unsafe_allow_html=True)
    
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    st.markdown('<span class="card-label">Database File</span>', unsafe_allow_html=True)
    file1 = st.file_uploader("db", type=['xlsx', 'xls'], label_visibility="collapsed", key="file1")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    st.markdown('<span class="card-label">Input File</span>', unsafe_allow_html=True)
    file2 = st.file_uploader("input", type=['xlsx', 'xls'], label_visibility="collapsed", key="file2")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if file1 and file2:
        try:
            with st.spinner("Loading..."):
                df1 = pd.read_excel(file1)
                df2 = pd.read_excel(file2)
            
            col1, col2 = st.columns(2)
            
            with col1:
                match_col1 = st.selectbox("Match column (Database)", df1.columns.tolist())
            
            with col2:
                match_col2 = st.selectbox("Match column (Input)", df2.columns.tolist())
            
            return_columns = st.multiselect(
                "Return columns",
                df1.columns.tolist(),
                default=[col for col in df1.columns.tolist()[:3]]
            )
            
            if not return_columns:
                st.warning("Select at least one return column")
                return
            
            if st.button("Match", use_container_width=True):
                progress_bar = st.progress(0)
                
                try:
                    results_df = process_matching(
                        df1.copy(), df2.copy(),
                        match_col1, match_col2,
                        return_columns, progress_bar
                    )
                    
                    progress_bar.empty()
                    st.success("Done")
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        st.metric("Total", len(results_df))
                    with col2:
                        st.metric("Exact", len(results_df[results_df['Match Type'] == 'Exact']))
                    with col3:
                        st.metric("TLD", len(results_df[results_df['Match Type'] == 'TLD']))
                    with col4:
                        st.metric("Fuzzy", len(results_df[results_df['Match Type'].isin(['Fuzzy', 'AI'])]))
                    with col5:
                        st.metric("None", len(results_df[results_df['Match Type'] == 'No Match']))
                    
                    with st.expander("View Results", expanded=True):
                        st.dataframe(results_df, use_container_width=True, height=400)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        output = BytesIO()
                        results_df.to_excel(output, index=False, engine='openpyxl')
                        output.seek(0)
                        st.download_button("Download Excel", output, "results.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True)
                    
                    with col2:
                        csv = results_df.to_csv(index=False)
                        st.download_button("Download CSV", csv, "results.csv", "text/csv",
                            use_container_width=True)
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        except Exception as e:
            st.error(f"Error loading files: {str(e)}")
    
    else:
        st.markdown("""
        <div class="info-section">
            <div class="info-title">How It Works</div>
            <div class="match-types">
                <div class="match-type">
                    <div class="match-type-name">Exact</div>
                    <div class="match-type-desc">Perfect matches</div>
                </div>
                <div class="match-type">
                    <div class="match-type-name">TLD</div>
                    <div class="match-type-desc">Same base, diff extension</div>
                </div>
                <div class="match-type">
                    <div class="match-type-name">Fuzzy</div>
                    <div class="match-type-desc">Similar strings</div>
                </div>
                <div class="match-type">
                    <div class="match-type-name">AI</div>
                    <div class="match-type-desc">Smart matching</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
