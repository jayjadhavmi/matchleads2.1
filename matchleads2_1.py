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

# Clean Uniform CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Sora', sans-serif;
    }
    
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #e0f2fe 0%, #ddd6fe 100%);
    }
    
    .main {
        background: transparent;
    }
    
    .block-container {
        max-width: 750px;
        padding: 2.5rem 1rem;
    }
    
    /* Hide Streamlit elements */
    #MainMenu, footer, header {visibility: hidden;}
    
    /* Header */
    .premium-header {
        font-size: 3rem;
        font-weight: 800;
        color: #1e3a5f !important;
        text-align: center;
        margin: 0 0 2.5rem 0;
        letter-spacing: -0.02em;
    }
    
    /* Upload sections */
    .upload-section {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
    }
    
    .section-title {
        font-size: 1.125rem;
        font-weight: 700;
        color: #1e3a5f !important;
        margin-bottom: 1.25rem;
        display: block;
    }
    
    /* File uploader unified background */
    [data-testid="stFileUploader"] {
        background: white !important;
        border: 2px dashed #22d3ee;
        border-radius: 16px;
        padding: 2.5rem 2rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #06b6d4;
        background: #f0fdff !important;
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
    
    .upload-text {
        color: #64748b !important;
        font-size: 1rem;
        font-weight: 500;
        text-align: center;
        margin-bottom: 0.75rem;
    }
    
    /* Browse button */
    [data-testid="stFileUploader"] button {
        background: #22d3ee !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        margin: 0 auto !important;
        display: block !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stFileUploader"] button:hover {
        background: #06b6d4 !important;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(34, 211, 238, 0.4);
    }
    
    /* File uploader text color fix - CRITICAL */
    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] div,
    [data-testid="stFileUploader"] small {
        color: #1e3a5f !important;
        background: transparent !important;
    }
    
    /* File uploader inner content area */
    [data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] {
        color: #1e3a5f !important;
    }
    
    /* Drag and drop text */
    [data-testid="stFileUploader"] [role="button"] {
        color: #1e3a5f !important;
    }
    
    /* Select boxes */
    .stSelectbox, .stMultiSelect {
        margin-bottom: 1rem;
    }
    
    .stSelectbox label, .stMultiSelect label {
        font-size: 0.95rem;
        font-weight: 600;
        color: #1e3a5f !important;
        margin-bottom: 0.5rem;
    }
    
    [data-baseweb="select"] > div {
        background-color: white !important;
        border: 2px solid #e2e8f0 !important;
        border-radius: 12px !important;
        color: #1e3a5f !important;
    }
    
    [data-baseweb="select"] > div:hover {
        border-color: #22d3ee !important;
    }
    
    /* Match button */
    .stButton > button {
        width: 100%;
        background: #22d3ee !important;
        color: white !important;
        border: none !important;
        padding: 1rem 2rem !important;
        font-size: 1.05rem !important;
        font-weight: 800 !important;
        border-radius: 14px !important;
        cursor: pointer;
        transition: all 0.3s ease;
        margin-top: 1.5rem;
        box-shadow: 0 6px 20px rgba(34, 211, 238, 0.3);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .stButton > button:hover {
        background: #06b6d4 !important;
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(34, 211, 238, 0.5);
    }
    
    /* Download buttons */
    .stDownloadButton > button {
        background: #10b981 !important;
        color: white !important;
        border: none !important;
        font-weight: 700 !important;
        padding: 0.875rem 2rem !important;
        border-radius: 12px !important;
        transition: all 0.3s ease;
        width: 100%;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
    }
    
    .stDownloadButton > button:hover {
        background: #059669 !important;
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.4);
    }
    
    /* Results section */
    .results-section {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 800;
        color: #1e3a5f;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.75rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    
    .stMetric {
        background: #f8fafc;
        padding: 1.25rem;
        border-radius: 14px;
        border: 2px solid #e2e8f0;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #22d3ee 0%, #06b6d4 100%);
        border-radius: 10px;
        height: 8px;
    }
    
    .stProgress > div {
        background: #e2e8f0;
        border-radius: 10px;
    }
    
    /* Info section */
    .info-section {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        margin-top: 2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
    }
    
    .info-title {
        font-size: 0.95rem;
        font-weight: 800;
        color: #1e3a5f !important;
        margin-bottom: 1.25rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    
    .match-types {
        display: grid;
        gap: 0.875rem;
    }
    
    .match-type {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 14px;
        text-align: center;
        border: 2px solid #e2e8f0;
        transition: all 0.3s ease;
    }
    
    .match-type:hover {
        border-color: #22d3ee;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(34, 211, 238, 0.2);
    }
    
    .match-type-name {
        font-size: 0.95rem;
        font-weight: 800;
        color: #1e3a5f !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.5rem;
    }
    
    .match-type-desc {
        font-size: 0.875rem;
        color: #64748b !important;
        line-height: 1.5;
        font-weight: 500;
    }
    
    /* Alerts */
    .stAlert {
        border-radius: 14px;
        background: white;
        border: 2px solid #e2e8f0;
        color: #1e3a5f !important;
        font-weight: 600;
    }
    
    .stSuccess {
        background: #d1fae5 !important;
        border-color: #10b981 !important;
        color: #065f46 !important;
    }
    
    .stWarning {
        background: #fed7aa !important;
        border-color: #f59e0b !important;
        color: #78350f !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: white;
        border-radius: 14px;
        font-weight: 700;
        color: #1e3a5f !important;
        border: 2px solid #e2e8f0;
        padding: 1rem 1.25rem;
    }
    
    .streamlit-expanderHeader:hover {
        background: #f8fafc;
        border-color: #22d3ee;
    }
    
    /* Dataframe */
    [data-testid="stDataFrame"] {
        border-radius: 14px;
        border: 2px solid #e2e8f0;
        overflow: hidden;
    }
    
    /* Force all text colors */
    h1, h2, h3, h4, h5, h6 {
        color: #1e3a5f !important;
    }
    
    p, span, div, label {
        color: #1e3a5f !important;
    }
    
    /* Override Streamlit's default dark mode colors */
    .stMarkdown, .stText {
        color: #1e3a5f !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #22d3ee !important;
    }
    
    /* Mobile responsive */
    @media (max-width: 768px) {
        .block-container {
            padding: 1.5rem 1rem;
        }
        
        .premium-header {
            font-size: 2.25rem;
            margin-bottom: 2rem;
        }
        
        .upload-section {
            padding: 1.5rem;
            margin-bottom: 1rem;
        }
        
        .section-title {
            font-size: 1rem;
        }
        
        [data-testid="stFileUploader"] {
            padding: 2rem 1.5rem;
        }
        
        .info-section, .results-section {
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
    
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown('<span class="section-title">Database File</span>', unsafe_allow_html=True)
    file1 = st.file_uploader("db", type=['xlsx', 'xls'], label_visibility="collapsed", key="file1")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown('<span class="section-title">Input File</span>', unsafe_allow_html=True)
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
            
            if st.button("Run Match", use_container_width=True):
                progress_bar = st.progress(0)
                
                try:
                    results_df = process_matching(
                        df1.copy(), df2.copy(),
                        match_col1, match_col2,
                        return_columns, progress_bar
                    )
                    
                    progress_bar.empty()
                    st.success("✓ Matching Complete")
                    
                    st.markdown('<div class="results-section">', unsafe_allow_html=True)
                    
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
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
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
                    <div class="match-type-desc">Same base, different extension</div>
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
