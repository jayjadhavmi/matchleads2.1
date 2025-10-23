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

# Premium CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main {
        background: #fafafa;
    }
    
    .block-container {
        max-width: 680px;
        padding: 3rem 1rem 2rem 1rem;
    }
    
    /* Hide default elements */
    #MainMenu, footer, header {visibility: hidden;}
    
    /* Header */
    .premium-header {
        font-size: 2.75rem;
        font-weight: 800;
        color: #0a0a0a;
        text-align: center;
        margin-bottom: 3rem;
        letter-spacing: -0.03em;
    }
    
    /* Upload section */
    .upload-container {
        background: white;
        border-radius: 24px;
        padding: 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        border: 1px solid #f0f0f0;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: #fafafa;
        border: 2px dashed #e0e0e0;
        border-radius: 16px;
        padding: 2rem 1.5rem;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #0a0a0a;
        background: #f5f5f5;
    }
    
    [data-testid="stFileUploader"] section {
        border: none;
        padding: 0;
    }
    
    [data-testid="stFileUploader"] section > div {
        background: transparent;
    }
    
    /* Labels */
    .stFileUploader label {
        font-size: 0.9rem;
        font-weight: 600;
        color: #0a0a0a;
        letter-spacing: -0.01em;
    }
    
    /* Select boxes */
    [data-baseweb="select"] {
        border-radius: 12px;
    }
    
    .stSelectbox label, .stMultiSelect label {
        font-size: 0.875rem;
        font-weight: 600;
        color: #0a0a0a;
        margin-bottom: 0.5rem;
    }
    
    /* Primary button */
    .stButton > button {
        width: 100%;
        background: #0a0a0a;
        color: white;
        border: none;
        padding: 1rem 2rem;
        font-size: 0.95rem;
        font-weight: 600;
        border-radius: 14px;
        cursor: pointer;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        letter-spacing: -0.01em;
        margin-top: 1.5rem;
    }
    
    .stButton > button:hover {
        background: #1a1a1a;
        transform: translateY(-1px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.12);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Download buttons */
    .stDownloadButton > button {
        background: white;
        color: #0a0a0a;
        border: 1.5px solid #e0e0e0;
        font-weight: 600;
        padding: 0.875rem 2rem;
        border-radius: 12px;
        transition: all 0.25s ease;
    }
    
    .stDownloadButton > button:hover {
        background: #fafafa;
        border-color: #0a0a0a;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #0a0a0a;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.8rem;
        font-weight: 600;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .stMetric {
        background: white;
        padding: 1.25rem;
        border-radius: 16px;
        border: 1px solid #f0f0f0;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: #0a0a0a;
        border-radius: 10px;
        height: 6px;
    }
    
    .stProgress > div {
        background: #f0f0f0;
        border-radius: 10px;
    }
    
    /* Dataframe */
    [data-testid="stDataFrame"] {
        border-radius: 16px;
        border: 1px solid #f0f0f0;
    }
    
    /* Alerts */
    .stAlert {
        border-radius: 14px;
        border: none;
        padding: 1rem 1.25rem;
        background: white;
        border: 1px solid #f0f0f0;
    }
    
    /* Info section */
    .info-section {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        margin-top: 2rem;
        border: 1px solid #f0f0f0;
    }
    
    .info-title {
        font-size: 0.875rem;
        font-weight: 700;
        color: #0a0a0a;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .match-types {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .match-type {
        background: #fafafa;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #f0f0f0;
    }
    
    .match-type-name {
        font-size: 0.75rem;
        font-weight: 700;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.25rem;
    }
    
    .match-type-desc {
        font-size: 0.8rem;
        color: #999;
        line-height: 1.4;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: white;
        border-radius: 14px;
        font-weight: 600;
        color: #0a0a0a;
        border: 1px solid #f0f0f0;
        padding: 1rem 1.25rem;
    }
    
    .streamlit-expanderHeader:hover {
        background: #fafafa;
        border-color: #e0e0e0;
    }
    
    /* Mobile */
    @media (max-width: 768px) {
        .block-container {
            padding: 2rem 1rem;
        }
        
        .premium-header {
            font-size: 2rem;
            margin-bottom: 2rem;
        }
        
        .upload-container {
            padding: 1.5rem;
        }
        
        .info-section {
            padding: 1.5rem;
        }
        
        .match-types {
            grid-template-columns: 1fr;
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
    
    st.markdown('<div class="upload-container">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        file1 = st.file_uploader("Database File", type=['xlsx', 'xls'], label_visibility="visible")
    
    with col2:
        file2 = st.file_uploader("Input File", type=['xlsx', 'xls'], label_visibility="visible")
    
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
            <div class="info-title">How it works</div>
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
