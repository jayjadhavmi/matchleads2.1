import streamlit as st
import pandas as pd
from io import BytesIO
import torch
from tqdm import tqdm

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
    page_title="Match Leads",
    page_icon="🎯",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS - Modern, clean, mobile-friendly design
st.markdown("""
    <style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main container */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
    }
    
    .block-container {
        max-width: 900px;
        padding: 2rem 1rem;
    }
    
    /* Header */
    .main-header {
        font-size: clamp(2rem, 5vw, 3rem);
        font-weight: 700;
        color: #1a202c;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: clamp(0.9rem, 2vw, 1.1rem);
        margin-bottom: 2.5rem;
        font-weight: 400;
    }
    
    /* Card styling */
    .upload-card, .config-card, .results-card {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
        border: 1px solid #e2e8f0;
    }
    
    .card-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1a202c;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* File uploader styling */
    .stFileUploader {
        border: 2px dashed #cbd5e1;
        border-radius: 12px;
        padding: 1.5rem;
        background: #f8fafc;
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: #94a3b8;
        background: #f1f5f9;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.875rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 12px;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(102, 126, 234, 0.25);
        letter-spacing: 0.02em;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.35);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Download buttons */
    .stDownloadButton > button {
        background: white;
        color: #667eea;
        border: 2px solid #667eea;
        font-weight: 600;
    }
    
    .stDownloadButton > button:hover {
        background: #667eea;
        color: white;
    }
    
    /* Select boxes */
    .stSelectbox, .stMultiSelect {
        border-radius: 8px;
    }
    
    /* Metrics */
    .stMetric {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
    }
    
    .stMetric label {
        color: #64748b;
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: #1a202c;
        font-size: 1.875rem;
        font-weight: 700;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Dataframe */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #e2e8f0;
    }
    
    /* Info/Success messages */
    .stAlert {
        border-radius: 12px;
        border: none;
        padding: 1rem 1.25rem;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #f8fafc;
        border-radius: 8px;
        font-weight: 500;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Mobile responsive */
    @media (max-width: 768px) {
        .block-container {
            padding: 1rem 0.5rem;
        }
        
        .upload-card, .config-card, .results-card {
            padding: 1.25rem;
        }
        
        .main-header {
            margin-bottom: 0.25rem;
        }
        
        .subtitle {
            margin-bottom: 1.5rem;
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
        except Exception as e:
            st.error(f"Error loading AI model: {str(e)}")
            return None
    return None

def process_matching(sfdc_df, input_df, match_col_sfdc, match_col_input, return_columns, progress_bar=None):
    """
    Main matching function with exact, TLD, fuzzy, and AI matching
    """
    # Standardize columns
    sfdc_df.columns = sfdc_df.columns.str.strip().str.lower()
    input_df.columns = input_df.columns.str.strip().str.lower()
    
    match_col_sfdc = match_col_sfdc.strip().lower()
    match_col_input = match_col_input.strip().lower()
    return_columns = [col.strip().lower() for col in return_columns]
    
    # Clean and prepare data
    sfdc_df[match_col_sfdc] = sfdc_df[match_col_sfdc].astype(str).str.strip().str.lower()
    input_df[match_col_input] = input_df[match_col_input].astype(str).str.strip().str.lower()
    
    # Create mapping dictionary for SFDC data
    sfdc_domains = sfdc_df[match_col_sfdc].tolist()
    sfdc_dict = {}
    for _, row in sfdc_df.iterrows():
        domain = row[match_col_sfdc]
        sfdc_dict[domain] = {col: row[col] for col in return_columns if col in sfdc_df.columns}
    
    # Create base domain mapping for TLD matching
    sfdc_base_domains = {}
    for domain in sfdc_domains:
        base = get_base_domain(domain)
        if base not in sfdc_base_domains:
            sfdc_base_domains[base] = []
        sfdc_base_domains[base].append(domain)
    
    # Load AI model if available
    model = None
    if HAVE_TRANSFORMERS:
        model = load_ai_model()
        if model:
            # Pre-encode SFDC domains for efficiency
            sfdc_embeddings = model.encode(sfdc_domains, convert_to_tensor=True, show_progress_bar=False)
    
    # Get input domains
    input_domains = input_df[match_col_input].dropna().tolist()
    
    # Process each input domain
    results = []
    total = len(input_domains)
    
    for idx, domain in enumerate(input_domains):
        if progress_bar:
            progress_bar.progress((idx + 1) / total)
        
        # Initialize result row
        result_row = {
            'Input Domain': domain,
            'Match Type': 'No Match',
            'Matched Domain': '',
            'Match Score': ''
        }
        
        # 1. EXACT MATCH
        if domain in sfdc_dict:
            result_row['Match Type'] = 'Exact Match'
            result_row['Matched Domain'] = domain
            result_row['Match Score'] = '100%'
            for col in return_columns:
                result_row[col] = sfdc_dict[domain].get(col, '')
        
        # 2. TLD MATCH (same base domain, different extension)
        elif result_row['Match Type'] == 'No Match':
            input_base = get_base_domain(domain)
            if input_base in sfdc_base_domains:
                # Find matches with same base but different TLD
                tld_matches = [d for d in sfdc_base_domains[input_base] if d != domain]
                if tld_matches:
                    matched_domain = tld_matches[0]
                    result_row['Match Type'] = 'TLD Match'
                    result_row['Matched Domain'] = matched_domain
                    result_row['Match Score'] = '95%'
                    for col in return_columns:
                        result_row[col] = sfdc_dict[matched_domain].get(col, '')
        
        # 3. FUZZY + AI MATCH
        if result_row['Match Type'] == 'No Match' and HAVE_RAPIDFUZZ:
            # Calculate fuzzy matches
            fuzzy_matches = []
            for sfdc_domain in sfdc_domains:
                ratio = fuzz.ratio(domain, sfdc_domain)
                if ratio > 85 and sfdc_domain != domain:
                    fuzzy_matches.append((sfdc_domain, ratio))
            
            if fuzzy_matches:
                # Sort by fuzzy score
                fuzzy_matches.sort(key=lambda x: x[1], reverse=True)
                
                # If AI is available, re-rank using semantic similarity
                if model and HAVE_TRANSFORMERS:
                    try:
                        # Get only the domain names
                        candidate_domains = [match[0] for match in fuzzy_matches[:10]]  # Top 10 candidates
                        
                        # Encode input domain
                        input_embedding = model.encode(domain, convert_to_tensor=True, show_progress_bar=False)
                        
                        # Encode candidate domains
                        candidate_embeddings = model.encode(candidate_domains, convert_to_tensor=True, show_progress_bar=False)
                        
                        # Calculate cosine similarities
                        cos_scores = util.cos_sim(input_embedding, candidate_embeddings)[0]
                        
                        # Get best match
                        best_idx = torch.argmax(cos_scores).item()
                        best_score = cos_scores[best_idx].item()
                        
                        if best_score >= 0.85:
                            matched_domain = candidate_domains[best_idx]
                            result_row['Match Type'] = 'AI Match'
                            result_row['Matched Domain'] = matched_domain
                            result_row['Match Score'] = f'{round(best_score * 100, 1)}%'
                            for col in return_columns:
                                result_row[col] = sfdc_dict[matched_domain].get(col, '')
                    except Exception as e:
                        # Fallback to fuzzy match if AI fails
                        matched_domain = fuzzy_matches[0][0]
                        result_row['Match Type'] = 'Fuzzy Match'
                        result_row['Matched Domain'] = matched_domain
                        result_row['Match Score'] = f'{fuzzy_matches[0][1]}%'
                        for col in return_columns:
                            result_row[col] = sfdc_dict[matched_domain].get(col, '')
                else:
                    # Use fuzzy match only
                    matched_domain = fuzzy_matches[0][0]
                    result_row['Match Type'] = 'Fuzzy Match'
                    result_row['Matched Domain'] = matched_domain
                    result_row['Match Score'] = f'{fuzzy_matches[0][1]}%'
                    for col in return_columns:
                        result_row[col] = sfdc_dict[matched_domain].get(col, '')
        
        # Add empty values for return columns if no match
        if result_row['Match Type'] == 'No Match':
            for col in return_columns:
                result_row[col] = ''
        
        results.append(result_row)
    
    return pd.DataFrame(results)

def main():
    # Header
    st.markdown('<p class="main-header">🔍 Advanced Domain Matcher Tool</p>', unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    <div class="info-box">
    <h3>Welcome to the Domain Matcher!</h3>
    <p>This tool helps you match domains between two Excel files using multiple matching strategies:</p>
    <ul>
        <li><strong>Exact Match:</strong> Finds identical domains</li>
        <li><strong>TLD Match:</strong> Matches domains with the same base but different extensions (e.g., example.com vs example.net)</li>
        <li><strong>Fuzzy Match:</strong> Finds similar domains using string similarity</li>
        <li><strong>AI Match:</strong> Uses semantic similarity to find related domains</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for file uploads
    with st.sidebar:
        st.header("📁 Upload Files")
        
        sfdc_file = st.file_uploader(
            "Upload SFDC File (Excel)",
            type=['xlsx', 'xls'],
            help="Upload your SFDC database file"
        )
        
        input_file = st.file_uploader(
            "Upload Input File (Excel)",
            type=['xlsx', 'xls'],
            help="Upload the file with domains to match"
        )
        
        st.markdown("---")
        st.markdown("### 📊 Statistics")
        if sfdc_file:
            st.success("✅ SFDC file uploaded")
        if input_file:
            st.success("✅ Input file uploaded")
    
    # Main content
    if sfdc_file and input_file:
        try:
            # Load data
            with st.spinner("Loading files..."):
                sfdc_df = pd.read_excel(sfdc_file)
                input_df = pd.read_excel(input_file)
            
            st.success("✅ Files loaded successfully!")
            
            # Show file previews in expanders
            col1, col2 = st.columns(2)
            
            with col1:
                with st.expander("📄 SFDC File Preview"):
                    st.dataframe(sfdc_df.head(), use_container_width=True)
                    st.info(f"Rows: {len(sfdc_df)} | Columns: {len(sfdc_df.columns)}")
            
            with col2:
                with st.expander("📄 Input File Preview"):
                    st.dataframe(input_df.head(), use_container_width=True)
                    st.info(f"Rows: {len(input_df)} | Columns: {len(input_df.columns)}")
            
            st.markdown("---")
            
            # Column selection
            st.header("🎯 Configure Matching")
            
            col1, col2 = st.columns(2)
            
            with col1:
                match_col_sfdc = st.selectbox(
                    "Select matching column from SFDC file",
                    options=sfdc_df.columns.tolist(),
                    help="Choose the column containing domains in the SFDC file"
                )
            
            with col2:
                match_col_input = st.selectbox(
                    "Select matching column from Input file",
                    options=input_df.columns.tolist(),
                    help="Choose the column containing domains in the Input file"
                )
            
            # Return columns selection
            return_columns = st.multiselect(
                "Select columns to return from SFDC file",
                options=sfdc_df.columns.tolist(),
                default=[col for col in sfdc_df.columns.tolist()[:3]],  # Default to first 3 columns
                help="Choose which columns from the SFDC file to include in the results"
            )
            
            if not return_columns:
                st.warning("⚠️ Please select at least one column to return")
                return
            
            st.markdown("---")
            
            # Matching button
            if st.button("🚀 Start Matching", type="primary", use_container_width=True):
                with st.spinner("Processing matches..."):
                    # Create progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    status_text.text("Matching domains...")
                    
                    try:
                        # Perform matching
                        results_df = process_matching(
                            sfdc_df.copy(),
                            input_df.copy(),
                            match_col_sfdc,
                            match_col_input,
                            return_columns,
                            progress_bar
                        )
                        
                        progress_bar.empty()
                        status_text.empty()
                        
                        # Display results
                        st.markdown('<div class="success-box">', unsafe_allow_html=True)
                        st.success("✅ Matching completed successfully!")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Statistics
                        st.header("📈 Results Summary")
                        
                        col1, col2, col3, col4, col5 = st.columns(5)
                        
                        with col1:
                            st.metric("Total Domains", len(results_df))
                        with col2:
                            exact_matches = len(results_df[results_df['Match Type'] == 'Exact Match'])
                            st.metric("Exact Matches", exact_matches)
                        with col3:
                            tld_matches = len(results_df[results_df['Match Type'] == 'TLD Match'])
                            st.metric("TLD Matches", tld_matches)
                        with col4:
                            fuzzy_ai_matches = len(results_df[results_df['Match Type'].isin(['Fuzzy Match', 'AI Match'])])
                            st.metric("Fuzzy/AI Matches", fuzzy_ai_matches)
                        with col5:
                            no_matches = len(results_df[results_df['Match Type'] == 'No Match'])
                            st.metric("No Matches", no_matches)
                        
                        # Results preview
                        st.header("📋 Results Preview")
                        st.dataframe(results_df, use_container_width=True, height=400)
                        
                        # Download section
                        st.header("💾 Download Results")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Excel download
                            output = BytesIO()
                            results_df.to_excel(output, index=False, engine='openpyxl')
                            output.seek(0)
                            
                            st.download_button(
                                label="📥 Download as Excel",
                                data=output,
                                file_name="domain_matching_results.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )
                        
                        with col2:
                            # CSV download
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download as CSV",
                                data=csv,
                                file_name="domain_matching_results.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        
                    except Exception as e:
                        st.error(f"❌ Error during matching: {str(e)}")
                        st.exception(e)
        
        except Exception as e:
            st.error(f"❌ Error loading files: {str(e)}")
            st.exception(e)
    
    else:
        # Instructions when no files are uploaded
        st.info("👆 Please upload both SFDC and Input files to get started")
        
        st.markdown("""
        ### 📝 Instructions:
        
        1. **Upload Files**: Use the sidebar to upload your SFDC database file and input file
        2. **Select Columns**: Choose which columns to use for matching
        3. **Choose Return Columns**: Select which data you want to retrieve from the SFDC file
        4. **Start Matching**: Click the button to begin the matching process
        5. **Download Results**: Export your results in Excel or CSV format
        
        ### 💡 Tips:
        
        - Ensure your Excel files have headers in the first row
        - Domain columns should contain clean domain names (e.g., example.com)
        - Larger files may take longer to process
        - AI matching provides the best results but requires more processing time
        """)

if __name__ == "__main__":
    main()
