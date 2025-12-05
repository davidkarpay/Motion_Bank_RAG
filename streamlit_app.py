"""
Motion RAG Assistant - Streamlit Frontend
Attorney interface for motion strategy and drafting
"""
import streamlit as st
import streamlit_authenticator as stauth
import requests
from datetime import datetime
from pathlib import Path
import json
import yaml

# Configuration
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Motion RAG Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1a365d;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #4a5568;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f7fafc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3182ce;
    }
    .source-card {
        background: #edf2f7;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .chat-user {
        background: #e2e8f0;
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .chat-assistant {
        background: #ebf8ff;
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 3px solid #3182ce;
    }
    .warning-box {
        background: #fffaf0;
        border-left: 4px solid #ed8936;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ============== Helper Functions ==============

def api_request(endpoint: str, method: str = "GET", data: dict = None):
    """Make API request with error handling"""
    try:
        url = f"{API_URL}{endpoint}"
        if method == "GET":
            response = requests.get(url, params=data)
        elif method == "POST":
            response = requests.post(url, json=data)
        elif method == "PATCH":
            response = requests.patch(url, json=data)
        else:
            response = requests.request(method, url, json=data)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Make sure the server is running.")
        return None
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None


def init_session_state():
    """Initialize session state variables"""
    if "chat_session_id" not in st.session_state:
        st.session_state.chat_session_id = None
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "current_draft" not in st.session_state:
        st.session_state.current_draft = None


# ============== Sidebar ==============

def render_sidebar():
    """Render the sidebar navigation"""
    st.sidebar.markdown("## ⚖️ Motion RAG")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation",
        ["🔍 Search Motions", "💬 Strategy Chat", "📝 Draft Motion", 
         "📊 Analytics", "📁 Manage Database"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    
    # System status
    with st.sidebar.expander("System Status", expanded=False):
        health = api_request("/health")
        if health:
            st.write(f"**Status:** {health['status'].upper()}")
            st.write(f"Vector DB: {health['vector_db']}")
            st.write(f"LLM: {health['llm']}")
        
        stats = api_request("/stats")
        if stats:
            st.write(f"**Motions:** {stats['total_motions']}")
            st.write(f"**Vectors:** {stats['total_vectors']}")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div class="warning-box">
    ⚠️ <strong>Review Required</strong><br>
    All AI-generated content requires attorney verification before filing.
    </div>
    """, unsafe_allow_html=True)
    
    return page


# ============== Search Page ==============

def render_search_page():
    """Render the motion search page"""
    st.markdown('<p class="main-header">🔍 Search Motions</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Find relevant precedent motions from your database</p>', unsafe_allow_html=True)
    
    # Search form
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "Search Query",
            placeholder="e.g., Fourth Amendment traffic stop warrantless search",
            help="Describe the legal issue or facts you're researching"
        )
    
    with col2:
        top_k = st.number_input("Results", min_value=1, max_value=20, value=5)
    
    # Filters
    with st.expander("Advanced Filters", expanded=False):
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        motion_types = api_request("/motion-types")
        motion_options = ["Any"] + [m["name"] for m in motion_types.get("motion_types", [])] if motion_types else ["Any"]
        motion_ids = [""] + [m["id"] for m in motion_types.get("motion_types", [])] if motion_types else [""]
        
        with filter_col1:
            motion_type_name = st.selectbox("Motion Type", motion_options)
            motion_type = motion_ids[motion_options.index(motion_type_name)] if motion_type_name != "Any" else None
        
        with filter_col2:
            outcome = st.selectbox(
                "Outcome",
                ["Any", "Granted", "Denied", "Granted in Part"]
            )
            outcome = outcome.lower().replace(" ", "_") if outcome != "Any" else None
        
        with filter_col3:
            judge = st.text_input("Judge (optional)")
            judge = judge if judge else None
    
    # Search button
    if st.button("Search", type="primary") and query:
        with st.spinner("Searching..."):
            results = api_request("/search", method="POST", data={
                "query": query,
                "motion_type": motion_type,
                "outcome": outcome,
                "judge": judge,
                "top_k": top_k
            })
        
        if results and results.get("results"):
            st.markdown(f"### Found {len(results['results'])} relevant motions")
            
            for i, result in enumerate(results["results"], 1):
                with st.expander(
                    f"**{i}. {result['title']}** | "
                    f"Type: {result['motion_type'].replace('_', ' ').title()} | "
                    f"Outcome: {result['outcome'] or 'N/A'} | "
                    f"Score: {result['relevance_score']:.2f}"
                ):
                    st.markdown(f"**Section:** {result['section'] or 'General'}")
                    st.markdown("**Preview:**")
                    st.markdown(f">{result['text_preview']}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"View Full Motion", key=f"view_{result['motion_id']}"):
                            motion = api_request(f"/motions/{result['motion_id']}")
                            if motion:
                                st.json(motion)
                    with col2:
                        if st.button(f"Find Similar", key=f"similar_{result['motion_id']}"):
                            similar = api_request(f"/search/similar/{result['motion_id']}")
                            if similar:
                                st.write("Similar motions:")
                                for s in similar.get("similar_motions", []):
                                    st.write(f"- {s['title']} ({s['outcome']})")
        else:
            st.info("No matching motions found. Try broadening your search.")


# ============== Chat Page ==============

def render_chat_page():
    """Render the strategy chat page"""
    st.markdown('<p class="main-header">💬 Strategy Session</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Discuss motion strategy with AI assistance grounded in your successful motions</p>', unsafe_allow_html=True)
    
    # Session setup
    if not st.session_state.chat_session_id:
        st.markdown("### Start a New Strategy Session")
        
        with st.form("new_session"):
            motion_types = api_request("/motion-types")
            motion_options = ["Not specified"] + [m["name"] for m in motion_types.get("motion_types", [])] if motion_types else ["Not specified"]
            motion_ids = [""] + [m["id"] for m in motion_types.get("motion_types", [])] if motion_types else [""]
            
            motion_type_name = st.selectbox("Motion Type (optional)", motion_options)
            motion_type = motion_ids[motion_options.index(motion_type_name)] if motion_type_name != "Not specified" else None
            
            charge_type = st.text_input("Charge Type (optional)", placeholder="e.g., burglary, DUI, possession")
            
            key_facts = st.text_area(
                "Key Case Facts (optional)",
                placeholder="Brief summary of relevant facts for this motion...",
                height=100
            )
            
            submitted = st.form_submit_button("Start Session", type="primary")
            
            if submitted:
                session = api_request("/sessions", method="POST", data={
                    "motion_type": motion_type,
                    "charge_type": charge_type if charge_type else None,
                    "key_facts": key_facts if key_facts else None
                })
                
                if session:
                    st.session_state.chat_session_id = session["session_id"]
                    st.session_state.chat_messages = []
                    st.rerun()
    
    else:
        # Active session
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"**Session:** {st.session_state.chat_session_id[:8]}...")
        with col2:
            if st.button("End Session"):
                st.session_state.chat_session_id = None
                st.session_state.chat_messages = []
                st.rerun()
        
        st.markdown("---")
        
        # Chat history
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.chat_messages:
                if msg["role"] == "user":
                    st.markdown(f'<div class="chat-user">🧑‍💼 <strong>You:</strong><br>{msg["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-assistant">🤖 <strong>Assistant:</strong><br>{msg["content"]}</div>', unsafe_allow_html=True)
                    if msg.get("sources_used", 0) > 0:
                        st.caption(f"📚 Based on {msg['sources_used']} source(s)")
        
        # Input
        st.markdown("---")
        
        with st.form("chat_input", clear_on_submit=True):
            user_input = st.text_area(
                "Your message",
                placeholder="Ask about motion strategy, request analysis, or ask for a draft...",
                height=100,
                label_visibility="collapsed"
            )
            
            col1, col2, col3 = st.columns([1, 1, 4])
            with col1:
                use_retrieval = st.checkbox("Use motion database", value=True)
            with col2:
                send = st.form_submit_button("Send", type="primary")
        
        if send and user_input:
            # Add user message
            st.session_state.chat_messages.append({
                "role": "user",
                "content": user_input
            })
            
            # Get response
            with st.spinner("Thinking..."):
                response = api_request(
                    f"/sessions/{st.session_state.chat_session_id}/chat",
                    method="POST",
                    data={
                        "message": user_input,
                        "use_retrieval": use_retrieval
                    }
                )
            
            if response:
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": response["content"],
                    "sources_used": response.get("sources_used", 0)
                })
            
            st.rerun()
        
        # Generate draft button
        st.markdown("---")
        if st.button("📝 Generate Draft from Discussion", type="secondary"):
            with st.spinner("Generating draft..."):
                draft = api_request(
                    f"/sessions/{st.session_state.chat_session_id}/generate-draft",
                    method="POST"
                )
            
            if draft:
                st.session_state.current_draft = draft
                st.success("Draft generated! View in the Draft Motion tab.")


# ============== Draft Page ==============

def render_draft_page():
    """Render the motion drafting page"""
    st.markdown('<p class="main-header">📝 Draft Motion</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Generate motion drafts based on successful precedents</p>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["New Draft", "Current Draft"])
    
    with tab1:
        st.markdown("### Generate New Motion Draft")
        
        motion_types = api_request("/motion-types")
        
        if motion_types:
            motion_type = st.selectbox(
                "Motion Type",
                options=[m["id"] for m in motion_types["motion_types"]],
                format_func=lambda x: next(
                    (m["name"] for m in motion_types["motion_types"] if m["id"] == x), x
                )
            )
        else:
            motion_type = st.text_input("Motion Type", placeholder="e.g., motion_to_suppress")
        
        facts = st.text_area(
            "Statement of Facts",
            placeholder="Describe the relevant facts of your case...",
            height=200
        )
        
        legal_issues = st.text_input(
            "Legal Issues (comma-separated)",
            placeholder="e.g., warrantless search, lack of consent, Fourth Amendment"
        )
        legal_issues_list = [i.strip() for i in legal_issues.split(",")] if legal_issues else []
        
        if st.button("Generate Draft", type="primary"):
            if not facts:
                st.warning("Please provide case facts.")
            else:
                with st.spinner("Generating draft... This may take a minute."):
                    draft = api_request("/draft", method="POST", data={
                        "motion_type": motion_type,
                        "facts": facts,
                        "legal_issues": legal_issues_list
                    })
                
                if draft:
                    st.session_state.current_draft = draft
                    st.success("Draft generated!")
                    st.rerun()
    
    with tab2:
        if st.session_state.current_draft:
            draft = st.session_state.current_draft
            
            st.markdown(f"### {draft['title']}")
            st.markdown(f"**Type:** {draft['motion_type'].replace('_', ' ').title()}")
            
            # Metadata
            with st.expander("📚 Sources & Citations"):
                st.markdown("**Cited Cases:**")
                for case in draft.get("cited_cases", []):
                    st.write(f"- {case}")
                
                st.markdown("**Cited Statutes:**")
                for statute in draft.get("cited_statutes", []):
                    st.write(f"- {statute}")
                
                st.markdown("**Based on Motions:**")
                for motion_id in draft.get("source_motion_ids", [])[:5]:
                    st.write(f"- {motion_id[:8]}...")
            
            # Content
            st.markdown("---")
            st.markdown("### Draft Content")
            
            # Editable content
            edited_content = st.text_area(
                "Edit draft",
                value=draft["content"],
                height=600,
                label_visibility="collapsed"
            )
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📋 Copy to Clipboard"):
                    st.code(edited_content)
                    st.info("Select and copy the text above")
            
            with col2:
                if st.button("💾 Download as Text"):
                    st.download_button(
                        "Download",
                        edited_content,
                        file_name=f"{draft['motion_type']}_draft.txt",
                        mime="text/plain"
                    )
            
            with col3:
                if st.button("🗑️ Clear Draft"):
                    st.session_state.current_draft = None
                    st.rerun()
            
            # Warning
            st.markdown("""
            <div class="warning-box">
            ⚠️ <strong>ATTORNEY REVIEW REQUIRED</strong><br>
            This draft is AI-generated and requires thorough review. Verify all citations, 
            check factual accuracy, and ensure compliance with local rules before filing.
            </div>
            """, unsafe_allow_html=True)
        
        else:
            st.info("No draft available. Generate one from the 'New Draft' tab or from a Strategy Session.")


# ============== Analytics Page ==============

def render_analytics_page():
    """Render the analytics page"""
    st.markdown('<p class="main-header">📊 Motion Analytics</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Analyze patterns and success rates across your motion database</p>', unsafe_allow_html=True)
    
    # Select motion type
    motion_types = api_request("/motion-types")
    
    if motion_types:
        motion_type = st.selectbox(
            "Select Motion Type",
            options=[m["id"] for m in motion_types["motion_types"]],
            format_func=lambda x: next(
                (m["name"] for m in motion_types["motion_types"] if m["id"] == x), x
            )
        )
        
        if st.button("Load Analytics", type="primary"):
            with st.spinner("Loading analytics..."):
                analytics = api_request(f"/analytics/{motion_type}")
            
            if analytics:
                # Overview metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Motions", analytics["total_motions"])
                
                with col2:
                    st.metric(
                        "Success Rate",
                        f"{analytics['success_rate']:.1%}"
                    )
                
                with col3:
                    st.metric("Judges Analyzed", len(analytics.get("by_judge", {})))
                
                st.markdown("---")
                
                # Judge analysis
                st.markdown("### Success Rate by Judge")
                
                if analytics.get("by_judge"):
                    judge_data = []
                    for judge, stats in analytics["by_judge"].items():
                        judge_data.append({
                            "Judge": judge,
                            "Success Rate": f"{stats['success_rate']:.1%}",
                            "Granted": stats["granted"],
                            "Denied": stats["denied"],
                            "Total": stats["total"]
                        })
                    
                    st.table(judge_data)
                else:
                    st.info("No judge-specific data available.")
                
                # Common arguments
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Common Legal Issues")
                    if analytics.get("common_issues"):
                        for issue, count in list(analytics["common_issues"].items())[:10]:
                            st.write(f"- **{issue}** ({count})")
                    else:
                        st.info("No issue data available.")
                
                with col2:
                    st.markdown("### Frequently Cited")
                    
                    st.markdown("**Cases:**")
                    if analytics.get("top_cases"):
                        for case, count in list(analytics["top_cases"].items())[:5]:
                            st.write(f"- {case} ({count})")
                    
                    st.markdown("**Statutes:**")
                    if analytics.get("top_statutes"):
                        for statute, count in list(analytics["top_statutes"].items())[:5]:
                            st.write(f"- {statute} ({count})")
            else:
                st.warning("No analytics available for this motion type.")


# ============== Database Management Page ==============

def render_manage_page():
    """Render the database management page"""
    st.markdown('<p class="main-header">📁 Manage Database</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload and manage motion documents</p>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Upload Motion", "Batch Import"])
    
    with tab1:
        st.markdown("### Upload Single Motion")
        
        uploaded_file = st.file_uploader(
            "Choose a motion file",
            type=["pdf", "docx", "txt"],
            help="Upload a PDF, Word document, or text file"
        )
        
        if uploaded_file:
            st.markdown("#### Motion Metadata")
            
            col1, col2 = st.columns(2)
            
            with col1:
                outcome = st.selectbox(
                    "Outcome",
                    ["unknown", "granted", "denied", "granted_in_part", "pending", "withdrawn", "moot"]
                )
                judge = st.text_input("Judge", placeholder="e.g., Hon. Smith")
            
            with col2:
                charge_type = st.text_input("Charge Type", placeholder="e.g., burglary")
                court = st.text_input("Court", placeholder="e.g., 15th Judicial Circuit")
            
            if st.button("Upload & Process", type="primary"):
                with st.spinner("Processing motion..."):
                    # Upload file
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    params = {
                        "outcome": outcome if outcome != "unknown" else None,
                        "judge": judge if judge else None,
                        "charge_type": charge_type if charge_type else None
                    }
                    
                    try:
                        response = requests.post(
                            f"{API_URL}/ingest/upload",
                            files=files,
                            params={k: v for k, v in params.items() if v}
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.success(f"✅ Uploaded: {result['title']}")
                            st.write(f"- Motion ID: {result['motion_id']}")
                            st.write(f"- Type: {result['motion_type']}")
                            st.write(f"- Chunks: {result['chunk_count']}")
                        else:
                            st.error(f"Upload failed: {response.text}")
                    except Exception as e:
                        st.error(f"Error: {e}")
    
    with tab2:
        st.markdown("### Batch Import")
        st.markdown("Import all motions from a directory on the server.")
        
        directory = st.text_input(
            "Directory Path",
            placeholder="/path/to/motions/folder"
        )
        
        if st.button("Start Batch Import") and directory:
            with st.spinner("Processing directory..."):
                result = api_request("/ingest/batch", method="POST", data=directory)
            
            if result:
                st.success(f"✅ Batch import complete!")
                st.write(f"- Processed: {result['processed']}")
                st.write(f"- Ingested: {result['ingested']}")
                st.write(f"- Failed: {result['failed']}")
    
    st.markdown("---")
    
    # Database stats
    st.markdown("### Database Statistics")
    
    stats = api_request("/stats")
    if stats:
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Motions", stats["total_motions"])
            st.metric("Total Vectors", stats["total_vectors"])
        
        with col2:
            st.markdown("**By Motion Type:**")
            for mtype, count in stats.get("motion_types", {}).items():
                st.write(f"- {mtype.replace('_', ' ').title()}: {count}")


# ============== Authentication ==============

def load_auth_config():
    """Load authentication configuration from YAML file"""
    config_path = Path(__file__).parent / "auth_config.yaml"
    if not config_path.exists():
        st.error("Authentication configuration not found. Please create auth_config.yaml from auth_config.yaml.example")
        st.stop()

    with open(config_path) as file:
        return yaml.safe_load(file)


# ============== Main ==============

def main():
    # Load authentication config
    config = load_auth_config()

    # Initialize authenticator
    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config.get('preauthorized', {})
    )

    # Render login form
    name, authentication_status, username = authenticator.login('Login', 'main')

    if authentication_status == False:
        st.error('Username/password is incorrect')
        return

    if authentication_status == None:
        st.warning('Please enter your username and password')
        return

    # User is authenticated - proceed with app
    init_session_state()

    # Store user info in session
    st.session_state['username'] = username
    st.session_state['name'] = name

    page = render_sidebar()

    # Add logout button to sidebar
    with st.sidebar:
        st.markdown("---")
        st.markdown(f"**Logged in as:** {name}")
        authenticator.logout('Logout', 'sidebar')

    if page == "🔍 Search Motions":
        render_search_page()
    elif page == "💬 Strategy Chat":
        render_chat_page()
    elif page == "📝 Draft Motion":
        render_draft_page()
    elif page == "📊 Analytics":
        render_analytics_page()
    elif page == "📁 Manage Database":
        render_manage_page()


if __name__ == "__main__":
    main()
