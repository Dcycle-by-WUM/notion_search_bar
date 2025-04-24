import streamlit as st
from query import query_notion_data, format_results_for_csv
import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get Amplitude API key from environment
AMPLITUDE_API_KEY = os.getenv('AMPLITUDE_API_KEY')
st.write(f"Amplitude API Key loaded: {'Yes' if AMPLITUDE_API_KEY else 'No'}")

# Add Amplitude script and initialization
st.markdown(f"""
<script src="https://cdn.amplitude.com/libs/analytics-browser-2.6.2-min.js.gz"></script>
<script>
    amplitude.init('{AMPLITUDE_API_KEY}');
    console.log('Amplitude initialized');
</script>
""", unsafe_allow_html=True)

st.title("Search whatever tf you want in Signals")

# Initialize session state for results if not already present
if 'query_results' not in st.session_state:
    st.session_state.query_results = None

# Query input
query = st.text_input("Enter your query:", placeholder="Find the most relevant opportunities related to...")

# Number of results slider
top_k = st.slider("Number of results to return:", min_value=1, max_value=20, value=5)

# Query button
if st.button("Search"):
    if query:
        # Track the search event
        st.markdown(f"""
        <script>
            console.log('Tracking search with query:', '{query}', 'and top_k:', {top_k});
            amplitude.track('Search Button Clicked', {{
                query: '{query}',
                top_k: {top_k}
            }});
            console.log('Event tracked');
        </script>
        """, unsafe_allow_html=True)
        
        with st.spinner("Searching..."):
            results = query_notion_data(query, top_k)
            st.session_state.query_results = results
    else:
        st.warning("Please enter a query.")

# Display results
if st.session_state.query_results and st.session_state.query_results['matches']:
    st.subheader("Search Results")
    
    for i, match in enumerate(st.session_state.query_results['matches'], 1):
        st.markdown(f"### Result {i} (Score: {match['score']:.2f})")
        if match['metadata'].get('submission_time'):
            st.write(f"**Submission Time:** {match['metadata']['submission_time']}")
        if match['metadata'].get('how_found'):
            st.write(f"**How Found:** {match['metadata']['how_found']}")
        if match['metadata'].get('opportunity'):
            st.write(f"**Opportunity:** {match['metadata']['opportunity']}")
        st.markdown("---")  # Add a separator between results
    
    # Add download button for CSV
    csv_data = format_results_for_csv(st.session_state.query_results)
    if csv_data:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"query_results_{timestamp}.csv"
        st.download_button(
            label="Download Results as CSV",
            data=csv_data,
            file_name=filename,
            mime="text/csv"
        )
else:
    st.info("No results to display. Enter a query and click Search.")

# Pass Amplitude API key to the template
st.markdown(f"""
<script>
    window.AMPLITUDE_API_KEY = '{AMPLITUDE_API_KEY}';
</script>
""", unsafe_allow_html=True) 