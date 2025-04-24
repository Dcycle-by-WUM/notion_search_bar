import streamlit.components.v1 as components

def init_amplitude(api_key):
    components.html(f"""
    <script src="https://cdn.amplitude.com/libs/analytics-browser-2.6.2-min.js.gz"></script>
    <script>
        amplitude.init('{api_key}');
    </script>
    """, height=0)

def track_event(event_name, properties=None):
    if properties is None:
        properties = {}
    components.html(f"""
    <script>
        amplitude.track('{event_name}', {properties});
    </script>
    """, height=0) 