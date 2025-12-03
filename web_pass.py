import streamlit as st
import hashlib

# Configure the page
st.set_page_config(page_title="Secure App", page_icon="🔒")

# Password configuration (in production, store this securely)
CORRECT_PASSWORD_HASH = hashlib.sha256("mypassword123".encode()).hexdigest()

def check_password(password):
    """Check if the provided password is correct"""
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    return password_hash == CORRECT_PASSWORD_HASH

def login_page():
    """Display login page"""
    st.title("🔒 Login Required")
    st.write("Please enter your password to access the application.")
    
    password = st.text_input("Password", type="password", key="password_input")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        login_button = st.button("Login", use_container_width=True)
    
    if login_button:
        if check_password(password):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("❌ Incorrect password. Please try again.")
            st.info("💡 Hint: Default password is 'mypassword123'")

def main_app():
    """Main application content (shown after successful login)"""
    st.title("🎉 Welcome to the Secure App!")
    
    # Logout button in sidebar
    with st.sidebar:
        st.write(f"👤 Logged in")
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.rerun()
    
    # Main content
    st.write("## You have successfully logged in!")
    
    st.write("### This is your protected content")
    
    # Example features
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📝 Data Entry", "⚙️ Settings"])
    
    with tab1:
        st.write("### Dashboard")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Users", "1,234", "+12%")
        with col2:
            st.metric("Revenue", "$45,678", "+8%")
        with col3:
            st.metric("Active Sessions", "89", "-3%")
        
        st.line_chart({"data": [1, 2, 3, 4, 5, 6, 7]})
    
    with tab2:
        st.write("### Data Entry Form")
        name = st.text_input("Name")
        email = st.text_input("Email")
        message = st.text_area("Message")
        if st.button("Submit"):
            st.success("✅ Data submitted successfully!")
    
    with tab3:
        st.write("### Settings")
        st.checkbox("Enable notifications")
        st.checkbox("Dark mode")
        st.selectbox("Language", ["English", "Spanish", "French"])

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# Show appropriate page based on authentication status
if st.session_state.authenticated:
    main_app()
else:
    login_page()