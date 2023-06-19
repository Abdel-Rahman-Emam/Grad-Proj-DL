import streamlit as st
import firebase_admin
from firebase_admin import db , credentials
st.empty
if not firebase_admin._apps:
        cred = credentials.Certificate('gradproj-976fe-firebase-adminsdk-uf9pv-0ef165c526.json')
        default_app = firebase_admin.initialize_app(cred,{"databaseURL": "https://gradproj-976fe-default-rtdb.europe-west1.firebasedatabase.app/"})
def signup():
    st.title("Sign Up")
    st.write("Create your account")

    # Input fields for user details
    username = st.text_input("Username")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")
    data1 = {
    'Email': email,
    'Password': password
    }


    # Signup button
    if st.button("Sign Up"):
        x = db.reference("Number of Users").child("Number").get()
        x = x + 1
        db.reference("Number of Users").child("Number").set(x)
        if password == confirm_password:
            # Perform signup logic here
            st.success("Signup successful!")
            st.write(f"Username: {username}")
            st.write(f"Email: {email}")
            db.reference("Users/").child(str(x)).set(data1)
        else:
            st.error("Passwords do not match.")

# Run the signup function
signup()
