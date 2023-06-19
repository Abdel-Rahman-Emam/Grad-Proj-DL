import streamlit as st
import firebase_admin
from firebase_admin import db , credentials
if not firebase_admin._apps:
        cred = credentials.Certificate('gradproj-976fe-firebase-adminsdk-uf9pv-0ef165c526.json')
        default_app = firebase_admin.initialize_app(cred,{"databaseURL": "https://gradproj-976fe-default-rtdb.europe-west1.firebasedatabase.app/"})
def login():
    Verr = 0
    f = open("vars.txt","a")
    f.write(str(Verr))
    f.close()

    st.title("Login")
    st.write("Enter your credentials")

    # Input fields for user login
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    data1 = {
    'Email': username,
    'Password': password
}
    Ver = 0
    # Login button
    if st.button("Login"):
        x=db.reference("Number of Users").child("Number").get()
        for x in range(1,x+1):
             if (db.reference("Users/").child(str(x)).get() == data1):
                Verr=1
                Ver = 1
                f = open("vars.txt", "a")
                f.write(str(Verr))
                f.close()
                break
             else:
                Verr=0

        # Perform login logic here (e.g., check username and password against a database)
        if Ver == 1:
            st.success("Login successful!")
            st.write(f"Welcome, {username}!")

        else:
            st.error("Invalid username or password.")

# Run the login function
login()
