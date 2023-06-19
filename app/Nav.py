import streamlit as st
from streamlit_option_menu import option_menu


Navv = True
f = open("vars.txt", "r")
Verr = str(f.read())
f.close()
Logged = len(Verr)
with st.sidebar:
    selected = option_menu(
        menu_title = "Main Menu",
        options =["Sign up","Log in","ASL Translator","About Us"],
    )

if selected =="Sign up":
    if (int(Verr[Logged - 1])) == 0:
        exec(open('signup.py').read())
    else:
        st.write("You are Logged In!")
if selected == "Log in":
    if (int(Verr[Logged-1])) == 0:
        exec(open('login.py').read())
    else:
        st.write("You are Logged In!")

if selected == "ASL Translator":
    if (int(Verr[Logged - 1])) == 1:
        exec(open('translator.py').read())
        st.write(" Log ")
    else:
        st.write("Please Login First")
if selected == "About Us":
    if (int(Verr[Logged-1])) == 1:
        exec(open('Aboutus.py').read())
    else:
        st.write("Please Login First")



