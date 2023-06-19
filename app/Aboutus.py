import os
import streamlit as st
if st.button("LogOut"):
    f = open("vars.txt", "a")
    f.write(str(0))
    f.close()




st.title("About Us")
st.markdown('**Team Members**')
st.write('Abdel Rahman Emam Ali Sadek Kassab 18P3602')
st.write( 'Hossam ElDin Mohamed Mostafa 18P4607 ')
st.write( 'Hazem Ahmed Youssef Ibrahim 18P4060 ')
st.write( 'Amr Tarek Mohamed Reda Eldib Ahmed Mohamed Eldib 18P2992')

st.markdown('**More text**')

st.write('Our project, developed as the final graduation project for Ain Shams University Faculty of Engineering,'
         ' is an innovative and comprehensive solution designed to address a real-world problem. With a team of dedicated students,'
         ' we have embarked on this journey to develop a cutting-edge system that combines computer vision and artificial intelligence technologies. '
         'Our project aims to revolutionize the field of sign language recognition,'
         ' enabling seamless communication between individuals with hearing impairments and the broader community. '
         'By leveraging advanced machine learning algorithms and deep neural networks, '
         'our system can accurately interpret and translate sign language gestures in real-time. We have conducted extensive research, '
         'rigorous testing, and iterative development to ensure the robustness and reliability of our solution. '
         'We are proud to present our project as a testament to our technical expertise, creativity, '
         'and commitment to making a positive impact in the lives of individuals with hearing impairments')

