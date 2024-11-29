import os
import streamlit as st
import pandas as pd
import requests as rs

#from st_pages import hide_pages
from time import sleep

from streamlit import source_util
from streamlit.runtime.scriptrunner import get_script_run_ctx

from dotenv import load_dotenv

load_dotenv()

#The tutorial page
#https://oguzhari.medium.com/making-user-login-application-with-streamlit-91ce5e598f23

st.title('Amazing User Login App')

#********************************

page = "pages/bugatti.py" # MODIFY TO YOUR PAGE

ctx = get_script_run_ctx()
ctx_main_script = ""
if ctx:
  ctx_main_script = ctx.main_script_path

st.write("**Main Script File**")
st.text(f"\"{ctx_main_script}\"")

st.write("**Current Working Directory**")
st.text(f"\"{os.getcwd()}\"")

st.write("**Normalized Current Working Directory**")
st.text(f"\"{os.path.normpath(os.getcwd())}\"")

st.write("**Main Script Path**")
main_script_path = os.path.join(os.getcwd(), ctx_main_script)
st.text(f"\"{main_script_path}\"")

st.write("**Main Script Directory**")
main_script_directory = os.path.dirname(main_script_path)
st.text(f"\"{main_script_directory}\"")

st.write("**Normalized Path**")
page = os.path.normpath(page)
st.text(f"\"{page}\"")

st.write("**Requested Page**")
requested_page = os.path.join(main_script_directory, page)
st.text(f"\"{requested_page}\"")

st.write("**All Pages Page**")
all_app_pages = list(source_util.get_pages(ctx_main_script).values())
st.json(all_app_pages, expanded=True)

st.page_link(page, label="Page Two")




#********************************


#sheet_csv = st.secrets["database_url"]
#sheet_csv = os.environ.get["database_url"]
sheet_csv = 'https://docs.google.com/spreadsheets/d/1QAAqdQaie3Alt4l1QucyC9SS_XvpeG9vgaOH8GXJCSs/export?format=csv'

res = rs.get(url=sheet_csv)
open('database.csv', 'wb').write(res.content)
database = pd.read_csv('database.csv', header=0)


# Create user_state
if 'user_state' not in st.session_state:
    st.session_state.user_state = {
        'name_surname': '',
        'password': '',
        'logged_in': False,
        'user_type': '',
        'mail_adress': '',
        'fixed_user_message': ''
    }

if not st.session_state.user_state['logged_in']:
    # Create login form
    st.write('Please login')
    mail_adress = st.text_input('E-Mail')
    password = st.text_input('Password', type='password')
    submit = st.button('Login')

    # Check if user is logged in
    if submit:
        user_ = database[database['mail_adress'] == mail_adress].copy()
        if len(user_) == 0:
            st.error('User not found')
        else:
            if user_['mail_adress'].values[0] == mail_adress and user_['password'].values[0] == password:
                st.session_state.user_state['mail_adress'] = mail_adress
                st.session_state.user_state['password'] = password
                st.session_state.user_state['logged_in'] = True
                st.session_state.user_state['user_type'] = user_['user_type'].values[0]
                st.session_state.user_state['mail_adress'] = user_['mail_adress'].values[0]
                st.session_state.user_state['fixed_user_message'] = user_['fixed_user_message'].values[0]
                st.write('You are logged in')
                st.rerun()

                #sleep(0.5)
                #st.switch_page("st-chatApp/bugatti.py")
            else:
                st.write('Invalid username or password')
    

elif st.session_state.user_state['logged_in']:
    st.write('Welcome to the app')
    st.write('You are logged in as:', st.session_state.user_state['mail_adress'])
    st.write('You are a:', st.session_state.user_state['user_type'])
    st.write('Your fixed user message:', st.session_state.user_state['fixed_user_message'])
    
    sleep(0.5)
    #st.switch_page("pages/bugatti.py")
    #st.page_link("pages/bugatti.py")
    
    if st.session_state.user_state['user_type'] == 'admin':
        st.write('You have admin rights. Here is the database')
        st.table(database)
