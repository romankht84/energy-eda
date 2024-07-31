# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 20:39:12 2024

@author: amrit
"""
import streamlit as st

# Custom CSS for the sidebar and navigation buttons
sidebar_style = """
    <style>
        /* Change the background color of the sidebar */
        .css-1d391kg {
            background-color: #333333; /* Dark grey */
        }
        /* Change the color of the text in the sidebar */
        .css-1d391kg .css-10trblm {
            color: white; /* White text */
        }
        /* Custom styles for the navigation buttons */
        .nav-button-home, .nav-button-main, .nav-button-howto, .nav-button-download {
            display: block;
            border: none;
            color: white;
            padding: 15px 32px;
            text-align: center;
            text-decoration: none;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 4px;
            width: 100%;
        }
        .nav-button-home {
            background-color: #4CAF50;
        }
        .nav-button-main {
            background-color: #FF6347;
        }
        .nav-button-howto {
            background-color: #1E90FF;
        }
        .nav-button-download {
            background-color: #4682B4; /* Blue Violet */
        }
        .nav-button-home a, .nav-button-main a, .nav-button-howto a, .nav-button-download a {
            color: white;
            text-decoration: none;
        }
        .nav-button-home:hover {
            background-color: #45a049;
        }
        .nav-button-main:hover {
            background-color: #FF4500;
        }
        .nav-button-howto:hover {
            background-color: #1C86EE;
        }
        .nav-button-download:hover {
            background-color: #DAA520; 
        }
    </style>
"""

st.markdown(sidebar_style, unsafe_allow_html=True)

# Navigation function
def navigate_to(page):
    st.experimental_set_query_params(page=page)

# Get the current page from query params
query_params = st.experimental_get_query_params()
current_page = query_params.get('page', ['home'])[0]

# Custom HTML for buttons with unique classes for each
nav_buttons = """
<div style="text-align: center;">
    <div class="nav-button-home"><a href="?page=home">Home</a></div>
    <div class="nav-button-main"><a href="?page=main">Main App</a></div>
    <div class="nav-button-howto"><a href="?page=howto">How to Use</a></div>
</div>
"""

st.sidebar.markdown(nav_buttons, unsafe_allow_html=True)


# Page navigation
if current_page == 'home':
    exec(open("home.py").read())

elif current_page == 'main':
    exec(open("main.py").read())
    
elif current_page == 'howto':
    exec(open("howto.py").read())
