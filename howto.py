import streamlit as st

def styled_text_block(text, color='#000000', background_color='#FFFFE0', font_size='24px'):
    html_code = f"""
    <style>
    .styled-text {{
        font-size: {font_size};
        font-weight: bold;
        color: {color};
        background-color: {background_color};
        padding: 10px;
        border-radius: 8px;
        text-align: center;
        box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.1);
        margin: 20px 0;
    }}
    </style>
    <div class="styled-text">
        {text}
    </div>
    """
    st.markdown(html_code, unsafe_allow_html=True)

    



# Page style
page_style = """
<style>
    body {
        font-family: 'Arial', sans-serif;
        background: linear-gradient(135deg, #ffdd00, #ff7300);
        color: #333;
    }
    .section-title {
        font-size: 36px;
        font-weight: bold;
        color: #333;
        background-color: rgba(255, 255, 255, 0.7); /* Semi-transparent white background */
        padding: 10px 20px;
        border-radius: 5px;
        text-align: center;
        margin: 20px auto;
        width: 100%; /* Set width to 100% */
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .section-icon {
        font-size: 48px;
        color: #ff6347;
        margin-right: 10px;
    }
    .section-subtitle {
        font-size: 16px;
        color: #555;
        text-align: center;
        margin: 20px auto;
        background-color: rgba(255, 255, 255, 0.7); /* Semi-transparent white background */
        padding: 10px 20px;
        border-radius: 5px;
        width: 80%; /* Set width to 80% */
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .button-container {
        text-align: center;
        margin-top: 30px;
    }
    .button-container .button {
        display: inline-block;
        padding: 10px 20px;
        font-size: 18px;
        color: #fff;
        background-color: #333;
        border-radius: 5px;
        text-decoration: none;
        transition: background-color 0.3s ease;
    }
    .button-container .button:hover {
        background-color: #555;
    }
</style>
"""

# How-to guides section


how_to_guides_section = """

<div class="section-subtitle">Explore the step-by-step guides to understand how to use the app effectively.</div>
"""

# App output section

app_output_section = """

<div class="section-subtitle">Discover the key features and outputs of the app.</div>
"""

# Streamlit app layout
st.markdown(page_style, unsafe_allow_html=True)
styled_text_block("How-to Guides", color='#333333', background_color='#FFFFE0', font_size='30px')
st.markdown(how_to_guides_section, unsafe_allow_html=True)

with st.expander("Step 1: Upload Your Dataset"):
    st.write("""
        Upload a time series dataset where Column A has hourly datetime entries and Column B contains energy consumption data in kWh.
        You can download a sample file [here](https://github.com/amritPVre/energy-eda/raw/main/sample_eda.xlsx).
    """)

with st.expander("Step 2: Upload Your Company Logo"):
    st.write("""
        Upload your company logo in PNG or JPEG format. This logo will be used in the generated report.
    """)

with st.expander("Step 3: Fill in Company Information"):
    st.write("""
        Provide your company name, client name, and contact details. This information will be displayed on the cover page of the report.
    """)

with st.expander("Step 4: Input Solar Panel Details"):
    st.write("""
        Enter the required solar panel details including latitude, longitude, surface tilt, surface azimuth, module efficiency, and performance ratio.
        These inputs are crucial for estimating the solar energy production.
    """)

with st.expander("Step 5: Analyze Energy Consumption"):
    st.write("""
        Once the data is uploaded and all details are provided, the app will analyze the hourly, daily, and monthly energy consumption.
        It will also estimate the optimized PV capacity and generate insights on excess energy production.
    """)

with st.expander("Step 6: Download the Report"):
    st.write("""
        After the analysis is complete, you can download a comprehensive report that includes all the insights, charts, and data visualizations.
    """)

styled_text_block("What You Get from the App", color='#333333', background_color='#FFFFE0', font_size='30px')
st.markdown(app_output_section, unsafe_allow_html=True)

with st.expander("Hourly Energy Consumption Distribution"):
    st.write("""
        Visualize the hourly distribution of your energy consumption with detailed charts.
    """)

with st.expander("Daily and Monthly Consumption Profile"):
    st.write("""
        Get insights into your daily and monthly energy consumption patterns.
    """)

with st.expander("Solar PV Capacity Estimation"):
    st.write("""
        The app estimates the optimal solar PV capacity required to meet your annual energy consumption.
    """)

with st.expander("Excess Energy Production Analysis"):
    st.write("""
        Analyze periods of high solar generation and low consumption, and identify excess energy production.
    """)

with st.expander("Data Cleaning and Preprocessing"):
    st.write("""
        The app automatically identifies and fills in missing hours in the dataset with zeros to maintain data integrity.
    """)

with st.expander("Detailed Project Report"):
    st.write("""
        Generate a comprehensive report summarizing all the findings and analyses. The report includes charts, tables, and insights derived from your data.
    """)

# Button to go to main app
st.markdown("""
<div class="button-container">
    <a href="?page=main" class="button">Go to Main App</a>
</div>
""", unsafe_allow_html=True)
