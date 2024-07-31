# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 13:24:37 2024

@author: amrit
"""

import streamlit as st
import pandas as pd
from io import BytesIO
import matplotlib.pyplot as plt
import base64
import tempfile
from fpdf import FPDF
from PIL import Image
import numpy as np
import pvlib
from pvlib import location, irradiance, pvsystem, modelchain
import pytz
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict
import tempfile
import base64
from firebase_config import initialize_firebase
from firebase_admin import db
import xlsxwriter


# Streamlit app layout
#st.set_page_config(page_title="Solar Analytics", layout="centered", initial_sidebar_state="auto")

# Initialize Firebase
initialize_firebase()

def save_to_firebase(data):
    ref = db.reference('your-database-path')
    ref.push(data)



# CSS for custom styling
st.markdown("""
    <style>
        .main {
            background-color: #f5f5f5;
            padding: 20px;
        }
        .stApp {
            background-color: #f5f5f5;
        }
        .metric {
            font-size: 14px;
            color: #333;
            font-weight: bold;
        }
        .header {
            font-size: 24px;
            color: #2e7d32;
            font-weight: bold;
            margin-top: 20px;
        }
        .subheader {
            font-size: 18px;
            color: #2e7d32;
            font-weight: bold;
            margin-top: 10px;
        }
    </style>
""", unsafe_allow_html=True)


# Add custom CSS to set fixed column widths
table_css = """
<style>
    table {
        width: 100%;
        border-collapse: collapse;
    }
    th, td {
        border: 1px solid black;
        padding: 8px;
        text-align: left;
        width: 150px; /* Fixed width for all columns */
    }
</style>
"""
#
#-------Section Header CSS-------#

def styled_text_block(text, color='#000000', background_color='#DBEAFE'):
    html_code = f"""
    <style>
    .styled-text {{
        font-size: 24px;
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

#----------------------------------------------

#------Title CSS-------------#
def center_title(title_text):
    html_code = f"""
    <style>
    .center-title {{
        text-align: center;
        font-size: 48px;
        font-weight: bold;
        margin: 20px 0; /* Adds space above and below the title */
    }}
    </style>
    <div class="center-title">
        {title_text}
    </div>
    """
    st.markdown(html_code, unsafe_allow_html=True)

#--------------------------------#



#------Main Program for Calculation----------------#

# Function to ensure datetime columns are in proper format
def ensure_datetime_format(df, datetime_col):
    try:
        df[datetime_col] = pd.to_datetime(df[datetime_col])
    except (ValueError, TypeError):
        st.error(f"Could not convert {datetime_col} to datetime format.")
        return None
    return df

# Function to fill missing datetime rows with zeroes and rename columns
def fill_missing_rows(df, datetime_col, energy_col):
    # Ensure datetime column is in proper format
    df = ensure_datetime_format(df, datetime_col)
    if df is None:
        return None, None

    # Create a complete datetime range
    full_datetime_range = pd.date_range(start=df[datetime_col].min(), end=df[datetime_col].max(), freq='H')
    
    # Create a DataFrame with the full datetime range
    full_df = pd.DataFrame({datetime_col: full_datetime_range})

    # Merge with the original DataFrame to identify missing rows
    merged_df = full_df.merge(df, on=datetime_col, how='left')

    # Identify missing rows
    missing_rows = merged_df[merged_df[energy_col].isnull()][datetime_col].tolist()

    # Fill missing energy consumption values with zeroes
    merged_df[energy_col].fillna(0, inplace=True)
    
    # Rename the energy column to 'energy_comp_kWh'
    merged_df.rename(columns={energy_col: 'energy_comp_kWh'}, inplace=True)
    
    return merged_df, missing_rows

# Function to calculate analytics

def calculate_analytics(df, datetime_col, energy_col):
    analytics = {}
    
    # Average hourly energy consumption for the year
    hourly_consumption = df.groupby(df[datetime_col].dt.hour)[energy_col].mean().reset_index()
    hourly_consumption.columns = ['hour', 'average_hourly_consumption']
    analytics['hourly_consumption'] = hourly_consumption.round(2)
    
    # Average daily energy consumption for the year
    df['date'] = df[datetime_col].dt.date
    daily_consumption = df.groupby('date')['energy_comp_kWh'].sum().reset_index()
    analytics['average_daily_consumption'] = daily_consumption['energy_comp_kWh'].mean().round(2)
    
    # Max, min, and average hourly energy demand
    analytics['max_hourly_demand'] = df[energy_col].max()
    analytics['min_hourly_demand'] = df[energy_col][df[energy_col] > 0].min()
    analytics['average_hourly_demand'] = df[energy_col].mean()

    # Monthly average, max, and min daily energy consumption
    df['date'] = df[datetime_col].dt.date
    daily_consumption = df.groupby('date')['energy_comp_kWh'].sum().reset_index()
    daily_consumption['month'] = pd.to_datetime(daily_consumption['date']).dt.strftime('%b')
    daily_consumption['month_num'] = pd.to_datetime(daily_consumption['date']).dt.month

    monthly_stats = daily_consumption.groupby(['month', 'month_num'])['energy_comp_kWh'].agg(['mean', 'max', lambda x: x[x > 0].min()]).reset_index()
    monthly_stats.columns = ['month', 'month_num', 'average_daily_consumption', 'max_daily_consumption', 'min_daily_consumption']
    monthly_stats = monthly_stats.sort_values('month_num')
    monthly_stats = monthly_stats.drop('month_num', axis=1)

    analytics['monthly_stats'] = monthly_stats.round(2)

    # Monthly total energy consumption
    monthly_consumption = df.groupby(df[datetime_col].dt.to_period('M'))[energy_col].sum().reset_index()
    monthly_consumption.columns = ['month', 'total_monthly_consumption']
    monthly_consumption['month'] = monthly_consumption['month'].dt.strftime('%b')
    monthly_consumption['month_num'] = pd.to_datetime(monthly_consumption['month'], format='%b').dt.month
    monthly_consumption = monthly_consumption.sort_values('month_num')
    monthly_consumption = monthly_consumption.drop('month_num', axis=1)

    analytics['monthly_consumption'] = monthly_consumption.round(2)

    # Yearly total energy consumption
    analytics['yearly_total_consumption'] = df[energy_col].sum().round(2)
    
    return analytics



# Streamlit App

#------Header Section-------#

st.title('Energy Data Updater and Analytics')




#-----Rest of the display Elements-------#

if 'cover_page_submitted' not in st.session_state:
        st.session_state.cover_page_submitted = False

# Upload the user dataset
uploaded_file = st.file_uploader("Choose a dataset Excel file", type=["xlsx"])

# Upload the company logo
logo_file = st.file_uploader("Choose a company logo (PNG/JPEG)", type=["png", "jpeg", "jpg"])

# Cover page form
with st.form(key='cover_page_form'):
    col1, col2 = st.columns(2)

    with col1:
        title = st.text_input('Title', 'Energy Consumption and Solar Analytics Report')
        company_name = st.text_input('Company Name', '')
        contact_details = st.text_input('Contact Details (Email, Phone)', '')

    with col2:
        client_name = st.text_input('Client Name', '')
        project_title = st.text_input('Project Title', '')

    cover_page_submitted = st.form_submit_button("Submit")

    if cover_page_submitted:
        form_data = {
            'title': title,
            'company_name': company_name,
            'client_name': client_name,
            'contact_details': contact_details,
            'project_title': project_title,
        }
        save_to_firebase(form_data)
        st.success('Form data saved successfully!')
        
        st.session_state.cover_page_submitted = True
        st.session_state.title = title
        st.session_state.company_name = company_name
        st.session_state.client_name = client_name
        st.session_state.contact_details = contact_details
        st.session_state.project_title = project_title


if uploaded_file:
    input_df = pd.read_excel(uploaded_file)
    #st.write("Input DataFrame:")
    #st.write(input_df)

    if input_df.shape[1] < 2:
        st.error("The uploaded file must have at least two columns: datetime and energy consumption.")
    else:
        datetime_col = input_df.columns[0]
        energy_col = input_df.columns[1]

        # Fill missing rows and update 'energy_comp_kWh'
        col1,col2,col3=st.columns([1.5,1.5,.75])
        filled_df, missing_rows = fill_missing_rows(input_df, datetime_col, energy_col)

        if filled_df is not None:
            #col2.write("Updated DataFrame:")
            #col2.write(filled_df)

            if missing_rows:
                col2.write("Missing rows added:")
                for row in missing_rows:
                    col2.write(row)

            # Provide download option for the updated dataset
            output = BytesIO()
            filled_df.to_excel(output, index=False)
            output.seek(0)

            
            col2.download_button(
                label="Download Updated Dataset",
                data=output,
                file_name='updated_energy_data.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )

            # Calculate and display analytics
            analytics = calculate_analytics(filled_df, datetime_col, 'energy_comp_kWh')
            
            # Display average daily energy consumption
            st.write('\n')
            st.write('\n')
            st.write('\n')
            styled_text_block(f"Average Daily Energy Consumption:", color='#008080', background_color='#DBEAFE')
            col1,col2,col3=st.columns([1.5,1.5,.75])
            col2.write(f"### {analytics['average_daily_consumption']} kWh")
            

            st.write('\n')
            st.write('\n')
            daily_avg_24h = analytics['average_daily_consumption']
            
            #st.write('Test Monthly df',analytics['monthly_consumption'])

            
            # Display average hourly energy consumption
            
            styled_text_block("Average Hourly Energy Consumption (kWh):", color='#333333', background_color='#FFFFE0')
            #st.dataframe(analytics['hourly_consumption'])
            # Display the dataframe as a scrollable HTML table
            hourly_consumption_html = analytics['hourly_consumption'].to_html(classes='scroll-table')
            
            scroll_table_style = """
            <style>
            .scroll-table {
                width: 100%;
                border-collapse: collapse;
            }
            
            .scroll-table th, .scroll-table td {
                border: 1px solid #ddd;
                padding: 8px;
            }
            
            .scroll-table th {
                padding-top: 12px;
                padding-bottom: 12px;
                text-align: left;
                background-color: #FF7F00;
                color: white;
            }
            
            .scroll-table-container {
                max-height: 600px; /* Adjust this value as needed */
                overflow-y: auto;
                overflow-x: hidden;
                border: 1px solid #ddd;
            }
            
            .scroll-table-container table {
                width: 100%;
                table-layout: fixed;
            }
            </style>
            """
            
            st.markdown(scroll_table_style, unsafe_allow_html=True)
            
            scroll_table_html = f"""
            <div class="scroll-table-container">
                {hourly_consumption_html}
            </div>
            """
            
            st.markdown(scroll_table_html, unsafe_allow_html=True)

            st.write('\n')
            st.write('\n')
            st.write('\n')
            

            # Plot hourly distribution of energy consumption
            fig = px.bar(
                analytics['hourly_consumption'],
                x='hour',
                y='average_hourly_consumption',
                title='Hourly Distribution of Energy Consumption',
                labels={'hour': 'Hour of Day', 'average_hourly_consumption': 'Average Hourly Consumption (kWh)'},
                color_discrete_sequence=['#228B22']  
            )
            
            # Center the title
            fig.update_layout(
                title={
                    'text': 'Hourly Distribution of Energy Consumption',
                    'y': 0.9,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                margin=dict(r=100)  # Add space on the right side
            )
            
            st.plotly_chart(fig)

            
            # Function to plot hourly distribution of energy consumption using Matplotlib
            def plot_hourly_distribution(hourly_consumption):
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Customize the bar color and edge color
                bars = ax.bar(hourly_consumption['hour'], hourly_consumption['average_hourly_consumption'], color='#FF6F61', edgecolor='black')
                
                # Add data labels inside the bars
                for bar in bars:
                    yval = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2.0, yval - (yval * 0.05), round(yval, 2), va='top', ha='center', color='white', fontsize=9, rotation=90)
                
                # Customize the title and labels
                ax.set_title('Hourly Distribution of Energy Consumption', fontsize=16, fontweight='bold', color='#333333')
                ax.set_xlabel('Hour of Day', fontsize=14, fontweight='bold', color='#333333')
                ax.set_ylabel('Average Hourly Consumption (kWh)', fontsize=14, fontweight='bold', color='#333333')
                
                # Customize the x-ticks
                ax.set_xticks(range(0, 24))
                ax.set_xticklabels(range(0, 24), fontsize=12, color='#333333')
                
                # Customize the y-ticks
                ax.yaxis.set_tick_params(labelsize=12, color='#333333')
                
                # Customize the grid
                ax.grid(True, linestyle='--', alpha=0.7, color='#cccccc')
                
                # Set background color
                ax.set_facecolor('#f9f9f9')
                fig.patch.set_facecolor('#f9f9f9')
                
                plt.tight_layout()
                return fig


            
            # Create and display the plot
            hourly_distribution_fig = plot_hourly_distribution(analytics['hourly_consumption'])
            #st.pyplot(hourly_distribution_fig)
            
            # Save the plot as an image
            hourly_distribution_plot_path = 'hourly_distribution_plot.png'
            hourly_distribution_fig.savefig(hourly_distribution_plot_path)
            
            
            # Display a message confirming the image was saved
            #st.write(f"Hourly distribution plot saved as {hourly_distribution_plot_path}")





            
            styled_text_block("Daily  Max, Min & Average Energy Demand", color='#333333', background_color='#FFFFE0')
            
            col1,col2,col3=st.columns(3)
            col1.metric(label="Max Hourly Demand", value=f"{analytics['max_hourly_demand']:.2f} kW")
            col2.metric(label="Min Hourly Demand", value=f"{analytics['min_hourly_demand']:.2f} kW")
            col3.metric(label="Average Hourly Demand", value=f"{analytics['average_hourly_demand']:.2f} kW")

            styled_text_block("Month-wise Average, Max, and Min Daily Energy Consumption", color='#333333', background_color='#FFFFE0')

            
            
            html_monthly_table=analytics['monthly_stats'].to_html(index=True)
            centered_table = f"""
            <div style="display: flex; justify-content: center;text-align: center;">
                {html_monthly_table}
            </div>
            """
            
            st.markdown(centered_table, unsafe_allow_html=True)
            

            
            styled_text_block("Monthly Energy Consumption for All 12 Months", color='#333333', background_color='#FFFFE0')
            html_monthly_comp_table=analytics['monthly_consumption'].to_html(index=True)
            centered_table_2 = f"""
            <div style="display: flex; justify-content: center;text-align: center;">
                {html_monthly_comp_table}
            </div>
            """
            
            st.markdown(centered_table_2, unsafe_allow_html=True)
            

            # Plot the monthly energy consumption using matplotlib
            fig, ax = plt.subplots()
            ax.bar(analytics['monthly_consumption']['month'], analytics['monthly_consumption']['total_monthly_consumption'])
            ax.set_xlabel('Month')
            ax.set_ylabel('Total Monthly Consumption (kWh)')
            ax.set_title('Monthly Energy Consumption')
            
            # Save plot to a file
            plot_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
            plt.savefig(plot_path)
            plt.close(fig)

            # Read the image into memory
            with open(plot_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode()

            # Streamlit sliders
            st.write("### Adjusted Energy Metrics")
            slider1 = st.slider("Adjust Max Hourly Energy Demand (%)", 0, 100, 50)
            slider2 = st.slider("Adjust Yearly Total Energy Demand (%)", 0, 100, 80)

            adjusted_max_hourly_demand = analytics['max_hourly_demand'] * (slider1 / 100)
            adjusted_yearly_total_consumption = analytics['yearly_total_consumption'] * (slider2 / 100)

            col1, col2 = st.columns(2)
            col1.metric(label="Adjusted Max Hourly Demand", value=f"{adjusted_max_hourly_demand:.2f} kWh")
            col2.metric(label="Adjusted Yearly Total Consumption", value=f"{adjusted_yearly_total_consumption:.2f} kWh")


            st.write('_______')
            #Solar Part#

            st.write('\n')



            # Solar Analytics
            # Form for Solar Analytics
            
            center_title('Solar Analytics')
            
            module_efficiency = 0.215
            optimized_pv_capacity = 0
            yearly_total_production = 0
            yearly_energy_consumption = 0
            monthly_merged_df = pd.DataFrame()
            excess_monthly_production_df = pd.DataFrame()
            #daily_production_plot_path = None
            
            # Form for user input
            st.write('\n')
            styled_text_block("Input Parameters", color='#333333', background_color='#FFFFE0')
            
            with st.form(key='my_form'):
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    latitude = st.number_input('Enter latitude', min_value=-90.0, max_value=90.0, value=22.7)
                with col2:
                    longitude = st.number_input('Enter longitude', min_value=-180.0, max_value=180.0, value=73.7)
                with col3:
                    all_timezones = pytz.all_timezones
                    default_ix = all_timezones.index('Asia/Calcutta')
                    tz_str = st.selectbox("Select your Time Zone", all_timezones, index=default_ix)
            
                col4, col5, col6 = st.columns(3)
                with col4:
                    surface_tilt = st.slider('Enter surface tilt', 0.0, 90.00, 20.00, 0.10)
                with col5:
                    surface_azimuth = st.slider('Enter surface azimuth', -180, 180, 180, 1)
                with col6:
                    pr = st.slider('Plant Performance ratio (%)', 0.0, 100.00, 81.50, 0.50, format='%f') / 100
            
                submitted = st.form_submit_button("Submit")
            
            if submitted:
                module_efficiency = 21.15 / 100
                total_area = 2.374596 * 2  # Assuming 2 modules
            
                def calculate_solar_production(latitude, longitude, tz_str, surface_tilt, surface_azimuth, module_efficiency, pr, total_area):
                    site = location.Location(latitude, longitude, tz=tz_str)
                    times = pd.date_range(start='2021-01-01', end='2021-12-31', freq='H', tz=tz_str)[:-1]
                    solar_position = site.get_solarposition(times)
                    irrad_data = site.get_clearsky(times)
                    poa_irrad = irradiance.get_total_irradiance(surface_tilt, surface_azimuth, solar_position['apparent_zenith'], solar_position['azimuth'], irrad_data['dni'], irrad_data['ghi'], irrad_data['dhi'])
                    hourly_production = (poa_irrad['poa_global'] / 1000) * module_efficiency * pr * total_area
                    return hourly_production, poa_irrad
            
                hourly_production, poa_irrad = calculate_solar_production(latitude, longitude, tz_str, surface_tilt, surface_azimuth, module_efficiency, pr, total_area)
            
                # Ensure the datetime column is in proper format and timezone-aware
                filled_df[datetime_col] = pd.to_datetime(filled_df[datetime_col]).dt.tz_localize(None)
                poa_irrad.index = poa_irrad.index.tz_localize(None)
                
                # Align both series to a generic year before merging
                def align_to_generic_year(series):
                    series.index = series.index.map(lambda x: x.replace(year=1990))
                    return series
                
                # Align the dataframes to a generic year
                aligned_energy_df = filled_df.set_index(datetime_col)
                aligned_energy_df.index = aligned_energy_df.index.map(lambda x: x.replace(year=1990))
                aligned_poa_irrad = poa_irrad.copy()
                aligned_poa_irrad.index = aligned_poa_irrad.index.map(lambda x: x.replace(year=1990))
                
                # Merge aligned dataframes
                combined_df = aligned_energy_df.merge(aligned_poa_irrad['poa_global'], left_index=True, right_index=True, how='left')
                
                # Set energy_comp_kWh to zero where poa_global is zero
                combined_df.loc[combined_df['poa_global'] == 0, 'energy_comp_kWh'] = 0
                #st.write(combined_df)
                
                #---------------------------
                #Avg Hourly Energy consumption during the day
                hourly_energy_consumption = combined_df['energy_comp_kWh']
                avg_hourly_consumption = hourly_energy_consumption.groupby(hourly_energy_consumption.index.hour).mean().reset_index()
                avg_hourly_consumption.columns = ['hour', 'average_hourly_consumption']
                
                # Resample daily data Energy Comp
                #Monthly and Yearly Energy Consumption Profile with Generic Year
                daily_energy_consumption = combined_df['energy_comp_kWh'].resample('D').sum()
                daily_energy_consumption.index = daily_energy_consumption.index.tz_localize(None)
                daily_energy_consumption = align_to_generic_year(daily_energy_consumption)
                avg_daily_consumption = daily_energy_consumption.mean()
                daily_avg_sun = avg_daily_consumption
                
                monthly_energy_consumption = daily_energy_consumption.resample('M').sum().round(2)
                yearly_energy_consumption = monthly_energy_consumption.sum()
                
                #
                #st.write('#monthly energy comp generic year',monthly_energy_consumption)
                #
                
                #-----------------------------
                # Resample daily production
                #Specific Solar Energy production Calculation
                
                daily_production = hourly_production.resample('D').sum()
                daily_production.index = daily_production.index.tz_localize(None)
                daily_production = align_to_generic_year(daily_production)
                monthly_production = daily_production.resample('M').sum()
                yearly_spec_prod=daily_production.sum()
                #
                #st.write('#### Specific Solar Energy Yield (kWh/kWp/month)')
                #st.table(monthly_production)
                
                # Merge daily data for final comparison
                daily_combined_df = daily_production.to_frame(name='Daily_Solar_Production').merge(
                    daily_energy_consumption.to_frame(name='Daily_Energy_Consumption'), left_index=True, right_index=True, how='inner'
                )
                
                # Display merged dataframe
                #st.write(daily_combined_df)
                
                #----------------
                # Smooth the energy consumption profile using a moving average
                daily_energy_consumption_smoothed = daily_energy_consumption.rolling(window=30, center=True).mean()

                #
                #st.write('#daily smoothed energy comp generic year',daily_energy_consumption_smoothed)
                #----------------


                
                # Calculate monthly PV capacities
                monthly_pv_capacities = monthly_energy_consumption / monthly_production
            
                # Calculate the mean value of these 12 PV capacities
                optimized_pv_capacity = monthly_pv_capacities.mean()
                
                #Optimized PV Cap Energy Production - daily, Monthly, Yearly
                
                #Average Hourly Total Production Calculation--------------
                hourly_total_production = hourly_production*optimized_pv_capacity
                avg_hourly_production = hourly_total_production.groupby(hourly_total_production.index.hour).mean().reset_index()
                avg_hourly_production.columns = ['hour', 'average_hourly_production']
                #----------------------------------
                daily_total_production=daily_production*optimized_pv_capacity
                monthly_total_production=daily_total_production.resample('M').sum().round(2)
                yearly_total_production=daily_total_production.sum()
                
                
                #----#Energy Consumption Hourly Distribution--------#
                # Plot hourly distribution of energy consumption
                def plot_comp_hourly_distribution(avg_hourly_consumption):
                    fig, ax1 = plt.subplots(figsize=(12, 8))
                
                    bar_width = 0.55  # Width of the bars
                    index = avg_hourly_consumption['hour']
                
                    # Plot energy consumption bars
                    bars1 = ax1.bar(index - bar_width / 2, avg_hourly_consumption['average_hourly_consumption'], bar_width, color='#FFA500', edgecolor='black', label='Energy Consumption')
                    for bar in bars1:
                        yval = bar.get_height()
                        ax1.text(bar.get_x() + bar.get_width() / 2.0, yval - (yval * 0.05), round(yval, 2), va='top', ha='center', color='white', fontsize=10, rotation=90)
                
                    ax1.set_xlabel('Hour of Day', fontsize=14, fontweight='bold', color='#333333')
                    ax1.set_ylabel('Average Hourly Consumption (kWh)', fontsize=14, fontweight='bold', color='#333333')
                    ax1.set_xticks(range(0, 24))
                    ax1.set_xticklabels(range(0, 24), fontsize=12, color='#333333')
                    ax1.yaxis.set_tick_params(labelsize=12, color='#333333')
                    ax1.set_facecolor('#f9f9f9')
                    ax1.grid(True, linestyle='--', alpha=0.7, color='#cccccc')
                
                    fig.suptitle('Hourly Distribution of Energy Consumption When Sun is Up', fontsize=16, fontweight='bold', color='#333333')
                    fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=2, fontsize=12)
                
                    plt.tight_layout()
                    return fig


                comp_distribution_plot = plot_comp_hourly_distribution(avg_hourly_consumption)
                st.pyplot(comp_distribution_plot)
                
                # Add a download option for the plot
                comp_distribution_plot_path = "comp_distribution_plot.png"
                comp_distribution_plot.savefig(comp_distribution_plot_path)

                
                
                
                #---------Display of Solar Pv capacity--------#
                # Enhanced CSS for the centered text block with reduced width and hover effect
                centered_text_style = """
                    <style>
                    .centered-text-container {
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        flex-direction: column;
                        text-align: center;
                        background-color: #ffffff;
                        border-radius: 10px;
                        padding: 30px;
                        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                        margin: 20px auto;
                        max-width: 400px; /* Adjust the width as needed */
                        transition: transform 0.3s ease, box-shadow 0.3s ease;
                    }
                    .centered-text-container:hover {
                        transform: translateY(-5px);
                        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
                    }
                    .centered-text-container h3 {
                        margin: 0;
                        font-size: 24px;
                        color: #555;
                    }
                    .centered-text-container h2 {
                        margin: 10px 0 0;
                        font-size: 32px;
                        font-weight: bold;
                        color: #000;
                    }
                    </style>
                """
                
                # HTML for centered text block
                centered_text_html = f"""
                    <div class="centered-text-container">
                        <h3>Optimized PV Capacity</h3>
                        <h2>{optimized_pv_capacity:.2f} kWp</h2>
                    </div>
                """
                
                # Render CSS and HTML in Streamlit
                st.markdown(centered_text_style, unsafe_allow_html=True)
                st.markdown(centered_text_html, unsafe_allow_html=True)


                #---------------------------#
                #------Hourly Average Solar Production/ Distribution Plot-------#
                
                def plot_solar_hourly_distribution(avg_hourly_production):
                    fig, ax1 = plt.subplots(figsize=(12, 8))
                
                    bar_width = 0.55  # Width of the bars
                    index = avg_hourly_production['hour']
                
                    # Plot energy production bars
                    bars1 = ax1.bar(index - bar_width/2, avg_hourly_production['average_hourly_production'], bar_width, color='#1E90FF', edgecolor='black', label='Solar Energy production')
                    for bar in bars1:
                        yval = bar.get_height()
                        ax1.text(bar.get_x() + bar.get_width()/2.0, yval - (yval * 0.05), round(yval, 2), va='top', ha='center', color='white', fontsize=10, rotation=90)
                
                    
                    ax1.set_xlabel('Hour of Day', fontsize=14, fontweight='bold', color='#333333')
                    ax1.set_ylabel('Average Hourly Production (kWh)', fontsize=14, fontweight='bold', color='#333333')
                    ax1.set_xticks(range(0, 24))
                    ax1.set_xticklabels(range(0, 24), fontsize=12, color='#333333')
                    ax1.yaxis.set_tick_params(labelsize=12, color='#333333')
                    ax1.set_facecolor('#f9f9f9')
                    ax1.grid(True, linestyle='--', alpha=0.7, color='#cccccc')
                
                    
                
                    fig.suptitle('Hourly Distribution of Solar Energy production', fontsize=16, fontweight='bold', color='#333333')
                    fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=2, fontsize=12)
                
                    plt.tight_layout()
                    return fig


                
                solar_distribution_plot = plot_solar_hourly_distribution(avg_hourly_production)
                st.pyplot(solar_distribution_plot)
                
                # Add a download option for the plot
                solar_distribution_plot_path = "solar_distribution_plot.png"
                solar_distribution_plot.savefig(solar_distribution_plot_path)
                
                
                #daily Optimized Capapcity energy production
                daily_total_production_df = daily_total_production.to_frame(name='Daily_Solar_Production')
                daily_energy_consumption_df = daily_energy_consumption.to_frame(name='Daily_Energy_Consumption')
                # Merge DataFrames
                daily_merged_df = daily_total_production_df.merge(daily_energy_consumption_df, left_index=True, right_index=True, how='inner')
                #st.write("### Daily Energy Consumption Vs Solar Energy Production (kWh)")
                #st.dataframe(daily_merged_df)
                
                #Monthly Optimized capacity Energy Vs enrgy comp profile
                monthly_total_production_df = monthly_total_production.to_frame(name='Monthly_Solar_Production')
                monthly_energy_consumption_df = monthly_energy_consumption.to_frame(name='Monthly_Energy_Consumption')
                # Merge DataFrames
                monthly_merged_df = monthly_total_production_df.merge(monthly_energy_consumption_df, left_index=True, right_index=True, how='inner')
                # Convert the datetime index to month names
                monthly_merged_df.index = monthly_merged_df.index.strftime('%B')
                
                #---------Display of Merged Monthly Energy data--------#
                html_monthly_merged_df=monthly_merged_df.to_html(index=True)
                styled_text_block("Monthly Energy Consumption Vs Solar Energy Production (kWh)", color='#333333', background_color='#FFFFE0')
                
                # Center the table using HTML and CSS
                centered_table = f"""
                <div style="display: flex; justify-content: center;text-align: center;">
                    {html_monthly_merged_df}
                </div>
                """
                
                st.markdown(centered_table, unsafe_allow_html=True)
                #----------------------------------------------------------#
                
                #--------Dsiplay of Annual Values as Metrics----------#
                #Yearly Optimized capacity energy yield and Energy comp profile
                # Enhanced CSS for aligning and styling metrics
                metric_style = """
                    <style>
                    .metric-container {
                        display: flex;
                        justify-content: space-between;
                        padding: 20px 0;
                    }
                    .metric-container div {
                        width: 45%;
                        text-align: center;
                        background-color: #FFF8DC;
                        border-radius: 10px;
                        padding: 30px;
                        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                        transition: transform 0.2s;
                    }
                    .metric-container div:hover {
                        transform: scale(1.05);
                    }
                    .metric-container p {
                        margin: 20px 0 0;
                        font-size: 28px;
                        font-weight: bold;
                        color: #000;
                    }
                    .header-container {
                        display: flex;
                        justify-content: space-between;
                        padding: 0 0 20px 0;
                    }
                    .header-left, .header-right {
                        width: 45%;
                        text-align: center;
                        font-size: 18px;
                        color: #555;
                        background-color: #e6e6e6;
                        padding: 10px;
                        border-radius: 5px;
                    }
                    </style>
                """
                
                # HTML for metrics without headings
                metric_html = f"""
                    <div class="metric-container">
                        <div>
                            <p>{yearly_total_production:.2f} kWh</p>
                        </div>
                        <div>
                            <p>{yearly_energy_consumption:.2f} kWh</p>
                        </div>
                    </div>
                """
                
                # HTML for headings
                header_html = """
                    <div class="header-container">
                        <div class="header-left">Total Yearly Solar Energy Yield</div>
                        <div class="header-right">Yearly Consumption - Day-time</div>
                    </div>
                """
                
                # Render CSS and HTML in Streamlit
                st.markdown(metric_style, unsafe_allow_html=True)
                st.markdown(header_html, unsafe_allow_html=True)
                st.markdown(metric_html, unsafe_allow_html=True)
                
                
                #------------------
                
                
                # ---------Calculate the excess solar energy production----------#
                excess_daily_production = daily_production*optimized_pv_capacity - daily_energy_consumption
                excess_daily_production[excess_daily_production < 0] = 0
                excess_monthly_production = excess_daily_production.resample('M').sum().round(2)
                excess_monthly_production_df = excess_monthly_production.to_frame(name='Monthly_Excess Solar_Yield_kWh')
                excess_monthly_production_df.index = excess_monthly_production_df.index.strftime('%B')
                total_excess_yearly_production = excess_monthly_production.sum()
                
                #--------Display Excess Solar Energy Gen data---------#
                # Enhanced CSS for the centered text block with reduced width
                centered_text_style_2 = """
                    <style>
                    .centered-text-container {
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        flex-direction: column;
                        text-align: center;
                        background-color: #ffffff;
                        border-radius: 10px;
                        padding: 30px;
                        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                        margin: 20px auto;
                        max-width: 800px; /* Adjust the width as needed */
                        transition: transform 0.3s ease, box-shadow 0.3s ease;
                    }
                    .centered-text-container:hover {
                        transform: translateY(-5px);
                        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
                    }
                    .centered-text-container h3 {
                        margin: 0;
                        font-size: 24px;
                        color: #555;
                    }
                    .centered-text-container h2 {
                        margin: 10px 0 0;
                        font-size: 32px;
                        font-weight: bold;
                        color: #000;
                    }
                    </style>
                """
                
                                
                # HTML for centered text block
                centered_text_html_2 = f"""
                    <div class="centered-text-container">
                        <h3>Yearly Excess Solar Energy Production</h3>
                        <h2>{total_excess_yearly_production:.2f} kWh</h2>
                    </div>
                """
                
                # Render CSS and HTML in Streamlit
                st.markdown(centered_text_style_2, unsafe_allow_html=True)
                st.markdown(centered_text_html_2, unsafe_allow_html=True)
                
                
                
                
                #---------Display of Merged Monthly Energy data--------#
                html_excess_monthly_production_df=excess_monthly_production_df.to_html(index=True)
                
                styled_text_block("Excess Solar Energy Generation -Ready to Export to Grid (kWh)", color='#333333', background_color='#FFFFE0')
                st.write('\n')
                # Center the table using HTML and CSS
                centered_table = f"""
                <div style="display: flex; justify-content: center;text-align: center;">
                    {html_excess_monthly_production_df}
                </div>
                """
                
                st.markdown(centered_table, unsafe_allow_html=True)
                st.write('\n')
                st.write('\n')
                st.write('\n')
                #----------------------------------------------------------#
                #st.write("Excess Daily Solar Energy Production:")
                #st.write(excess_daily_production)
            
                
                
                #--------------------------
            
                # Plotting the results
                fig7, ax3 = plt.subplots(figsize=(10, 6))
                ax3.plot(daily_total_production.index, daily_total_production, 'g-', label='Daily Solar Energy Production (kWh)')
                ax3.plot(daily_energy_consumption.index, daily_energy_consumption, 'b-', label='Daily Energy Consumption (kWh)')
                ax3.set_xlabel('Date')
                ax3.set_ylabel('Energy (kWh)')
                ax3.legend(loc='upper left')
                ax3.set_title('Daily Solar Energy Production vs Daily Energy Consumption')
                st.pyplot(fig7)
            
                fig8, ax4 = plt.subplots(figsize=(10, 6))
                ax4.bar(monthly_total_production.index.strftime('%b'), monthly_total_production, color='g', label='Monthly Solar Energy Production (kWh)')
                ax4.plot(monthly_energy_consumption.index.strftime('%b'), monthly_energy_consumption, 'b-', label='Monthly Energy Consumption (kWh)')
                ax4.set_xlabel('Month')
                ax4.set_ylabel('Energy (kWh)')
                ax4.legend(loc='upper left')
                ax4.set_title('Monthly Solar Energy Production vs Monthly Energy Consumption')
                st.pyplot(fig8)
            
                # Save plot to a file
                daily_production_plot_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
                plt.savefig(daily_production_plot_path)
                plt.close(fig8)

                # Read the image into memory
                with open(daily_production_plot_path, "rb") as image_file:
                    encoded_image = base64.b64encode(image_file.read()).decode()

                
                
                # Plot the excess monthly production
                fig3 = go.Figure()
                
                # Bar plot for excess optimized monthly solar energy production
                fig3.add_trace(go.Bar(
                    x=excess_monthly_production.index.strftime('%b'),
                    y=excess_monthly_production,
                    name="Excess Optimized Monthly Solar Energy Production",
                    marker_color='orange'
                ))
                
                fig3.update_layout(
                    title_text='Excess Monthly Solar Energy Production (kWh)',
                    xaxis_title="Month",
                    yaxis_title="Excess Energy (kWh)",
                    barmode='group'
                )
                
                st.plotly_chart(fig3)
                
                #-------PDF Report Generation---------#
                

                #
                
                
                
                
                
                # Generate PDF report
                
                class PDFReport(FPDF):
                    def __init__(self, logo_path=None):
                        super().__init__()
                        self.logo_path = logo_path
                        self.chapter_number = 0
                        self.sub_chapter_number = 0
                
                    def header(self):
                        if self.logo_path:
                            self.image(self.logo_path, 10, 6, 33)  # Adjust y-position to 6
                        self.set_font('Arial', 'B', 12)
                        self.cell(0, 10, 'Energy Consumption and Solar Analytics Report', 0, 1, 'C')
                        self.set_font('Arial', 'I', 10)
                        self.cell(0, 10, 'Generated by Energy Data Updater and Analytics App', 0, 1, 'C')
                        self.ln(5)
                        self.set_line_width(0.5)
                        self.line(10, self.get_y(), 200, self.get_y())
                        self.ln(5)
                
                    def footer(self):
                        self.set_y(-15)
                        self.set_font('Arial', 'I', 8)
                        self.cell(0, 5, f'Page {self.page_no()}', 0, 1, 'C')
                        self.set_y(-10)  # Adjust position for the link
                        self.cell(0, 5, 'App link: https://energy-eda-v01.streamlit.app/', 0, 0, 'C', link='https://energy-eda-v01.streamlit.app/')

                
                    def chapter_title(self, title):
                        self.chapter_number += 1
                        self.sub_chapter_number = 0
                        self.ln(1)  # Add space above the chapter title
                        self.set_font('Arial', 'B', 14)
                        title_text = f"Chapter {self.chapter_number}: {title}"
                        title_width = self.get_string_width(title_text) + 6
                        doc_width = self.w
                        self.set_x((doc_width - title_width) / 2)
                        self.cell(title_width, 7, title_text, 0, 1, 'C')
                        self.ln(1)
                
                    def sub_chapter_title(self, title):
                        self.sub_chapter_number += 1
                        self.ln(5)  # Add space above the sub-chapter title
                        self.set_font('Arial', 'B', 12)
                        title_text = f"{self.chapter_number}.{self.sub_chapter_number} {title}"
                        title_width = self.get_string_width(title_text) + 6
                        self.set_x(10)
                        self.cell(title_width, 10, title_text, 0, 1, 'L')
                        self.ln(2)
                
                    def add_cover_page(self, title, company_name, client_name, contact_details, project_title):
                        self.add_page()
                    
                        # Create a gradient background from orange to yellow with a bit of red
                        for i in range(297):
                            r = min(255, 255 - i // 6)
                            g = min(255, 215 + i // 10)
                            b = max(255, 240 - i // 6)
                            self.set_fill_color(r, g, b)
                            self.rect(0, i, 210, 1, 'F')
                    
                        # Add a logo (assuming you have a logo path)
                        if self.logo_path:
                            self.image(self.logo_path, x=10, y=10, w=50)
                    
                        # Title
                        self.set_xy(10, 80)
                        self.set_font('Arial', 'B', 24)
                        self.set_text_color(0, 51, 102)  # Dark Blue
                        self.multi_cell(0, 10, title, 0, 'C', False)
                        self.ln(20)
                    
                        # Company Name
                        self.set_font('Arial', 'B', 18)
                        self.set_text_color(0, 102, 204)  # Lighter Blue
                        self.multi_cell(0, 10, company_name, 0, 'C', False)
                        self.ln(10)
                    
                        # Client Name
                        self.set_font('Arial', 'I', 16)
                        self.set_text_color(0, 0, 0)  # Black
                        self.multi_cell(0, 10, f"Client: {client_name}", 0, 'C', False)
                        self.ln(10)
                    
                        # Project Title
                        self.set_font('Arial', 'B', 16)
                        self.set_text_color(255, 69, 0)  # Red-Orange
                        self.multi_cell(0, 10, project_title, 0, 'C', False)
                        self.ln(10)
                    
                        # Decorative line below the project title
                        self.set_line_width(1.5)
                        self.set_draw_color(0, 51, 102)  # Dark Blue
                        self.line(30, self.get_y(), 180, self.get_y())
                        self.ln(10)
                    
                        # Contact Details
                        self.set_font('Arial', '', 14)
                        self.set_text_color(0, 0, 0)  # Black
                        self.multi_cell(0, 10, f"Contact:\n{contact_details}", 0, 'C', False)
                        self.ln(20)

                
                    def chapter_body(self, body):
                        self.set_font('Arial', '', 12)
                        self.multi_cell(0, 10, body)
                        self.ln()
                    def add_space(pdf, height):
                        pdf.ln(height)
                        

                    def add_plot(self, image, width=None, height=None):
                        if width and height:
                            self.image(image, x=(210 - width) / 2, y=None, w=width, h=height)
                        elif width:
                            self.image(image, x=(210 - width) / 2, y=None, w=width)
                        elif height:
                            self.image(image, x=(210 - 190) / 2, y=None, h=height)
                        else:
                            self.image(image, x=(210 - 190) / 2)
                
                    def add_table(self, table_data, col_widths):
                        table_height = 10 * len(table_data)
                        if self.get_y() + table_height > 300:
                            self.add_page()
                        table_width = sum(col_widths)
                        start_x = (210 - table_width) / 2
                        self.set_font('Arial', '', 10)
                
                        self.set_fill_color(200, 220, 255)
                        self.set_text_color(0)
                        self.set_draw_color(0, 0, 0)
                        self.set_line_width(0.3)
                        self.set_font('Arial', 'B', 11)
                        self.set_x(start_x)
                        for i, header in enumerate(table_data[0]):
                            self.cell(col_widths[i], 5, str(header), border=1, align='C', fill=True)
                        self.ln()
                
                        self.set_font('Arial', '', 10)
                        self.set_fill_color(240, 240, 240)
                        fill = False
                        for row in table_data[1:]:
                            self.set_x(start_x)
                            for i, cell in enumerate(row):
                                if isinstance(cell, (int, float)):
                                    cell = f"{cell:.2f}"
                                elif isinstance(cell, pd.Timestamp):
                                    cell = cell.strftime('%Y-%m-%d')
                                self.cell(col_widths[i], 5, str(cell), border=1, align='C', fill=fill)
                            self.ln()
                            fill = not fill
                
                    def add_table_with_title(self, title, table_data, col_widths):
                        table_height = 5 * len(table_data) + 5  # Including title height
                        if self.get_y() + table_height > 270:
                            self.add_page()
                        self.sub_chapter_title(title)
                        self.add_table(table_data, col_widths)

                
                    def add_page_if_needed(self):
                        if self.get_y() > 250:
                            self.add_page()
                
                    def add_metrics_table(self, metrics_data):
                        self.add_page_if_needed()
                        self.sub_chapter_title("Max, Min, and Average Hourly Energy Demand")
                    
                        col_widths = [60, 60, 60]
                        table_width = sum(col_widths)
                        start_x = (210 - table_width) / 2
                        
                        # Table header
                        self.set_fill_color(255, 215, 0)  # Background color for header
                        self.set_text_color(139, 0, 0)  # Text color for header
                        self.set_font('Arial', 'B', 12)
                        self.set_x(start_x)
                        self.cell(col_widths[0], 10, "Max Hourly Demand", border=1, align='C', fill=True)
                        self.cell(col_widths[1], 10, "Min Hourly Demand", border=1, align='C', fill=True)
                        self.cell(col_widths[2], 10, "Avg Hourly Demand", border=1, align='C', fill=True)
                        self.ln()
                    
                        # Table values
                        self.set_fill_color(240, 240, 240)  # Background color for values
                        self.set_text_color(0)  # Text color for values
                        self.set_font('Arial', 'B', 14)  # Increase font size for values
                        self.set_x(start_x)
                        self.cell(col_widths[0], 15, f"{metrics_data['max']:.2f} kW", border=1, align='C', fill=True)
                        self.cell(col_widths[1], 15, f"{metrics_data['min']:.2f} kW", border=1, align='C', fill=True)
                        self.cell(col_widths[2], 15, f"{metrics_data['avg']:.2f} kW", border=1, align='C', fill=True)
                        self.ln(20)

                    
                    
                    def add_adjusted_metrics_table(self, adjusted_metrics):
                        self.add_page_if_needed()
                        self.sub_chapter_title("Adjusted Energy Metrics")
                        col_widths = [90, 90]
                        table_width = sum(col_widths)
                        start_x = (210 - table_width) / 2
                    
                        # Header row styling
                        self.set_fill_color(0, 51, 102)  # Amber
                        self.set_text_color(255, 255, 255)  # Dark Gray
                        self.set_font('Arial', 'B', 12)
                        self.set_x(start_x)
                        self.cell(col_widths[0], 10, "Adjusted Max Hourly Demand", border=1, align='C', fill=True)
                        self.cell(col_widths[1], 10, "Adjusted Yearly Total Consumption", border=1, align='C', fill=True)
                        self.ln()
                    
                        # Value row styling
                        self.set_fill_color(224, 224, 224)  # Light Gray
                        self.set_text_color(33, 37, 41)  # Dark Gray
                        self.set_font('Arial', 'B', 14)
                        self.set_x(start_x)
                        self.cell(col_widths[0], 10, f"{adjusted_metrics['max_hourly']:.2f} kW", border=1, align='C', fill=True)
                        self.cell(col_widths[1], 10, f"{adjusted_metrics['yearly_total']:.2f} kWh", border=1, align='C', fill=True)
                        self.ln(12)


                    
                    def add_summary(self, summary):
                        self.add_page_if_needed()
                        self.sub_chapter_title("Summary")
                        
                        # Set font and margins for summary
                        self.set_font('Arial', '', 12)
                        self.set_left_margin(10)
                        self.set_right_margin(10)
                        
                        # Justify text and control line spacing
                        self.set_auto_page_break(auto=True, margin=10)
                        self.multi_cell(0, 6, summary, align='J')  # Adjust line spacing (7) and justify text ('J')
                        
                        # Reset margins to default
                        self.set_left_margin(15)
                        self.set_right_margin(15)



 

                
                def generate_combined_summary(analytics, optimized_pv_capacity, yearly_total_production, yearly_energy_consumption):
                    monthly_stats = analytics['monthly_stats']
                    monthly_consumption = analytics['monthly_consumption']
                
                    # Peak and lowest months for consumption
                    peak_consumption_month = monthly_consumption.loc[monthly_consumption['total_monthly_consumption'].idxmax()]['month']
                    lowest_consumption_month = monthly_consumption.loc[monthly_consumption['total_monthly_consumption'].idxmin()]['month']
                
                    # Peak and lowest months for solar production
                    peak_production_month = monthly_stats.loc[monthly_stats['average_daily_consumption'].idxmax()]['month']
                    lowest_production_month = monthly_stats.loc[monthly_stats['average_daily_consumption'].idxmin()]['month']
                
                    summary_template = """
   This report provides a comprehensive analysis of energy consumption data and solar energy potential. Below is a brief summary of the key findings:
                
                    Energy Consumption Analysis:
                    - The maximum hourly energy demand recorded is {max_hourly_demand:.2f} kW.
                    - The minimum hourly energy demand recorded is {min_hourly_demand:.2f} kW.
                    - The average hourly energy demand is {average_hourly_demand:.2f} kW.
                    
                    Monthly Analysis:
                    - The month with the highest & lowest energy consumption are {peak_consumption_month} & {lowest_consumption_month}.
                    - The month with the highest & lowest solar energy production are {peak_production_month} & {lowest_production_month}.
                
                    Solar Energy Analysis:
                    - Optimized PV Capacity: {optimized_pv_capacity:.2f} kWp
                    - Total Yearly Solar Energy Yield: {yearly_total_production:.2f} kWh
                    - Yearly Total Consumption During the Day: {yearly_energy_consumption:.2f} kWh
                    
   The data indicates the potential for solar energy generation and the optimized capacity to meet the energy consumption demands while minimizing excess generation.
                    """
                
                    summary = summary_template.format(
                        max_hourly_demand=analytics['max_hourly_demand'],
                        min_hourly_demand=analytics['min_hourly_demand'],
                        average_hourly_demand=analytics['average_hourly_demand'],
                        adjusted_max_hourly_demand=adjusted_max_hourly_demand,
                        adjusted_yearly_total_consumption=adjusted_yearly_total_consumption,
                        peak_consumption_month=peak_consumption_month,
                        lowest_consumption_month=lowest_consumption_month,
                        peak_production_month=peak_production_month,
                        lowest_production_month=lowest_production_month,
                        optimized_pv_capacity=optimized_pv_capacity,
                        yearly_total_production=yearly_total_production,
                        yearly_energy_consumption=yearly_energy_consumption
                    )
                
                    return summary


                    
                def add_solar_details_table(pdf, system_details):
                    col_widths = [60, 60]
                    table_data = [
                        ["Parameter", "Value"],
                        ["Latitude", f"{system_details['latitude']}°"],
                        ["Longitude", f"{system_details['longitude']}°"],
                        ["Surface Tilt", f"{system_details['surface_tilt']}°"],
                        ["Surface Azimuth", f"{system_details['surface_azimuth']}°"],
                        ["Module Efficiency", f"{system_details['module_efficiency'] * 100:.2f}%"],
                        ["Performance Ratio", f"{system_details['pr'] * 100:.2f}%"]
                    ]
                    pdf.add_table_with_title("User Defined System Details", table_data, col_widths)
                
                def add_optimized_pv_capacity(pdf, optimized_pv_capacity):
                    # Add some space before the table
                    pdf.ln(2)
                
                    # Define table data
                    table_data = [
                        ["Optimized PV Capacity"],
                        [f"{optimized_pv_capacity:.2f} kWp"]
                    ]
                
                    # Define column widths
                    col_widths = [80]
                
                    # Add table
                    pdf.set_fill_color(240, 240, 240)
                    pdf.set_text_color(0)
                    pdf.set_draw_color(0, 0, 0)
                    pdf.set_line_width(0.3)
                    pdf.set_font('Arial', 'B', 14)
                
                    # Calculate starting x position to center the table
                    table_width = sum(col_widths)
                    start_x = (210 - table_width) / 2
                
                    # Add the table title row
                    pdf.set_x(start_x)
                    pdf.cell(col_widths[0], 10, table_data[0][0], border=1, align='C', fill=True)
                    pdf.ln()
                
                    # Add the table value row
                    pdf.set_font('Arial', 'B', 20)
                    pdf.set_text_color(0, 102, 204)  # Set text color to a standout blue
                    pdf.set_fill_color(255, 255, 204)  # Set background color for the value cell
                    
                    pdf.set_x(start_x)
                    pdf.cell(col_widths[0], 12, table_data[1][0], border=1, align='C', fill=True)
                    pdf.ln(10)


                def add_daily_average_metrics(pdf, daily_avg_24h, daily_avg_sun):
                    pdf.add_page_if_needed()
                    pdf.sub_chapter_title("Daily Average Energy Consumption")
                    
                    col_widths = [90, 90]
                    table_width = sum(col_widths)
                    start_x = (210 - table_width) / 2
                    
                    pdf.set_font('Arial', 'B', 12)
                    pdf.set_x(start_x)
                    pdf.set_fill_color(255, 223, 186)  # Light orange color
                    pdf.set_text_color(0)
                    pdf.set_draw_color(0, 0, 0)
                    pdf.cell(col_widths[0], 10, "24 Hours Average", border=1, align='C', fill=True)
                    pdf.cell(col_widths[1], 10, "Sun Availability Average", border=1, align='C', fill=True)
                    pdf.ln()
                    
                    pdf.set_font('Arial', '', 12)
                    pdf.set_x(start_x)
                    pdf.set_fill_color(255, 255, 255)  # White color for data cells
                    pdf.cell(col_widths[0], 10, f"{daily_avg_24h:.2f} kWh", border=1, align='C', fill=True)
                    pdf.cell(col_widths[1], 10, f"{daily_avg_sun:.2f} kWh", border=1, align='C', fill=True)
                    pdf.ln(5)


                
                def add_key_metrics(pdf, yearly_total_production, yearly_energy_consumption):
                    pdf.add_page_if_needed()
                    pdf.sub_chapter_title("Yearly Energy Yield Vs Consumption Stats")
                    
                    col_widths = [90, 90]
                    table_width = sum(col_widths)
                    start_x = (210 - table_width) / 2
                    
                    pdf.set_font('Arial', 'B', 12)
                    pdf.set_x(start_x)
                    pdf.set_fill_color(255, 223, 186)  # Light orange color
                    pdf.set_text_color(0)
                    pdf.set_draw_color(0, 0, 0)
                    pdf.cell(col_widths[0], 10, "Total Yearly Solar Energy Yield", border=1, align='C', fill=True)
                    pdf.cell(col_widths[1], 10, "Yearly Total Consumption During the Day", border=1, align='C', fill=True)
                    pdf.ln()
                    
                    pdf.set_font('Arial', '', 12)
                    pdf.set_x(start_x)
                    pdf.set_fill_color(255, 255, 255)  # White color for data cells
                    pdf.cell(col_widths[0], 10, f"{yearly_total_production:.2f} kWh", border=1, align='C', fill=True)
                    pdf.cell(col_widths[1], 10, f"{yearly_energy_consumption:.2f} kWh", border=1, align='C', fill=True)
                    pdf.ln(10)
                    
                
                def add_solar_tables_and_charts(pdf, monthly_merged_df, excess_monthly_production_df, daily_production_plot_path):
                    # Monthly Solar Energy Production vs Energy Consumption
                    monthly_table_data = [["Month", "Solar Energy Production (kWh)", "Energy Consumption (kWh)"]]
                    monthly_table_data.extend(monthly_merged_df.reset_index().values.tolist())
                    col_widths = [60, 60, 60]
                    pdf.add_table_with_title("Monthly Solar Energy Production Vs Energy Consumption (kWh)", monthly_table_data, col_widths)
                    pdf.add_page_if_needed()
                    pdf.sub_chapter_title("Monthly Solar Energy Production Vs Energy Consumption (Chart)")
                    pdf.add_plot(daily_production_plot_path, width=160, height=120)
                    
                    # Monthly Excess Solar Energy Production
                    excess_table_data = [["Month", "Excess Solar Energy Production (kWh)"]]
                    excess_table_data.extend(excess_monthly_production_df.reset_index().values.tolist())
                    col_widths = [60, 80]
                    pdf.add_table_with_title("Monthly Excess Solar Energy Production", excess_table_data, col_widths)

                
                
                

                
                
                
                #def generate_pdf(analytics, adjusted_max_hourly_demand, adjusted_yearly_total_consumption, plot_path, logo_file, solar_details, optimized_pv_capacity, yearly_total_production, yearly_energy_consumption, monthly_merged_df, excess_monthly_production_df, daily_production_plot_path):
                def generate_pdf(analytics, adjusted_max_hourly_demand, adjusted_yearly_total_consumption, plot_path, logo_file, solar_details, optimized_pv_capacity, yearly_total_production, yearly_energy_consumption, monthly_merged_df, excess_monthly_production_df, daily_production_plot_path, daily_avg_24h, daily_avg_sun, hourly_distribution_plot_path,comp_distribution_plot_path,solar_distribution_plot_path):
                    # Save the logo file temporarily if provided
                    logo_path = None
                    if logo_file is not None:
                        logo = Image.open(logo_file)
                        logo_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
                        logo.save(logo_path)
                
                    pdf = PDFReport(logo_path=logo_path)
                    
                    # Cover Page
                    # Add cover page if the form was submitted
                    if st.session_state.cover_page_submitted:
                        pdf.add_cover_page(
                            st.session_state.title,
                            st.session_state.company_name,
                            st.session_state.client_name,
                            st.session_state.contact_details,
                            st.session_state.project_title
                            
                        )
                        
                    pdf.add_page()
                    # Energy Consumption Analytics
                    pdf.chapter_title("Energy Consumption Analytics")
                
                    # Adding max, min, and average hourly demand
                    metrics_data = {
                        'max': analytics['max_hourly_demand'],
                        'min': analytics['min_hourly_demand'],
                        'avg': analytics['average_hourly_demand']
                    }
                    pdf.add_metrics_table(metrics_data)
                
                    # Adding adjusted metrics
                    adjusted_metrics = {
                        'max_hourly': adjusted_max_hourly_demand,
                        'yearly_total': adjusted_yearly_total_consumption
                    }
                    pdf.add_adjusted_metrics_table(adjusted_metrics)
                
                    # Adding monthly stats
                    pdf.add_table_with_title(
                        "Month-wise Average, Max, and Min Daily Energy Consumption (kWh)",
                        [["Month", "Avg Daily", "Max Daily", "Min Daily"]] + analytics['monthly_stats'].values.tolist(),
                        [40, 40, 40, 40]
                    )
                
                    # Adding daily average energy consumption
                    add_daily_average_metrics(pdf, daily_avg_24h, daily_avg_sun)
                
                    # Adding monthly energy consumption
                    pdf.add_table_with_title(
                        "Monthly Energy Consumption for All 12 Months",
                        [["Month", "Total Consumption (kWh)"]] + analytics['monthly_consumption'].values.tolist(),
                        [40, 50]
                    )
                
                    # Adding plot on the same page as the table
                    pdf.add_page_if_needed()
                    pdf.sub_chapter_title("Monthly Energy Consumption Plot")
                    pdf.add_plot(plot_path, width=160, height=120)
                    
                    
                    # Plot for Hourly energy consumption Distribution for 24 hours a Day
                    pdf.add_page_if_needed()
                    pdf.sub_chapter_title("Hourly Energy Consumption Distribution - for 24 hours a Day")
                    pdf.add_plot(hourly_distribution_plot_path, width=133.3, height=100)
                    
                    
                    # Plot for Hourly energy consumption Distribution When Sun is Available
                    pdf.add_page_if_needed()
                    pdf.sub_chapter_title("Hourly Energy Consumption Distribution - During the Day")
                    pdf.add_plot(comp_distribution_plot_path, width=133.3, height=100)
                
                
                    # Solar Energy Analytics
                    pdf.add_page()
                    pdf.chapter_title("Solar Energy Analytics")
                
                    # Add system details table
                    add_solar_details_table(pdf, solar_details)
                
                    # Add space
                    pdf.add_space(5)
                
                    # Add optimized PV capacity
                    add_optimized_pv_capacity(pdf, optimized_pv_capacity)
                
                    # Add key metrics
                    add_key_metrics(pdf, yearly_total_production, yearly_energy_consumption)
                    
                    # Plot for Hourly Solar Energy Production Distribution
                    pdf.add_page_if_needed()
                    pdf.sub_chapter_title("Hourly Solar Energy Yield Distribution Plot")
                    pdf.add_plot(solar_distribution_plot_path, width=120, height=90)
                
                    # Add tables and charts for solar analytics
                    add_solar_tables_and_charts(pdf, monthly_merged_df, excess_monthly_production_df, daily_production_plot_path)
                
                    # Generate and add combined summary
                    summary = generate_combined_summary(analytics, optimized_pv_capacity, yearly_total_production, yearly_energy_consumption)
                    pdf.add_summary(summary)
                
                    return pdf.output(dest='S').encode('latin1')
                
                # Generate PDF report
                pdf_content = generate_pdf(
                    analytics,
                    adjusted_max_hourly_demand,
                    adjusted_yearly_total_consumption,
                    plot_path,
                    logo_file,
                    solar_details={
                        'latitude': latitude,
                        'longitude': longitude,
                        'tz_str': tz_str,
                        'surface_tilt': surface_tilt,
                        'surface_azimuth': surface_azimuth,
                        'module_efficiency': module_efficiency,
                        'pr': pr,
                    },
                    optimized_pv_capacity=optimized_pv_capacity,
                    yearly_total_production=yearly_total_production,
                    yearly_energy_consumption=yearly_energy_consumption,
                    monthly_merged_df=monthly_merged_df,
                    excess_monthly_production_df=excess_monthly_production_df,
                    daily_production_plot_path=daily_production_plot_path,
                    daily_avg_24h=daily_avg_24h,
                    daily_avg_sun=daily_avg_sun,
                    hourly_distribution_plot_path=hourly_distribution_plot_path,
                    comp_distribution_plot_path=comp_distribution_plot_path,
                    solar_distribution_plot_path=solar_distribution_plot_path
                )
                
                st.write('_____')
                st.sidebar.write('_____')
                
                # Provide download option for the PDF report as a button
                b64_pdf = base64.b64encode(pdf_content).decode('latin1')
                href = f'<a href="data:application/octet-stream;base64,{b64_pdf}" download="energy_consumption_report.pdf"><button class="nav-button-download">Download PDF Report</button></a>'
                st.markdown(href, unsafe_allow_html=True)
                
                # Place the download button in the sidebar as well
                st.sidebar.markdown(href, unsafe_allow_html=True)

                #Function to convert DataFrame to Excel and encode as base64
                def convert_df_to_excel(df):
                    output = BytesIO()
                    writer = pd.ExcelWriter(output, engine='xlsxwriter')
                    df.to_excel(writer, index=True, sheet_name='Sheet1')  # Set index=True
                    writer.save()
                    processed_data = output.getvalue()
                    return processed_data
                # Convert DataFrame to Excel and encode as base64
                #monthly_merged_df.set_index('Month', inplace=True)
                excel_data = convert_df_to_excel(daily_merged_df)
                b64_excel = base64.b64encode(excel_data).decode()
                
                # Download Excel file button for `monthly_merged_df`
                excel_href = f'<a href="data:application/octet-stream;base64,{b64_excel}" download="daily_merged_df.xlsx"><button class="nav-button-download">Download daily Merged Data</button></a>'
                st.sidebar.markdown(excel_href, unsafe_allow_html=True)
                
                
                
                
                # Download Button for Excess Daily Solar Production
                excess_daily_production_df = pd.DataFrame(list(excess_daily_production.items()), columns=['Date', 'Excess Production (kWh)'])

                # Ensure the Date column is in datetime format
                excess_daily_production_df['Date'] = pd.to_datetime(excess_daily_production_df['Date'])

                # Set the Date column as the index
                excess_daily_production_df.set_index('Date', inplace=True)
                
                
                # Convert DataFrame to Excel and encode as base64
                excel_excess_data = convert_df_to_excel(excess_daily_production_df)
                b64_excess_excel = base64.b64encode(excel_excess_data).decode()
                
                # Download Excel file button for `daily_excess_df`
                excess_excel_href = f'<a href="data:application/octet-stream;base64,{b64_excess_excel}" download="daily_excess_df.xlsx"><button class="nav-button-download">Download daily Excess Data</button></a>'
                st.sidebar.markdown(excess_excel_href, unsafe_allow_html=True)




                #-------sidebar Download Area------#
                #with st.sidebar:
                    

                
                
                



            else:
                st.warning("Please enter valid latitude and longitude coordinates.")


#------------------PDF Report Generation Section----------#

            # Generate PDF report
            





            
            



            

        else:
            st.error("Failed to process the data due to datetime format issues.")
#-------------------------------#


