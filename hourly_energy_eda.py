# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 13:24:37 2024

@author: amrit
"""

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
st.title('Energy Data Updater and Analytics')

# Upload the user dataset
uploaded_file = st.file_uploader("Choose a dataset Excel file", type=["xlsx"])

if uploaded_file:
    input_df = pd.read_excel(uploaded_file)
    st.write("Input DataFrame:")
    st.write(input_df)

    if input_df.shape[1] < 2:
        st.error("The uploaded file must have at least two columns: datetime and energy consumption.")
    else:
        datetime_col = input_df.columns[0]
        energy_col = input_df.columns[1]

        # Fill missing rows and update 'energy_comp_kWh'
        filled_df, missing_rows = fill_missing_rows(input_df, datetime_col, energy_col)

        if filled_df is not None:
            st.write("Updated DataFrame:")
            st.write(filled_df)

            if missing_rows:
                st.write("Missing rows added:")
                for row in missing_rows:
                    st.write(row)

            # Provide download option for the updated dataset
            output = BytesIO()
            filled_df.to_excel(output, index=False)
            output.seek(0)

            st.download_button(
                label="Download Updated Dataset",
                data=output,
                file_name='updated_energy_data.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )

            # Calculate and display analytics
            analytics = calculate_analytics(filled_df, datetime_col, 'energy_comp_kWh')

            st.write("### Max, Min, and Average Hourly Energy Demand")
            st.metric(label="Max Hourly Demand", value=f"{analytics['max_hourly_demand']:.2f} kWh")
            st.metric(label="Min Hourly Demand", value=f"{analytics['min_hourly_demand']:.2f} kWh")
            st.metric(label="Average Hourly Demand", value=f"{analytics['average_hourly_demand']:.2f} kWh")

            st.write("### Month-wise Average, Max, and Min Daily Energy Consumption")
            st.write(analytics['monthly_stats'])

            st.write("### Monthly Energy Consumption for All 12 Months")
            st.write(analytics['monthly_consumption'])

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

            # Function to generate PDF report
            class PDFReport(FPDF):
                def header(self):
                    self.set_font('Arial', 'B', 12)
                    self.cell(0, 10, 'Energy Consumption Report', 0, 1, 'C')
            
                def chapter_title(self, title):
                    self.set_font('Arial', 'B', 12)
                    self.cell(0, 10, title, 0, 1, 'L')
                    self.ln(2)
            
                def chapter_body(self, body):
                    self.set_font('Arial', '', 12)
                    self.multi_cell(0, 10, body)
                    self.ln()
            
                def add_plot(self, image):
                    self.image(image, x=10, y=None, w=190)
            
            def generate_pdf(analytics, adjusted_max_hourly_demand, adjusted_yearly_total_consumption, encoded_image):
                pdf = PDFReport()
                pdf.add_page()
            
                # Adding max, min, and average hourly demand
                pdf.chapter_title("Max, Min, and Average Hourly Energy Demand")
                pdf.chapter_body(f"Max Hourly Demand: {analytics['max_hourly_demand']:.2f} kWh\n"
                                 f"Min Hourly Demand: {analytics['min_hourly_demand']:.2f} kWh\n"
                                 f"Average Hourly Demand: {analytics['average_hourly_demand']:.2f} kWh")
            
                

                # Adding adjusted metrics
                pdf.chapter_title("Adjusted Energy Metrics")
                pdf.chapter_body(f"Adjusted Max Hourly Demand: {adjusted_max_hourly_demand:.2f} kWh\n"
                                 f"Adjusted Yearly Total Consumption: {adjusted_yearly_total_consumption:.2f} kWh")
                
                # Adding monthly stats
                pdf.chapter_title("Month-wise Average, Max, and Min Daily Energy Consumption")
                monthly_stats_body = analytics['monthly_stats'].to_string(index=False)
                pdf.chapter_body(monthly_stats_body)
                
                # Adding monthly energy consumption
                pdf.chapter_title("Monthly Energy Consumption for All 12 Months")
                monthly_consumption_body = analytics['monthly_consumption'].to_string(index=False)
                pdf.chapter_body(monthly_consumption_body)
                
                # Adding plot
                pdf.chapter_title("Monthly Energy Consumption Plot")
                pdf.add_plot(plot_path)
                
                return pdf.output(dest='S').encode('latin1')

            # Generate PDF report
            # Generate PDF report
            pdf_content = generate_pdf(analytics, adjusted_max_hourly_demand, adjusted_yearly_total_consumption, plot_path)
            
            # Provide download option for the PDF report
            b64_pdf = base64.b64encode(pdf_content).decode('latin1')
            href = f'<a href="data:application/octet-stream;base64,{b64_pdf}" download="energy_consumption_report.pdf">Download PDF Report</a>'
            st.markdown(href, unsafe_allow_html=True)

        else:
            st.error("Failed to process the data due to datetime format issues.")
