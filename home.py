import streamlit as st

# CSS styling for the modern landing page
page_style = """
    <style>
        body {
            font-family: 'Arial', sans-serif;
        }
        .hero {
            background-image: url('https://i.imgur.com/8g1pRFY.png');
            background-size: cover;
            background-position: center;
            padding: 20px 20px;  /* Reduced padding */
            text-align: center;
            color: white;
        }
        .hero h1 {
            font-size: 48px;
            font-weight: bold;
            color: #333333; /* Dark grey color */
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.4); /* Dark shadow effect */
            background-color: rgba(255, 255, 255, 0.7); /* Semi-transparent white background */
            display: inline-block;
            padding: 10px 20px;
            border-radius: 5px;
            margin-bottom: 20px; /* Added space between header and sub-header */
        }
        .hero p {
            font-size: 24px;
            color: #333333; /* Dark grey color */
            background-color: rgba(255, 255, 255, 0.7); /* Semi-transparent white background */
            display: inline-block;
            padding: 10px 20px;
            border-radius: 5px;
        }
        .features {
            max-width: 800px;
            margin: 40px auto;
            padding: 20px;
            background: #fff;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            text-align: center;
        }
        .features h2 {
            color: #2e8b57;
            font-size: 28px;
            margin-bottom: 20px;
        }
        .features-grid {
            display: flex;
            justify-content: space-between;
            flex-wrap: wrap;
        }
        .feature-item {
            flex: 1 1 45%;
            margin: 10px;
            padding: 20px;
            background: #f9f9f9;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s ease;
        }
        .feature-item:hover {
            transform: scale(1.05);
        }
        .feature-item h4 {
            color: #ff6347;
            font-size: 20px;
            margin-bottom: 10px;
        }
        .feature-item p {
            font-size: 16px;
            color: #555;
        }
        .feature-item i {
            font-size: 48px;
            color: #ff6347;
            margin-bottom: 10px;
        }
        .contact {
            max-width: 800px;
            margin: 40px auto;
            padding: 20px;
            background: #fff;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            text-align: center;
        }
        .contact h2 {
            color: #2e8b57;
            font-size: 28px;
            margin-bottom: 20px;
        }
        .contact p {
            font-size: 18px;
            color: #555;
        }
        .contact a {
            color: #1e90ff;
            text-decoration: none;
        }
        .contact a:hover {
            text-decoration: underline;
        }
        .contact-icons {
            display: flex;
            justify-content: center;
            margin-top: 20px;
        }
        .contact-icons a {
            margin: 0 10px;
            font-size: 24px;
            color: #2e8b57;
            transition: color 0.3s ease, transform 0.3s ease;
        }
        .contact-icons a:hover {
            color: #ff6347;
            transform: scale(1.2);
        }
    </style>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">
"""

# Hero section with background image
hero_section = """
<div class="hero">
    <h1>Energy Consumption and Solar Analytics</h1>
    <p>Analyze your energy consumption and optimize solar energy generation with ease.</p>
</div>
"""

# Features section
features_section = """
<div class="features">
    <h2>Key Features</h2>
    <div class="features-grid">
        <div class="feature-item">
            <i class="fas fa-upload"></i>
            <h4>Easy Data Upload</h4>
            <p>Easily upload your timeseries dataset for analysis.</p>
        </div>
        <div class="feature-item">
            <i class="fas fa-broom"></i>
            <h4>Data Cleaning</h4>
            <p>Automatically clean and preprocess your data for accurate results.</p>
        </div>
        <div class="feature-item">
            <i class="fas fa-chart-line"></i>
            <h4>Detailed Analysis</h4>
            <p>Get comprehensive insights into your energy consumption patterns.</p>
        </div>
        <div class="feature-item">
            <i class="fas fa-solar-panel"></i>
            <h4>Solar Optimization</h4>
            <p>Optimize your solar energy generation to meet your energy needs.</p>
        </div>
        <div class="feature-item">
            <i class="fas fa-file-alt"></i>
            <h4>Report Generation</h4>
            <p>Generate detailed reports for further analysis and decision making.</p>
        </div>
        <div class="feature-item">
            <i class="fas fa-chart-bar"></i>
            <h4>Interactive EDA Dashboard</h4>
            <p>Explore interactive charts and visualizations for deeper insights.</p>
        </div>
    </div>
</div>
"""

# Contact section
contact_section = """
<div class="contact">
    <h2>Contact</h2>
    <p>If you have any questions, feel free to reach out to us:</p>
    <div class="contact-icons">
        <a href="mailto:amrit.mandal0191@gmail.com"><i class="fas fa-envelope"></i></a>
        <a href="https://www.linkedin.com/in/amritmandal/" target="_blank"><i class="fab fa-linkedin"></i></a>
    </div>
</div>
"""

# Streamlit app layout
st.markdown(page_style, unsafe_allow_html=True)
st.markdown(hero_section, unsafe_allow_html=True)
st.markdown(features_section, unsafe_allow_html=True)
st.markdown(contact_section, unsafe_allow_html=True)
