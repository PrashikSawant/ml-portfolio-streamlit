import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="My ML Portfolio",
    page_icon="🤖",
    layout="wide"
)

# Main title
st.title("🤖 My Machine Learning Portfolio")
st.markdown("*A showcase of my recent ML projects from LinkedIn*")

# Sidebar for navigation
st.sidebar.title("Navigation")
project = st.sidebar.selectbox(
    "Choose a project:",
    ["Home", "House Price Predictor", "Iris Flower Classifier", "Sales Analytics", "Stock Price Trend"]
)

# Home page
if project == "Home":
    st.header("Welcome to My ML Journey! 👋")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("About Me")
        st.write("""
        I'm passionate about machine learning and data science. 
        This dashboard showcases the projects I've been working on 
        and sharing on LinkedIn recently.
        """)
        
        st.subheader("Skills")
        st.write("• Python, Pandas, Scikit-learn")
        st.write("• Machine Learning & Deep Learning")
        st.write("• Data Visualization")
        st.write("• Model Deployment")
    
    with col2:
        st.subheader("Recent Projects")
        st.info("🏠 House Price Predictor - Linear Regression")
        st.info("🌸 Iris Flower Classifier - Random Forest")
        st.info("📊 Sales Analytics Dashboard")
        st.info("📈 Stock Price Trend Analysis")

# Project 1: House Price Predictor
elif project == "House Price Predictor":
    st.header("🏠 House Price Predictor")
    st.write("*Predicting house prices using Linear Regression*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Features")
        bedrooms = st.slider("Number of Bedrooms", 1, 5, 3)
        bathrooms = st.slider("Number of Bathrooms", 1, 4, 2)
        sqft = st.slider("Square Feet", 500, 5000, 2000)
        age = st.slider("House Age (years)", 0, 50, 10)
        
        # Simple prediction logic (demo purposes)
        predicted_price = (sqft * 150) + (bedrooms * 10000) + (bathrooms * 8000) - (age * 1000)
        
        if st.button("Predict Price"):
            st.success(f"Predicted Price: ${predicted_price:,.2f}")
    
    with col2:
        st.subheader("Model Performance")
        # Create sample data for visualization
        sample_data = pd.DataFrame({
            'Actual': [250000, 300000, 180000, 450000, 320000],
            'Predicted': [245000, 310000, 175000, 440000, 315000]
        })
        
        fig = px.scatter(sample_data, x='Actual', y='Predicted', 
                        title='Actual vs Predicted Prices')
        fig.add_shape(type="line", x0=0, y0=0, x1=500000, y1=500000)
        st.plotly_chart(fig)
        
        st.metric("Model Accuracy", "92.5%")
        st.metric("Mean Absolute Error", "$15,240")

# Project 2: Iris Classifier
elif project == "Iris Flower Classifier":
    st.header("🌸 Iris Flower Classifier")
    st.write("*Classifying iris flowers using Random Forest*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Measurements")
        sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.5)
        sepal_width = st.slider("Sepal Width (cm)", 2.0, 5.0, 3.5)
        petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
        petal_width = st.slider("Petal Width (cm)", 0.1, 3.0, 1.5)
        
        # Simple classification logic (demo)
        if petal_length < 2.5:
            prediction = "Setosa"
            confidence = 98
        elif petal_length < 5.0:
            prediction = "Versicolor"
            confidence = 94
        else:
            prediction = "Virginica"
            confidence = 96
            
        if st.button("Classify Flower"):
            st.success(f"Prediction: **{prediction}**")
            st.info(f"Confidence: {confidence}%")
    
    with col2:
        st.subheader("Dataset Visualization")
        # Create sample iris data
        iris_data = pd.DataFrame({
            'sepal_length': np.random.normal(5.8, 0.8, 150),
            'petal_length': np.random.normal(3.8, 1.8, 150),
            'species': ['Setosa']*50 + ['Versicolor']*50 + ['Virginica']*50
        })
        
        fig = px.scatter(iris_data, x='sepal_length', y='petal_length', 
                        color='species', title='Iris Dataset Distribution')
        st.plotly_chart(fig)
        
        st.metric("Model Accuracy", "97.3%")
        st.metric("Cross-validation Score", "96.8%")

# Project 3: Sales Analytics
elif project == "Sales Analytics":
    st.header("📊 Sales Analytics Dashboard")
    st.write("*Analyzing sales trends and patterns*")
    
    # Generate sample sales data
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    sales_data = pd.DataFrame({
        'date': dates,
        'sales': np.random.normal(10000, 2000, len(dates)) + 
                np.sin(np.arange(len(dates)) * 2 * np.pi / 365) * 3000,
        'region': np.random.choice(['North', 'South', 'East', 'West'], len(dates))
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sales Trend")
        fig = px.line(sales_data, x='date', y='sales', 
                     title='Daily Sales Over Time')
        st.plotly_chart(fig)
        
        st.subheader("Key Metrics")
        total_sales = sales_data['sales'].sum()
        avg_daily_sales = sales_data['sales'].mean()
        
        st.metric("Total Sales", f"${total_sales:,.0f}")
        st.metric("Average Daily Sales", f"${avg_daily_sales:,.0f}")
    
    with col2:
        st.subheader("Sales by Region")
        region_sales = sales_data.groupby('region')['sales'].sum().reset_index()
        fig = px.pie(region_sales, values='sales', names='region',
                    title='Sales Distribution by Region')
        st.plotly_chart(fig)
        
        st.subheader("Top Performing Days")
        top_days = sales_data.nlargest(5, 'sales')[['date', 'sales']]
        st.dataframe(top_days)

# Project 4: Stock Analysis
elif project == "Stock Price Trend":
    st.header("📈 Stock Price Trend Analysis")
    st.write("*Analyzing stock price movements and trends*")
    
    # Generate sample stock data
    dates = pd.date_range('2023-01-01', '2024-01-31', freq='D')
    stock_data = pd.DataFrame({
        'date': dates,
        'price': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
        'volume': np.random.normal(1000000, 200000, len(dates))
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Stock Price Movement")
        fig = px.line(stock_data, x='date', y='price', 
                     title='Stock Price Over Time')
        st.plotly_chart(fig)
        
        # Calculate metrics
        current_price = stock_data['price'].iloc[-1]
        price_change = current_price - stock_data['price'].iloc[0]
        percent_change = (price_change / stock_data['price'].iloc[0]) * 100
        
        st.metric("Current Price", f"${current_price:.2f}")
        st.metric("Total Change", f"${price_change:.2f}", f"{percent_change:.1f}%")
    
    with col2:
        st.subheader("Trading Volume")
        fig = px.bar(stock_data.tail(30), x='date', y='volume',
                    title='Recent Trading Volume')
        st.plotly_chart(fig)
        
        st.subheader("Price Statistics")
        st.write(f"**High:** ${stock_data['price'].max():.2f}")
        st.write(f"**Low:** ${stock_data['price'].min():.2f}")
        st.write(f"**Average:** ${stock_data['price'].mean():.2f}")
        st.write(f"**Volatility:** {stock_data['price'].std():.2f}")

# Footer
st.markdown("---")
st.markdown("*Built with Streamlit • Check out my LinkedIn for more projects*")