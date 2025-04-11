import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import logging

from src.data_processing import load_data
from src.preprocessing import preprocess_data
from src.data_split import split_data
from src.model_training import train_model
from src.model_evaluation import evaluate_model

st.title("🏡 Real Estate Price Prediction & Visualizations")

# Load dataset from the repo (final.csv)
df = load_data("final.csv")
if df is None:
    st.error("Error loading dataset.")
else:
    st.write("### Data Preview")
    st.dataframe(df.head())

    # Preprocess the data and train the model
    df_processed = preprocess_data(df.copy())
    try:
        X_train, X_test, y_train, y_test = split_data(df_processed, "price")  # using lowercase "price"
    except Exception as e:
        st.error("Error splitting data: " + str(e))
        st.stop()
    
    try:
        model = train_model(X_train, y_train)
    except Exception as e:
        st.error("Error training model: " + str(e))
        st.stop()

    try:
        mae, mse = evaluate_model(model, X_test, y_test)
    except Exception as e:
        st.error("Error evaluating model: " + str(e))
        st.stop()

    st.write("### Model Evaluation")
    st.write(f"**MAE:** {mae}")
    st.write(f"**MSE:** {mse}")

    # Generate dropdown options from the dataset
    year_sold_options = sorted(df["year_sold"].unique())
    property_tax_options = sorted(df["property_tax"].unique())
    insurance_options = sorted(df["insurance"].unique())
    beds_options = sorted(df["beds"].unique())
    baths_options = sorted(df["baths"].unique())
    sqft_options = sorted(df["sqft"].unique())
    year_built_options = sorted(df["year_built"].unique())
    lot_size_options = sorted(df["lot_size"].unique())
    property_age_options = sorted(df["property_age"].unique())

    st.write("### Predict a Property Price")
    with st.form("predict_form"):
        st.subheader("Enter Property Details")
        # Dropdowns for each numeric parameter
        default_year_sold = 2007
        idx = year_sold_options.index(default_year_sold) if default_year_sold in year_sold_options else 0
        year_sold = st.selectbox("Year Sold", options=year_sold_options, index=idx)
        
        default_tax = 461
        idx = property_tax_options.index(default_tax) if default_tax in property_tax_options else 0
        property_tax = st.selectbox("Property Tax", options=property_tax_options, index=idx)
        
        default_insurance = 139
        idx = insurance_options.index(default_insurance) if default_insurance in insurance_options else 0
        insurance = st.selectbox("Insurance", options=insurance_options, index=idx)
        
        default_beds = 3
        idx = beds_options.index(default_beds) if default_beds in beds_options else 0
        beds = st.selectbox("Beds", options=beds_options, index=idx)
        
        default_baths = 3
        idx = baths_options.index(default_baths) if default_baths in baths_options else 0
        baths = st.selectbox("Baths", options=baths_options, index=idx)
        
        default_sqft = 1900
        idx = sqft_options.index(default_sqft) if default_sqft in sqft_options else 0
        sqft = st.selectbox("Sqft", options=sqft_options, index=idx)
        
        default_year_built = 1986
        idx = year_built_options.index(default_year_built) if default_year_built in year_built_options else 0
        year_built = st.selectbox("Year Built", options=year_built_options, index=idx)
        
        default_lot_size = 5846
        idx = lot_size_options.index(default_lot_size) if default_lot_size in lot_size_options else 0
        lot_size = st.selectbox("Lot Size", options=lot_size_options, index=idx)
        
        default_property_age = 21
        idx = property_age_options.index(default_property_age) if default_property_age in property_age_options else 0
        property_age = st.selectbox("Property Age", options=property_age_options, index=idx)
        
        # Dropdowns for binary/categorical variables
        basement = st.selectbox("Basement (0 = No, 1 = Yes)", options=[0, 1])
        popular = st.selectbox("Popular (0 = No, 1 = Yes)", options=[0, 1])
        recession = st.selectbox("Recession (0 = No, 1 = Yes)", options=[0, 1])
        property_type = st.selectbox("Property Type", options=["Bunglow", "Condo"])
        
        submitted = st.form_submit_button("Predict Price")
    
    if submitted:
        # Create dictionary for user input
        input_dict = {
            "year_sold": year_sold,
            "property_tax": property_tax,
            "insurance": insurance,
            "beds": beds,
            "baths": baths,
            "sqft": sqft,
            "year_built": year_built,
            "lot_size": lot_size,
            "basement": basement,
            "popular": popular,
            "recession": recession,
            "property_age": property_age,
        }
        # Create dummy variables for property type
        if property_type == "Bunglow":
            input_dict["property_type_Bunglow"] = 1
            input_dict["property_type_Condo"] = 0
        else:
            input_dict["property_type_Bunglow"] = 0
            input_dict["property_type_Condo"] = 1

        # Convert dictionary to DataFrame
        input_df = pd.DataFrame([input_dict])
        input_df = input_df.reindex(columns=X_train.columns, fill_value=0)

        # Generate prediction
        try:
            prediction = model.predict(input_df)[0]
            st.write("### Predicted Price")
            st.write(f"${prediction:,.2f}")
        except Exception as e:
            st.error("Error generating prediction: " + str(e))
        
        st.write("### Visualizations with Prediction Marker")
        
        # Visualization 1: Price Distribution with predicted price marker
        st.write("#### Distribution of Price")
        plt.figure()
        plt.hist(df["price"], bins=30, edgecolor='black')
        plt.axvline(x=prediction, color='red', linestyle='--', label="Predicted Price")
        plt.xlabel("Price")
        plt.ylabel("Frequency")
        plt.title("Distribution of Price")
        plt.legend()
        st.pyplot(plt.gcf())
        
        # Visualization 2: Scatter Plot (sqft vs. price) with input point highlighted
        st.write("#### Scatter Plot: Sqft vs Price")
        plt.figure()
        plt.scatter(df["sqft"], df["price"], alpha=0.5, label="Data Points")
        plt.scatter(sqft, prediction, color='red', s=100, label="Your Input")
        plt.xlabel("Sqft")
        plt.ylabel("Price")
        plt.title("Sqft vs Price")
        plt.legend()
        st.pyplot(plt.gcf())
        
        # Visualization 3: Correlation Heatmap
        st.write("#### Correlation Heatmap")
        plt.figure(figsize=(10, 8))
        corr = df.corr()
        im = plt.imshow(corr, cmap='viridis', interpolation='none')
        plt.colorbar(im)
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.columns)), corr.columns)
        plt.title("Correlation Heatmap")
        st.pyplot(plt.gcf())
