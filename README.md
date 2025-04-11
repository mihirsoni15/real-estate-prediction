# 🏡 Real Estate Price Prediction & Visualization App

This Streamlit application allows users to explore a real estate dataset, perform predictions on house prices based on user-selected property features, and visualize various insights from the dataset.

---

## 📌 Overview

This project leverages machine learning (Linear Regression) to predict house prices. It includes:
- Exploratory Data Analysis (EDA)
- Data preprocessing and feature engineering
- Interactive visualizations
- User-friendly dropdown-based inputs for predictions

---

## 🛠 Project Structure

```
Real_Estate_Solution/
├── src/
│   ├── utils.py
│   ├── data_processing.py
│   ├── eda.py
│   ├── preprocessing.py
│   ├── data_split.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── visualization.py
├── main.py
├── app.py
├── final.csv
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### 🔧 Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Real_Estate_Solution.git
cd Real_Estate_Solution
```

### 🌐 Step 2: Set Up a Virtual Environment

```bash
python -m venv venv
# Activate the environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 📦 Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### ▶️ Step 4: Run the Streamlit App

```bash
streamlit run app.py
```

After running this command, your app will be accessible at [http://localhost:8501](http://localhost:8501).

---

## 🎯 Features

- **Interactive Price Predictions:** Choose property features from dropdowns to predict property prices.
- **Model Evaluation:** See the Mean Absolute Error (MAE) and Mean Squared Error (MSE) for model accuracy.
- **Visual Insights:**
  - Distribution of property prices
  - Scatter plot of square footage vs. price
  - Correlation heatmap to understand feature relationships

---

## 🗃 Data

- **Dataset:** `final.csv` includes property features such as:
  - Price
  - Year sold
  - Property tax
  - Insurance
  - Beds & baths
  - Square footage
  - Property type
  - Lot size

---

## 📈 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn (Linear Regression)
- Matplotlib
- Streamlit

---

## 🌩 Deployment

Deploy easily on [Streamlit Cloud](https://streamlit.io/cloud) by linking your GitHub repository.

---

## 📜 License

Licensed under the [MIT License](LICENSE).

---

## 🙌 Contribution

Feel free to fork this repository and make improvements or adjustments. Pull requests and contributions are always welcome!

---

## ✉️ Contact

- **Author:** Sree Sankaran Chackoth
- **GitHub:** [github.com/just-sree](https://github.com/just-sree)