# 🌾 CropVision – Yield Analytics & Smart Farming Dashboard

A production-grade **machine learning system** for predicting crop yield in Indian states, powered by an interactive **Streamlit dashboard** with a premium analytics UI inspired by modern SaaS dashboards.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **🔐 Login System** | Session-based authentication with multiple user accounts. Protects dashboard access. |
| **📊 Analytics Dashboard** | Premium dark-themed dashboard with 8 KPI cards, area charts, donut charts, bar charts, scatter plots, and data tables. |
| **🌾 Yield Prediction** | ML-powered crop yield forecasting using trained Random Forest / Polynomial Regression models. |
| **🌧️ Rainfall Advisor** | Smart crop recommendations based on expected annual rainfall — 5 rainfall zones with tailored advice. |
| **📂 Crop Categories** | Browse crops by type (Vegetables, Spices/Dhanya, Cereals, Pulses, Oilseeds, Cash Crops, Fruits) with per-category charts & stats. |
| **⚠️ Precautions** | Farming best practices covering drought, pest management, fertilizer use, soil health, and seasonal planning. |
| **🎨 Premium UI** | Custom CSS with gradient cards, hover effects, Inter font, color-coded KPI accents, and responsive layout. |

---

## 📸 Dashboard Pages

### Login Page
- Branded login screen with `🌾 CropVision` branding
- Demo account popover for quick access
- Secure password hashing (SHA-256)

### Dashboard
- **8 KPI Cards**: Average Yield, Best Crop, Top State, Avg Rainfall, Total Records, Crop Types, States, Avg Fertilizer
- **Yield Trend**: Area chart showing yield over years
- **Season Donut**: Pie chart of yield contribution by season
- **Top Crops Bar**: Horizontal bar chart of top 10 crops
- **Rainfall vs Yield**: Scatter plot correlation analysis
- **State Performance**: Top 10 states by yield
- **Data Table**: Top 15 Crop–State–Season combinations

### Predict Yield
- 3-column input form (Crop Details, Location & Weather, Inputs Applied)
- ML-powered prediction with yield rating (High / Medium / Low)
- Automatic rainfall-zone crop suggestions

### Rainfall Advisor
- Interactive slider for expected rainfall (100–4000 mm)
- 5 zones: Arid, Low, Moderate, High, Very High
- Historical crop performance at similar rainfall levels
- Complete zone reference guide

### Crop Categories
- Tabbed interface: Vegetables, Spices & Dhanya, Cereals, Pulses, Oilseeds, Cash Crops, Fruits
- Per-category bar charts and data tables

### Precautions
- 6 precaution categories with actionable tips
- Key takeaways section with visual icons

---

## 🗂️ Project Structure

```
crop_yield_prediction/
├── app.py                                    # Streamlit web application (main UI)
├── modeltrain.py                             # Model training with detailed logging
├── setup.py                                  # One-command setup to train all models
├── verifyresult.py                           # Model verification script
├── crop_yield_pred.ipynb                     # Jupyter notebook for exploration
├── README.md                                 # This file
├── LEARN.md                                  # Learning guide & methodology
├── crop-yield-in-indian-states-dataset/
│   └── crop_yield.csv                        # Dataset (10 columns, ~10k rows)
├── models/                                   # Saved model artifacts directory
├── best_reg_model.pkl                        # Best regression model
├── best_classification_model.pkl             # Best classification model
├── scaler_reg.pkl                            # StandardScaler for regression
├── le_item.pkl                               # LabelEncoder for crops
├── le_state.pkl                              # LabelEncoder for states
├── le_season.pkl                             # LabelEncoder for seasons
└── crop/                                     # Python virtual environment
```

---

## 📊 Dataset

The dataset contains historical crop production data for Indian states with **10 columns**:

| Column | Description |
|--------|-------------|
| `Crop` | Type of crop (55 unique crops) |
| `Crop_Year` | Year of production |
| `Season` | Growing season (Kharif, Rabi, Summer, Winter, Whole Year, Autumn) |
| `State` | Indian state |
| `Area` | Land area under cultivation (hectares) |
| `Production` | Total output (tonnes) |
| `Annual_Rainfall` | Yearly rainfall (mm) |
| `Fertilizer` | Fertilizer usage (tonnes) |
| `Pesticide` | Pesticide usage (litres) |
| `Yield` | Crop yield per hectare (hg/ha) |

### Crop Categories in Dataset

| Category | Crops |
|----------|-------|
| **🥬 Vegetables** | Onion, Potato, Sweet Potato, Tapioca, Garlic, Ginger, Dry Chillies |
| **🌿 Spices & Dhanya** | Coriander, Black Pepper, Cardamom, Turmeric, Ginger |
| **🌾 Cereals** | Rice, Wheat, Maize, Bajra, Jowar, Ragi, Barley, Small Millets |
| **🫘 Pulses** | Arhar/Tur, Gram, Masoor, Moong, Urad, Cowpea, Horse-Gram, Khesari, Moth |
| **🥜 Oilseeds** | Groundnut, Sesamum, Linseed, Castor Seed, Soyabean, Sunflower, Safflower |
| **🏭 Cash Crops** | Sugarcane, Cotton, Jute, Mesta, Tobacco |
| **🍌 Fruits** | Banana, Coconut, Arecanut, Cashewnut |

---

## 🛠️ Technologies Used

| Technology | Purpose |
|-----------|---------|
| **Python 3.x** | Core programming language |
| **Streamlit** | Web application framework |
| **Scikit-learn** | Machine learning (Random Forest, Decision Tree, SVM, etc.) |
| **Pandas** | Data manipulation and analysis |
| **Plotly** | Interactive data visualization |
| **Joblib** | Model serialization |
| **Hashlib** | Password hashing for authentication |

---

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/ixabrar/Crop-Yield-Prediction-With-Dashboard.git
cd crop_yield_prediction
```

### 2. Create & Activate Virtual Environment

```bash
python -m venv crop

# Windows
crop\Scripts\activate

# macOS / Linux
source crop/bin/activate
```

### 3. Install Dependencies

```bash
pip install pandas scikit-learn joblib streamlit plotly
```

### 4. Train Models (One-Time Setup)

```bash
python setup.py
```

This will train all regression and classification models, saving `.pkl` artifacts to the project root. Takes 2–5 minutes.

### 5. Run the Application

```bash
streamlit run app.py
```

Opens at **http://localhost:8501**

---

## 🔐 Login Credentials

| Username | Password | Role |
|----------|----------|------|
| `admin` | `admin123` | Administrator |
| `farmer` | `crop2026` | Farmer |
| `demo` | `demo` | Demo User |

---

## 🌧️ Rainfall-Based Crop Recommendations

The system provides intelligent sowing suggestions across 5 rainfall zones:

| Zone | Rainfall | Recommended Crops |
|------|----------|-------------------|
| 🏜️ Arid | < 500 mm | Bajra, Moth, Guar Seed, Castor Seed, Barley, Gram |
| 🌤️ Low | 500–750 mm | Jowar, Groundnut, Sesamum, Safflower, Sunflower |
| 🌦️ Moderate | 750–1150 mm | Wheat, Maize, Soyabean, Onion, Potato, Coriander |
| 🌧️ High | 1150–2000 mm | Rice, Sugarcane, Jute, Banana, Ginger, Black Pepper |
| ⛈️ Very High | > 2000 mm | Rice, Arecanut, Coconut, Cashewnut, Cardamom |

---

## ⚠️ Agricultural Precautions

The app provides best practices across 6 key areas:

1. **🌊 Excessive Rainfall** — Drainage, fungicide prevention, harvest timing
2. **☀️ Drought Conditions** — Drip irrigation, mulching, drought-resistant varieties
3. **🐛 Pest Management** — IPM, crop rotation, natural predators
4. **🧪 Fertilizer Usage** — Soil testing, organic alternatives, split application
5. **🌱 Soil Health** — Crop rotation, green manuring, pH management
6. **🌡️ Seasonal Planning** — Kharif/Rabi/Zaid sowing calendars

---

## 🧪 Model Details

### Regression Models (Yield Prediction)

| Model | Purpose |
|-------|---------|
| Linear Regression | Fast baseline |
| Polynomial Regression | Non-linear feature interactions |
| Decision Tree | Interpretable patterns |
| **Random Forest** | Best generalization (typically selected) |

### Classification Models (Yield Category)

| Model | Purpose |
|-------|---------|
| Logistic Regression | Linear classification baseline |
| KNN | Similarity-based predictions |
| SVM | Complex decision boundaries |
| Decision Tree | Rule-based classification |
| Random Forest | Robust ensemble approach |
| Naive Bayes | Probabilistic classification |

### Features Used

```
Crop, Crop_Year, Season, State, Annual_Rainfall, Fertilizer, Pesticide → Yield
```

---

## 📁 Project Files

| File | Description |
|------|-------------|
| `app.py` | Main Streamlit application with 5 pages and premium UI |
| `setup.py` | One-command model training and artifact generation |
| `modeltrain.py` | Detailed training script with verbose logging |
| `verifyresult.py` | Model verification and sanity checks |
| `crop_yield_pred.ipynb` | Jupyter notebook for exploratory data analysis |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit changes (`git commit -m 'Add your feature'`)
4. Push to branch (`git push origin feature/your-feature`)
5. Submit a Pull Request

---

## 📜 License

This project is open source under the **MIT License**.

---

## 📬 Contact

For questions, issues, or feature requests, please open an issue in the repository.
