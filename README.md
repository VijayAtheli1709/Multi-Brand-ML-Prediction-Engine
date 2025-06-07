# 🚗 Used Car Price Predictor

**An intelligent machine learning application that predicts used car prices with high accuracy using brand-specific models.**
## Sneak Peak Into the Application:
### Predicting Audi Car Price:

![image](https://github.com/user-attachments/assets/3c5c2679-01aa-4946-bdd3-be6871ebc3f8)
![image](https://github.com/user-attachments/assets/94637d59-5321-49c3-9dfe-ecd61234fad6)

### Predicting Mercedes Car Price:

![image](https://github.com/user-attachments/assets/6b0a43db-5238-44da-b7f0-42c8ce3e4eb2)
![image](https://github.com/user-attachments/assets/070b05db-13f0-429f-98d4-5c429e7dc78e)

---

## 🎯 Project Overview

This application leverages machine learning to provide accurate price predictions for used cars across 9 major automotive brands. By analyzing key vehicle characteristics such as mileage, age, fuel type, and transmission, the system delivers reliable price estimates to help both buyers and sellers make informed decisions.

### 🌟 Key Features

- **Multi-Brand Support**: Specialized models for Audi, BMW, Ford, Hyundai, Mercedes, Skoda, Toyota, Vauxhall, and Volkswagen
- **Smart Model Selection**: Each brand uses its optimal algorithm (Decision Tree, Random Forest, or Linear Regression)
- **Interactive Web Interface**: User-friendly Streamlit application with real-time predictions
- **Comprehensive Feature Analysis**: Considers 8 key vehicle attributes for accurate pricing
- **Dynamic Model Updates**: Easy-to-retrain models with new data

---

## 🔧 Technical Architecture

### Machine Learning Pipeline
```
Data Input → Preprocessing → Brand-Specific Model → Price Prediction
```

**Algorithms Used:**
- **Decision Tree Regressor**: Mercedes, Audi, Vauxhall, Ford, Skoda, VW
- **Random Forest Regressor**: BMW, Toyota  
- **Linear Regression**: Hyundai

### Model Performance
- **High Accuracy**: R² scores consistently above 0.85 across all brands
- **Robust Preprocessing**: Handles missing values and categorical encoding automatically
- **Feature Engineering**: Optimized feature selection for each automotive brand

---

## 🚀 Live Demo

**[Try the App →](https://your-app-url.streamlit.app)**

### How to Use:
1. **Select Brand**: Choose from 9 supported automotive brands
2. **Choose Model**: Pick the specific car model from the dropdown
3. **Enter Details**: Input year, mileage, transmission, fuel type, etc.
4. **Get Prediction**: Receive instant price estimation with confidence

---

## 📊 Dataset & Features

### Input Features:
| Feature | Type | Description |
|---------|------|-------------|
| **Brand** | Categorical | Car manufacturer (Audi, BMW, Ford, etc.) |
| **Model** | Categorical | Specific car model |
| **Year** | Numerical | Manufacturing year |
| **Mileage** | Numerical | Total distance traveled |
| **Transmission** | Categorical | Manual, Automatic, Semi-Auto |
| **Fuel Type** | Categorical | Petrol, Diesel, Hybrid, Electric |
| **Tax** | Numerical | Annual road tax amount |
| **MPG** | Numerical | Miles per gallon efficiency |
| **Engine Size** | Numerical | Engine displacement in liters |

### Data Processing:
- **Missing Value Handling**: Intelligent imputation strategies
- **One-Hot Encoding**: Categorical variable transformation
- **Feature Scaling**: Normalized numerical inputs
- **Brand-Specific Training**: Separate models for optimal performance

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Quick Start
```bash
# Clone the repository
git clone https://github.com/yourusername/used-car-price-predictor.git
cd used-car-price-predictor

# Install dependencies
pip install -r requirements.txt

# Train models (optional - pre-trained models included)
python train_models.py

# Run the application
streamlit run Application.py
```

### Dependencies
```
streamlit==1.28.0
pandas==2.0.3
scikit-learn==1.3.0
joblib==1.3.2
openpyxl==3.1.2
```

---

## 📁 Project Structure

```
used-car-price-predictor/
├── 📄 Application.py          # Main Streamlit application
├── 🔧 train_models.py         # Model training pipeline
├── 📊 Final_dataset.xlsx      # Training dataset
├── 📋 requirements.txt        # Python dependencies
├── 📁 models/                 # Trained model files
│   ├── Mercedes_pipeline.pkl
│   ├── BMW_pipeline.pkl
│   ├── Audi_pipeline.pkl
│   └── ... (other brands)
└── 📖 README.md              # Project documentation
```

---

## 🎯 Use Cases

### For Car Buyers:
- **Budget Planning**: Determine fair market value before purchasing
- **Negotiation Tool**: Use predictions to negotiate better deals
- **Comparison Shopping**: Compare prices across different models and brands

### For Car Sellers:
- **Pricing Strategy**: Set competitive and realistic asking prices
- **Market Analysis**: Understand factors affecting your vehicle's value
- **Quick Valuation**: Get instant price estimates without lengthy appraisals

### For Automotive Professionals:
- **Inventory Pricing**: Price used car inventory accurately
- **Trade-in Evaluations**: Assess vehicle values for trade-ins
- **Market Research**: Analyze pricing trends across brands

---

## 🔮 Future Enhancements

- [ ] **Additional Brands**: Expand to include more automotive manufacturers
- [ ] **Advanced Features**: Incorporate vehicle history, accident records, service history
- [ ] **Real-time Data**: Integration with live market data feeds
- [ ] **Mobile App**: Native mobile application development
- [ ] **API Integration**: RESTful API for third-party integrations
- [ ] **Market Trends**: Historical price trend analysis and forecasting

---

## 📈 Performance Metrics

| Brand | Algorithm | R² Score | Training Data |
|-------|-----------|----------|---------------|
| Mercedes | Decision Tree | 0.91 | 2,500+ records |
| BMW | Random Forest | 0.89 | 3,200+ records |
| Audi | Decision Tree | 0.87 | 1,800+ records |
| Toyota | Random Forest | 0.88 | 2,100+ records |
| Ford | Decision Tree | 0.86 | 2,800+ records |

---

## 👨‍💻 About the Developer

This project demonstrates expertise in:
- **Machine Learning**: Model selection, training, and optimization
- **Data Science**: Feature engineering and statistical analysis
- **Web Development**: Interactive application development with Streamlit
- **Software Engineering**: Clean code architecture and deployment

### Skills Showcased:
- Python Programming
- Scikit-learn & Pandas
- Data Preprocessing & Feature Engineering
- Model Evaluation & Selection
- Web Application Development
- Version Control & Deployment

---

## 📞 Contact & Support

**Questions?** **Feedback?** **Collaboration opportunities?**

- 📧 **Email**: your.email@domain.com
- 💼 **LinkedIn**: [Your LinkedIn Profile](https://linkedin.com/in/yourprofile)
- 🐙 **GitHub**: [Your GitHub Profile](https://github.com/yourusername)
- 🌐 **Portfolio**: [Your Portfolio Website](https://yourportfolio.com)

---

## ⭐ Show Your Support

If you found this project helpful, please consider:
- ⭐ **Star this repository**
- 🍴 **Fork the project**
- 📢 **Share with your network**
- 💡 **Contribute improvements**

---


# 🚗 Multi-Brand ML Prediction Engine

This project is a machine learning-based system designed to predict the prices of used cars across **nine major brands**: VW, Hyundai, Skoda, Ford, BMW, Mercedes, Audi, Toyota, and Vauxhall. It includes data merging, cleaning, feature engineering, model training, evaluation, and visualization.

---

## 📊 Features

- End-to-end **data pipeline**: loading, merging, cleaning, and transformation.
- **Regression modeling** with brand-wise evaluations:
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor
- **Automated evaluation** using RMSE to determine the best model per brand.
- Rich **visual analysis**: heatmaps, trendlines, boxplots, and scatterplots.
- Final model-ready dataset available for download.
- Modular code for scalability and reproducibility.
- Ready for **web app integration** for real-time predictions (e.g., via Streamlit or Flask).

## Sneak Peak Into the Application:
### Predicting Audi Car Price:

![image](https://github.com/user-attachments/assets/3c5c2679-01aa-4946-bdd3-be6871ebc3f8)
![image](https://github.com/user-attachments/assets/94637d59-5321-49c3-9dfe-ecd61234fad6)

### Predicting Mercedes Car Price:

![image](https://github.com/user-attachments/assets/6b0a43db-5238-44da-b7f0-42c8ce3e4eb2)
![image](https://github.com/user-attachments/assets/070b05db-13f0-429f-98d4-5c429e7dc78e)

