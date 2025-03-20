## 👨‍💻 Author

**Tomás Arêde**  
📍 Instituto Superior Técnico Student | Energy Services Subject

📧 **Email**: [tomas.arede@gmail.com](mailto:tomas.arede@gmail.com)  
💼 **LinkedIn**: [linkedin.com/in/tomasarede](https://linkedin.com/in/tomasarede)  
📂 **GitHub**: [github.com/TomasArede](https://github.com/tomasarede)  


# IST Energy Forecast ⚡🌍

## 🌍 Scientific Overview
This project systematically prepares and analyzes the energy consumption data of the **South Tower** of **Instituto Superior Técnico (IST)** for the years **2017 and 2018**. The methodology follows a structured **data-driven approach**, including:

- **Data Preprocessing**: Outlier detection and removal, missing data imputation, and dataset structuring.
- **Feature Engineering**: Creation of new features and selection from existing ones using various methods:
  - Correlation Analysis
  - F-Test (K-Best)
  - Mutual Information
  - Recursive Feature Elimination (RFE)
  - Ensemble Methods (Random Forest)
- **Model Testing & Selection**: Evaluation of various predictive models:
  - **Autoregressive Models (AR, ARIMA)**
  - **Decision Trees & Random Forest**
  - **Gradient Boosting (XGBoost)**
  - **Neural Networks (Multi-Layer Perceptron - MLP)**

After comparative analysis, **XGBoost and Neural Networks** were selected as the final models. Their architectures are detailed below:

### 🔬 **XGBoost Model Architecture**
- **Tree-based Gradient Boosting**
- **Hyperparameter tuning for optimal performance**
- **Handles feature interactions effectively**
- **Provides feature importance analysis**

### 🧠 **Neural Network Model Architecture**
A **fully connected deep learning model** was employed:
- **fc1:** Linear(13, 128)
- **bn1:** BatchNorm1d(128)
- **fc2:** Linear(128, 256)
- **bn2:** BatchNorm1d(256)
- **fc3:** Linear(256, 128)
- **bn3:** BatchNorm1d(128)
- **fc4:** Linear(128, 64)
- **fc5:** Linear(64, 1)
- **dropout:** Dropout(p=0.2)

Optimization strategies:
- **Adam optimizer with adaptive learning rate**
- **Batch normalization to improve convergence**
- **Dropout and L2 regularization to prevent overfitting**
- **Early stopping mechanism**

## 📊 **Interactive Dashboard**
A **dynamic, user-friendly dashboard** was developed to visualize energy consumption trends and model predictions:
- **Feature Selection Panel:** Displays feature importance rankings using various methods.
- **Model Evaluation Panel:** Visualizes the performance metrics (MSE, RMSE, R², etc.).
- **Interactive Time-Series Graphs:** Enables users to explore historical consumption data and forecasts.
- **AI-powered Insights:** Uses **Gemini AI** for automated analysis and recommendations.

## 🛠️ Installation & Setup
### 1️⃣ Clone the Repository
```sh
git clone https://github.com/yourusername/IST-Energy-Forecast.git
cd IST-Energy-Forecast
```



### NOTE THAT THE BASE TEMPLATE THAT I USED WAS

Template Name: iPortfolio
Template URL: https://bootstrapmade.com/iportfolio-bootstrap-portfolio-websites-template/
Author: BootstrapMade.com
License: https://bootstrapmade.com/license/
