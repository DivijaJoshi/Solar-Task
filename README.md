# Solar-Task

## Our model is hosted on the below link: 
https://huggingface.co/spaces/divija05/predictive_maintenance_equipment

## Dataset Availability:
https://www.kaggle.com/code/vineetmehar/binary-classification-of-machine-failure





# Predictive Maintenance Dashboard for Manufacturing Equipment

This project provides an **advanced predictive maintenance dashboard** to monitor and maintain manufacturing equipment. Built using `Streamlit`, `scikit-learn`, and `Plotly`, the dashboard provides insights into equipment health, failure predictions, maintenance recommendations, and performance analytics. The dashboard is currently hosted [here](https://huggingface.co/spaces/divija05/predictive_maintenance_equipment).

## Problem Statement

Manufacturing equipment experiences wear and tear, which can lead to breakdowns and costly downtime. This dashboard aims to proactively predict equipment failure and recommend timely maintenance by analyzing historical and real-time data. The main objectives are:
1. Predict equipment failures.
2. Suggest optimal maintenance schedules.
3. Enable interactive monitoring and data-driven decision-making for maintenance.

## Features

1. **Real-time Monitoring**: Track key metrics like temperature difference, tool wear, and power usage. Visualizations such as line charts and status distributions help identify risky equipment in real time.

2. **Failure Prediction & Maintenance Planning**: Use machine learning models (Random Forest Classifier for failure prediction and Gradient Boosting Regressor for tool wear estimation) to classify equipment health and suggest maintenance priorities based on customizable thresholds.

3. **Performance Analysis**: Analyze model performance using metrics like precision-recall and ROC curves. Feature importance and correlation analysis help understand which factors most affect equipment failure.

4. **Interactive Visualization**: Customize plots, thresholds, and model parameters directly in the dashboard sidebar.

5. **Maintenance Schedule**: View a timeline of scheduled maintenance events and predict next maintenance dates based on wear and failure risk.

## Technology Stack

- **Frontend**: Streamlit for interactive dashboard UI
- **Data Processing & Modeling**: Scikit-learn for machine learning, Pandas for data manipulation, and NumPy for numerical operations
- **Visualization**: Plotly for interactive data visualizations and Seaborn for static plots
- **Deployment**: Hosted on Hugging Face Spaces for easy access

## Installation

To run this dashboard locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://huggingface.co/spaces/divija05/predictive_maintenance_equipment
   cd predictive-maintenance-dashboard
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

## File Structure

- `app.py`: Main file containing code for data loading, model training, dashboard layout, and interactive visualizations.
- `models/`: Directory where trained models are saved.
- `data/`: Contains the input CSV file (`train.csv`) used for predictions.
- `requirements.txt`: Lists all required dependencies.
  
## Model Training & Caching

- Models are trained and cached locally. If pre-trained models exist in the `models/` directory, they are loaded instead of retraining. 
- Random Forest and Gradient Boosting models are used to classify equipment failure risk and predict tool wear.

## Usage

After running the app locally or accessing it online, the dashboard provides several main sections:

1. **Dashboard Controls (Sidebar)**: Adjust model parameters (e.g., tree depth, number of estimators), maintenance thresholds, and visualization settings.

2. **Equipment Health Overview**: Displays high-level metrics on overall equipment health, failure risks, and wear levels.

3. **Interactive Tabs**:
   - **Real-time Monitoring**: View live status distributions, receive maintenance alerts, and monitor selected metrics.
   - **Performance Analysis**: Analyze classification and regression performance metrics, feature importance, and correlations.
   - **Maintenance Planning**: Visualize the maintenance timeline and recommendations based on risk thresholds.
   - **Historical Analysis**: Explore historical equipment health and failure patterns.

4. **Model Performance Metrics**: View the classification report and feature importance rankings to understand model effectiveness.

## Sample Data

The dashboard requires data in CSV format, with columns similar to:
- **Air temperature [K]**
- **Process temperature [K]**
- **Rotational speed [rpm]**
- **Torque [Nm]**
- **Tool wear [min]**

The `train.csv` file in `data/` provides an example dataset used for modeling.

## Future Enhancements

- Integration with real-time data sources (e.g., IoT sensors)
- Additional algorithms for failure prediction
- Custom alerts and email notifications for critical equipment

## License

This project is open source and available under the MIT License.

## Acknowledgments

This project leverages multiple open-source libraries and was deployed using Hugging Face Spaces. Special thanks to the machine learning and data science communities for resources and inspiration.

## Contact

For questions or feedback, please reach out to the repository maintainer.

---


