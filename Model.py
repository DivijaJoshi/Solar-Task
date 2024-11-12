import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import classification_report, mean_squared_error, precision_recall_curve, roc_curve, auc
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Set page config
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

def load_and_prepare_data():
    """
    ETL Pipeline for data preparation
    Returns cleaned and feature-engineered dataset
    """
    # Load dataset
    data = pd.read_csv('playground-series-s3e17/train.csv')
    
    # Data Cleaning
    data = data.ffill().bfill()
    
    # Feature Engineering
    data['Failure'] = data[['TWF', 'HDF', 'PWF', 'OSF', 'RNF']].sum(axis=1) > 0
    
    # Advanced Feature Engineering
    data['Torque_RollingMean'] = data['Torque [Nm]'].rolling(window=10, min_periods=1).mean()
    data['RPM_Variance'] = data['Rotational speed [rpm]'].rolling(window=10, min_periods=1).var()
    data['Temperature_Difference'] = data['Process temperature [K]'] - data['Air temperature [K]']
    data['Power'] = data['Torque [Nm]'] * data['Rotational speed [rpm]'] / 9550  # Mechanical Power in kW
    data['Temperature_Rate'] = data['Process temperature [K]'].diff().fillna(0)
    data['Wear_Rate'] = data['Tool wear [min]'].diff().fillna(0)
    data['Power_to_Wear_Ratio'] = data['Power'] / (data['Tool wear [min]'] + 1)
    
    # Simulate maintenance history
    data['Last_Maintenance'] = np.random.randint(0, 1000, size=len(data))
    data['Maintenance_Count'] = np.random.randint(0, 5, size=len(data))
    
    return data

@st.cache_data
def get_failure_patterns(data):
    """Analyze common patterns leading to failures"""
    failure_data = data[data['Failure'] == 1]
    patterns = {
        'high_temp': failure_data[failure_data['Temperature_Difference'] > failure_data['Temperature_Difference'].mean()].shape[0],
        'high_wear': failure_data[failure_data['Tool wear [min]'] > failure_data['Tool wear [min]'].mean()].shape[0],
        'high_power': failure_data[failure_data['Power'] > failure_data['Power'].mean()].shape[0]
    }
    return patterns

def create_pipelines(model_params=None):
    """Create ML pipelines with configurable parameters"""
    if model_params is None:
        model_params = {
            'n_estimators_clf': 200,
            'max_depth_clf': 15,
            'n_estimators_reg': 150,
            'max_depth_reg': 7
        }
    
    # Use StratifiedKFold for classification
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    
    clf_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('feature_selection', SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))),
        ('classifier', RandomForestClassifier(
            n_estimators=model_params['n_estimators_clf'],
            max_depth=model_params['max_depth_clf'],
            class_weight='balanced',
            random_state=42
        ))
    ])
    
    reg_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('feature_selection', SelectFromModel(GradientBoostingRegressor(n_estimators=100, random_state=42))),
        ('regressor', GradientBoostingRegressor(
            n_estimators=model_params['n_estimators_reg'],
            max_depth=model_params['max_depth_reg'],
            learning_rate=0.1,
            random_state=42
        ))
    ])
    
    return clf_pipeline, reg_pipeline

def calculate_maintenance_metrics(failure_prob, tool_wear, last_maintenance, thresholds):
    """
    Calculate maintenance recommendations based on predictions and customizable thresholds
    """
    risk_threshold = thresholds['risk']
    wear_threshold = thresholds['wear']
    maintenance_age_threshold = thresholds['maintenance_age']
    
    maintenance_due = (
        (failure_prob > risk_threshold) | 
        (tool_wear > wear_threshold) | 
        (last_maintenance > maintenance_age_threshold)
    )
    
    priority = np.where(
        failure_prob > 0.7, 'High',
        np.where(failure_prob > 0.4, 'Medium', 'Low')
    )
    
    estimated_days = np.where(
        maintenance_due,
        0,
        np.ceil((wear_threshold - tool_wear) / np.maximum(0.1, tool_wear.mean()))
    )
    
    next_maintenance = np.where(
        maintenance_due,
        'Immediate',
        np.where(
            estimated_days <= 7,
            'Within 1 week',
            np.where(
                estimated_days <= 30,
                'Within 1 month',
                'No immediate action needed'
            )
        )
    )
    
    return maintenance_due, priority, next_maintenance, estimated_days

def create_failure_analysis_plots(data, X_train, y_train, X_test, y_test, predictions):
    """Create various failure analysis visualizations"""
    
    # Train the model (assuming a RandomForestClassifier for this example)
    model = RandomForestClassifier(n_estimators=100, max_depth=10)
    model.fit(X_train, y_train)  # Train the model with training data
    
    # Time series of key metrics
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        y=data['Tool wear [min]'],
        name='Tool Wear',
        line=dict(color='blue')
    ))
    fig1.add_trace(go.Scatter(
        y=data['Temperature_Difference'],
        name='Temperature Difference',
        line=dict(color='red')
    ))
    fig1.add_trace(go.Scatter(
        y=data['Power'],
        name='Power',
        line=dict(color='green')
    ))
    fig1.update_layout(title='Key Metrics Over Time', xaxis_title='Observation')
    
    # Failure probability distribution
    fig2 = px.histogram(
        predictions,
        nbins=50,
        title='Distribution of Failure Probabilities'
    )
    
    # Get predicted probabilities for the positive class
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class (binary classification)
    y_test_cls = y_test  # True class labels
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test_cls, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {roc_auc:.2f})'
    ))
    fig3.plot_bgcolor = 'white'
    fig3.update_layout(
        title='Receiver Operating Characteristic (ROC) Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        xaxis_range=[0, 1],
        yaxis_range=[0, 1]
    )
    
    return fig1, fig2, fig3

def plot_maintenance_calendar(schedule_df):
    """Create an interactive maintenance calendar view"""
    fig = px.timeline(
        schedule_df,
        x_start='Scheduled_Date',
        x_end='Due_Date',
        y='Equipment_ID',
        color='Priority',
        title='Maintenance Schedule Timeline'
    )
    fig.update_yaxes(autorange="reversed", title="Equipment ID")
    fig.update_xaxes(title="Date")
    return fig 

def sidebar_controls():
    """Create sidebar controls for user input"""
    st.sidebar.header('Dashboard Controls')
    
    # Model Parameters
    st.sidebar.subheader('Model Parameters')
    n_estimators_clf = st.sidebar.slider('Number of Trees (Classification)', 50, 300, 200)
    max_depth_clf = st.sidebar.slider('Max Tree Depth (Classification)', 5, 30, 15)
    n_estimators_reg = st.sidebar.slider('Number of Trees (Regression)', 50, 300, 150)
    max_depth_reg = st.sidebar.slider('Max Tree Depth (Regression)', 5, 30, 7)
    
    # Threshold Settings
    st.sidebar.subheader('Maintenance Thresholds')
    risk_threshold = st.sidebar.slider('Risk Threshold', 0.0, 1.0, 0.3)
    wear_threshold = st.sidebar.slider('Wear Threshold', 100, 300, 200)
    maintenance_age = st.sidebar.slider('Maintenance Age Threshold', 500, 1000, 800)
    
    # Visualization Settings
    st.sidebar.subheader('Visualization Settings')
    plot_height = st.sidebar.slider('Plot Height', 400, 800, 600)
    color_theme = st.sidebar.selectbox('Color Theme', ['blues', 'reds', 'greens'])
    
    return {
        'model_params': {
            'n_estimators_clf': n_estimators_clf,
            'max_depth_clf': max_depth_clf,
            'n_estimators_reg': n_estimators_reg,
            'max_depth_reg': max_depth_reg
        },
        'thresholds': {
            'risk': risk_threshold,
            'wear': wear_threshold,
            'maintenance_age': maintenance_age
        },
        'viz_params': {
            'plot_height': plot_height,
            'color_theme': color_theme
        }
    }

def main():
    st.title("🔧 Advanced Predictive Maintenance Dashboard")
    
    # Get user input parameters
    params = sidebar_controls()
    
    # Introduction
    with st.expander("ℹ️ Dashboard Overview", expanded=True):
        st.markdown("""
        This dashboard provides comprehensive predictive maintenance analytics for manufacturing equipment:
        
        1. *Real-time Monitoring*: Track equipment health metrics and failure predictions
        2. *Maintenance Planning*: Get AI-powered maintenance recommendations
        3. *Performance Analysis*: Analyze historical data and model performance
        4. *Interactive Features*: Customize thresholds and visualization parameters
        
        Use the sidebar controls to adjust model parameters and thresholds.
        """)
    
    # Load and prepare data
    with st.spinner("Loading and preparing data..."):
        data = load_and_prepare_data()
        
        # Define features
        feature_columns = [
            'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]',
            'Torque [Nm]', 'Tool wear [min]', 'Torque_RollingMean', 'RPM_Variance',
            'Temperature_Difference', 'Power', 'Temperature_Rate', 'Wear_Rate',
            'Power_to_Wear_Ratio'
        ]
        
        X = data[feature_columns]
        y_classification = data['Failure']
        y_regression = data['Tool wear [min]']
    
    # Load or train models with user parameters
    model_dir = './models'
    os.makedirs(model_dir, exist_ok=True)
    
    clf_pipeline_file = os.path.join(model_dir, 'clf_pipeline.pkl')
    reg_pipeline_file = os.path.join(model_dir, 'reg_pipeline.pkl')
    
    if os.path.exists(clf_pipeline_file) and os.path.exists(reg_pipeline_file):
        # Load pre-trained models
        clf_pipeline = joblib.load(clf_pipeline_file)
        reg_pipeline = joblib.load(reg_pipeline_file)
        
        
        # Data split for prediction
        X_train, X_test, y_train_cls, y_test_cls = train_test_split(
            X, y_classification, test_size=0.2, random_state=42, stratify=y_classification
        )
        _, _, y_train_reg, y_test_reg = train_test_split(
            X, y_regression, test_size=0.2, random_state=42
        )
        
    else:
        # Train models with user parameters
        with st.spinner("Training models with selected parameters..."):
            clf_pipeline, reg_pipeline = create_pipelines(params['model_params'])
            
            # Split data for training
            X_train, X_test, y_train_cls, y_test_cls = train_test_split(
                X, y_classification, test_size=0.2, random_state=42, stratify=y_classification
            )
            _, _, y_train_reg, y_test_reg = train_test_split(
                X, y_regression, test_size=0.2, random_state=42
            )
            
            # Train models
            clf_pipeline.fit(X_train, y_train_cls)
            reg_pipeline.fit(X_train, y_train_reg)
            
            # Save models
            joblib.dump(clf_pipeline, clf_pipeline_file)
            joblib.dump(reg_pipeline, reg_pipeline_file)
            st.write("Trained and saved new models to ./models folder.")
    
    # Make predictions
    y_pred_cls = clf_pipeline.predict(X_test)
    y_pred_proba = clf_pipeline.predict_proba(X_test)[:, 1]
    y_pred_reg = reg_pipeline.predict(X_test)
    
    # Calculate maintenance recommendations
    maintenance_due, priority, next_maintenance, estimated_days = calculate_maintenance_metrics(
        y_pred_proba,
        y_pred_reg,
        data['Last_Maintenance'].iloc[-len(y_pred_cls):],
        params['thresholds']
    )
    
    # Dashboard Layout
    
    # 1. Equipment Health Overview
    st.header("📊 Equipment Health Overview")
    
    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.metric(
            "Overall Health Index",
            f"{(1 - y_pred_proba.mean()):.1%}",
            delta=f"{-y_pred_proba.mean():.1%}",
            delta_color="inverse"
        )
    
    with metric_cols[1]:
        st.metric(
            "Average Failure Risk",
            f"{y_pred_proba.mean():.1%}",
            delta=f"{(y_pred_proba.mean() - 0.3):.1%}" if y_pred_proba.mean() > 0.3 else "Normal",
            delta_color="inverse"
        )
    
    with metric_cols[2]:
        st.metric(
            "Equipment Requiring Maintenance",
            f"{maintenance_due.sum()}",
            delta=f"{maintenance_due.sum() - 10}" if maintenance_due.sum() > 10 else "Within limits"
        )
    
    with metric_cols[3]:
        st.metric(
            "Average Tool Wear",
            f"{y_pred_reg.mean():.1f} min",
            delta=f"{y_pred_reg.mean() - params['thresholds']['wear']:.1f}"
        )
    
    # 2. Interactive Analysis Tabs
    tabs = st.tabs([
        "🔍 Real-time Monitoring",
        "📈 Performance Analysis",
        "🔧 Maintenance Planning",
        "📊 Historical Analysis"
    ])
    
    # Tab 1: Real-time Monitoring
    with tabs[0]:
        # Equipment Status Summary
        status_df = pd.DataFrame({
            'Status': ['Healthy', 'Warning', 'Critical'],
            'Count': [
                (y_pred_proba < 0.3).sum(),
                ((y_pred_proba >= 0.3) & (y_pred_proba < 0.7)).sum(),
                (y_pred_proba >= 0.7).sum()
            ]
        })
        fig = px.pie(
            status_df,
            values='Count',
            names='Status',
            title='Equipment Status Distribution',
            color='Status',
            color_discrete_map={
                'Healthy': 'green',
                'Warning': 'yellow',
                'Critical': 'red'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Real-time Alerts
        if maintenance_due.sum() > 0:
            st.warning(f"⚠️ {maintenance_due.sum()} equipment units require immediate attention!")
            
        # Interactive Equipment Explorer
        st.subheader("Equipment Explorer")
        selected_metric = st.selectbox(
            "Select Metric to Monitor:",
            options=['Temperature_Difference', 'Tool wear [min]', 'Power', 'Torque [Nm]', 'Rotational speed [rpm]']
        )
        
        time_window = st.slider(
            "Time Window (last N observations)",
            min_value=10,
            max_value=len(data),
            value=100
        )
        
        # Plot selected metric
        fig = px.line(
            data.tail(time_window),
            y=selected_metric,
            title=f'{selected_metric} - Last {time_window} Observations'
        )
        fig.add_hline(
            y=data[selected_metric].mean(),
            line_dash="dash",
            annotation_text="Average"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Performance Analysis
    with tabs[1]:
        st.subheader("Model Performance Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Classification Performance
            st.markdown("### Failure Prediction Performance")
            st.text("Classification Report:")
            st.code(classification_report(y_test_cls, y_pred_cls))
            
            # Precision-Recall curve
            precision, recall, _ = precision_recall_curve(y_test_cls, y_pred_proba)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=recall, y=precision,
                mode='lines',
                name='Precision-Recall curve',
                fill='tozeroy'
            ))
            fig.update_layout(
                title='Precision-Recall Curve',
                xaxis_title='Recall',
                yaxis_title='Precision'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Regression Performance
            st.markdown("### Tool Wear Prediction Performance")
            mse = mean_squared_error(y_test_reg, y_pred_reg)
            rmse = np.sqrt(mse)
            st.metric("Root Mean Squared Error", f"{rmse:.2f}")
            
            # Feature Importance
            feature_names = feature_columns
            feature_importances = clf_pipeline.named_steps['classifier'].feature_importances_
            
            # Ensure feature_names and feature_importances are of the same length
            len_features = len(feature_names)
            len_importances = len(feature_importances)
            
            if len_features > len_importances:
                feature_names = feature_names[:len_importances]
            elif len_importances > len_features:
                feature_importances = feature_importances[:len_features]
            
            feature_imp = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importances
            }).sort_values('Importance', ascending=True)
            
            fig = px.bar(
                feature_imp,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Feature Importance Analysis'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation Analysis
        st.subheader("Feature Correlation Analysis")
        
        # Calculate the correlation matrix
        correlation_matrix = data[feature_columns].corr()
        
        # Create a heatmap using plotly
        correlation_fig = px.imshow(correlation_matrix, 
                                    text_auto=True, 
                                    color_continuous_scale='Viridis', 
                                    title="Feature Correlation Heatmap")
        
        # Customize layout for better display
        correlation_fig.update_layout(
            width=800, 
            height=600,
            xaxis_title="Features",
            yaxis_title="Features",
            xaxis={'tickangle': 45},
            yaxis={'tickangle': -45}
        )
        
        # Display the correlation heatmap
        st.plotly_chart(correlation_fig, use_container_width=True)
        
            
    # Tab 3: Maintenance Planning
    with tabs[2]:
        st.subheader("Maintenance Schedule and Recommendations")
        
        # Create maintenance schedule DataFrame
        schedule_df = pd.DataFrame({
            'Equipment_ID': range(1, len(maintenance_due) + 1),
            'Failure_Probability': y_pred_proba,
            'Tool_Wear': y_pred_reg,
            'Priority': priority,
            'Next_Maintenance': next_maintenance,
            'Estimated_Days': estimated_days
        })
        
        # Add simulated dates
        today = datetime.now()
        schedule_df['Scheduled_Date'] = [
            today + timedelta(days=int(d)) for d in schedule_df['Estimated_Days']
        ]
        schedule_df['Due_Date'] = [
            d + timedelta(days=7) for d in schedule_df['Scheduled_Date']
        ]
        
        # Maintenance Calendar
        st.markdown("### 📅 Maintenance Calendar")
        calendar_fig = plot_maintenance_calendar(schedule_df)
        st.plotly_chart(calendar_fig, use_container_width=True)
        
        # Priority-based maintenance table
        st.markdown("### 🔧 Priority Maintenance Tasks")
        priority_df = schedule_df[schedule_df['Priority'] == 'High'].sort_values(
            'Failure_Probability', ascending=False
        )
        
        if not priority_df.empty:
            st.dataframe(
                priority_df[['Equipment_ID', 'Failure_Probability', 'Tool_Wear', 'Next_Maintenance']],
                use_container_width=True
            )
        else:
            st.success("No high-priority maintenance tasks at the moment!")
        
        # Maintenance Cost Analysis
        st.markdown("### 💰 Maintenance Cost Projection")
        est_cost_per_maintenance = st.number_input(
            "Estimated cost per maintenance (USD):",
            value=1000,
            step=100
        )
        
        total_maintenance = maintenance_due.sum()
        projected_cost = total_maintenance * est_cost_per_maintenance
        
        cost_col1, cost_col2 = st.columns(2)
        with cost_col1:
            st.metric(
                "Projected Maintenance Cost",
                f"${projected_cost:,.2f}",
                delta=f"${projected_cost - 10000:,.2f}" if projected_cost > 10000 else "Within budget"
            )
        
        with cost_col2:
            st.metric(
                "Average Cost per Equipment",
                f"${projected_cost/len(maintenance_due):,.2f}"
            )
    
    # Tab 4: Historical Analysis
    with tabs[3]:
        st.subheader("Historical Performance Analysis")
        
        # Time series analysis
        st.markdown("### 📈 Historical Trends")
        metric_for_history = st.selectbox(
            "Select metric for historical analysis:",
            options=['Tool wear [min]', 'Temperature_Difference', 'Power', 'Failure']
        )
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=data[metric_for_history],
            mode='lines',
            name=metric_for_history
        ))
        
        # Add trend line
        z = np.polyfit(range(len(data)), data[metric_for_history], 1)
        p = np.poly1d(z)
        fig.add_trace(go.Scatter(
            y=p(range(len(data))),
            mode='lines',
            name='Trend',
            line=dict(dash='dash')
        ))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Failure patterns analysis
        st.markdown("### 🔍 Failure Patterns")
        patterns = get_failure_patterns(data)
        
        pattern_cols = st.columns(3)
        for i, (pattern, count) in enumerate(patterns.items()):
            with pattern_cols[i]:
                st.metric(
                    f"Failures due to {pattern.replace('_', ' ').title()}",
                    count,
                    delta=f"{count/len(data['Failure'])*100:.1f}% of total"
                )
    
    # Footer with additional information
    st.markdown("---")
    st.markdown("""
    ### 📝 Notes and Recommendations
    - Adjust thresholds in the sidebar to customize maintenance triggers
    - Regular model retraining is recommended for optimal performance
    - Contact maintenance team for immediate issues
    """)
    
    # Download section for reports
    if st.button("Generate Maintenance Report"):
        # Create report DataFrame
        report_df = pd.DataFrame({
            'Equipment_ID': range(1, len(maintenance_due) + 1),
            'Failure_Risk': y_pred_proba,
            'Tool_Wear': y_pred_reg,
            'Maintenance_Priority': priority,
            'Next_Maintenance': next_maintenance,
            'Days_Until_Maintenance': estimated_days
        })
        
        # Convert to CSV
        csv = report_df.to_csv(index=False)
        st.download_button(
            label="Download Maintenance Report",
            data=csv,
            file_name="maintenance_report.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
