import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import time
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import sys
from sklearn.decomposition import PCA

def clean_data(df):
    cleaning_steps = {}

    # Step 1: Drop columns with >50% NaN
    df2 = df.dropna(thresh=len(df) * 0.5, axis=1)
    cleaning_steps['step1'] = f"Dropped {len(df.columns) - len(df2.columns)} columns with >50% NaN values"

    # Step 2: Handle various forms of unknown values
    df3 = df2.copy()
    unknown_values = ['unknown', 'Unknown', '???', '????', '??', 'NA', 'N/A', '', ' ']
    for column in df3.columns:
        if df3[column].dtype == 'object':
            df3[column] = df3[column].replace(unknown_values, np.nan)
        elif df3[column].dtype in ['int64', 'float64']:
            df3[column] = pd.to_numeric(df3[column].replace(unknown_values, np.nan), errors='coerce')
    cleaning_steps['step2'] = f"Replaced various forms of unknown values with NaN in {len(df3.columns)} columns"

    # Step 3: Fill NaN with mean or mode
    df4 = df3.copy()
    for column in df4.columns:
        if df4[column].dtype in ['int64', 'float64']:
            mean_value = df4[column].mean()
            df4[column] = df4[column].fillna(mean_value)
        else:
            mode_value = df4[column].mode().iloc[0] if not df4[column].mode().empty else np.nan
            df4[column] = df4[column].fillna(mode_value)
    cleaning_steps['step3'] = f"Filled NaN values with mean for numeric columns and mode for categorical columns in {len(df4.columns)} columns"

    # Step 4: Remove any remaining rows with NaN values
    df5 = df4.dropna()
    cleaning_steps['step4'] = f"Removed {len(df4) - len(df5)} rows with remaining NaN values"

    return df5, cleaning_steps

def get_model_size(model):
    return sys.getsizeof(model)

def is_categorical(y):
    return len(np.unique(y)) < 10 and isinstance(y.iloc[0], (str, bool, np.bool_))

def suggest_best_algorithm(performances, task_type):
    # Normalize the metrics
    normalized_performances = {}
    for algo, perf in performances.items():
        normalized_perf = {}
        for metric, value in perf.items():
            if metric in ['Time', 'Space']:
                # Lower is better for Time and Space
                normalized_perf[metric] = 1 / (value + 1e-10)  # Adding small epsilon to avoid division by zero
            elif metric == 'MSE':
                # Lower is better for MSE
                normalized_perf[metric] = 1 / (value + 1e-10)
            else:
                # Higher is better for Accuracy and R-squared
                normalized_perf[metric] = value
        normalized_performances[algo] = normalized_perf
   
    # Calculate overall score (you can adjust weights as needed)
    scores = {}
    for algo, perf in normalized_performances.items():
        if task_type == "classification":
            scores[algo] = 0.5 * perf['Accuracy'] + 0.25 * perf['Time'] + 0.25 * perf['Space']
        else:
            scores[algo] = 0.4 * perf['R-squared'] + 0.3 * perf['MSE'] + 0.15 * perf['Time'] + 0.15 * perf['Space']
   
    # Return the algorithm with the highest score
    return max(scores, key=scores.get)

def main():
    st.set_page_config(layout="wide", page_title="Advanced Data Analysis Tool", page_icon="📊")
   
    # Add custom CSS for dark theme
    st.markdown("""
    <style>
    .stApp {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .stButton>button {
        background-color: #007BFF;  /* Changed to a blue color */
        color: white;
        font-weight: bold;
        transition: background-color 0.3s ease;  /* Added transition for smooth hover effect */
    }
    .stButton>button:hover {
        background-color: #0056b3;  /* Darker blue on hover */
    }
    .stSelectbox {
        background-color: #2C2C2C;
        color: #FFFFFF;
    }
    .stTextInput>div>div>input {
        background-color: #2C2C2C;
        color: #FFFFFF;
    }
    .stDataFrame {
        background-color: #2C2C2C;
        color: #FFFFFF;
    }
    .css-1kyxreq {
        background-color: #2C2C2C;
        color: #FFFFFF;
    }
    .css-1kyxreq:hover {
        background-color: #3C3C3C;
        color: #FFFFFF;
    }
    </style>
    """, unsafe_allow_html=True)
   
    if 'page' not in st.session_state:
        st.session_state.page = 'upload'
 
    if st.session_state.page == 'upload':
        show_upload_page()
    elif st.session_state.page == 'clean':
        show_clean_page()
    elif st.session_state.page == 'process':
        show_process_page()
    elif st.session_state.page == 'Advanced Analysis':
        show_advanced_analysis_page()

def show_upload_page():
    st.title("📤 Data Upload and Preview")
 
    col1, col2 = st.columns([1, 2])
 
    with col1:
        st.header("📁 Data Set Input")
        uploaded_file = st.file_uploader("Import CSV file", type="csv")
 
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
 
        with col2:
            st.header("👀 Realtime Preview")
            st.dataframe(df, height=300)
 
        with col1:
            st.header("🔍 Input Fields")
            selected_columns = st.multiselect("Select columns for evaluation", df.columns)
            st.session_state.selected_columns = selected_columns
 
        if st.button("Clean Data 🧹"):
            if len(selected_columns) > 0:
                with st.spinner('Cleaning data...'):
                    time.sleep(2)  # Simulating cleaning process
                st.session_state.page = 'clean'
                st.rerun()
            else:
                st.warning("⚠️ Please select at least one column for evaluation.")

def show_clean_page():
    st.title("🧼 Data Cleaning")
 
    col1, col2 = st.columns([1, 2])
 
    with col1:
        st.header("🔧 Cleaning Process")
        if st.button("Start Cleaning 🚀"):
            with st.spinner('Cleaning data...'):
                cleaned_df, cleaning_steps = clean_data(st.session_state.df[st.session_state.selected_columns])
                st.session_state.cleaned_df = cleaned_df
                st.session_state.cleaning_steps = cleaning_steps
                time.sleep(2)  # Simulating cleaning process
            st.success("✅ Data cleaning completed!")
 
    if 'cleaned_df' in st.session_state:
        with col2:
            st.header("🔬 Cleaned Data Preview")
            st.dataframe(st.session_state.cleaned_df, height=300)
 
        with col1:
            st.header("📝 Cleaning Steps")
            for step, description in st.session_state.cleaning_steps.items():
                st.write(f"**{step}**: {description}")
 
        if st.button("Proceed to Analysis 📊"):
            with st.spinner('Preparing analysis...'):
                time.sleep(2)  # Simulating preparation process
            st.session_state.page = 'process'
            st.rerun()

def show_process_page():
    st.title("🧪 Algorithm Selection and Evaluation")

    if 'cleaned_df' not in st.session_state:
        st.warning("⚠️ Please upload and clean your data first.")
        return

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.header("🔢 Cleaned DataFrame")
        st.dataframe(st.session_state.cleaned_df, height=300)

    with col2:
        st.header("🎯 Target Variable")
        target_column = st.selectbox("Select target variable", st.session_state.cleaned_df.columns)

    with col3:
        st.header("📊 Performance Metrics")
        time_complexity = st.checkbox("⏱️ Time Complexity")
        space_complexity = st.checkbox("💾 Space Complexity")
        performance = st.checkbox("🏆 Performance Metrics")

    X = st.session_state.cleaned_df.drop(columns=[target_column])
    y = st.session_state.cleaned_df[target_column]
   
    if is_categorical(y):  # Classification task
        le = LabelEncoder()
        y = le.fit_transform(y)
        task_type = "classification"
        algorithms = {
            'SVM': SVC(),
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier(),
            'KNN': KNeighborsClassifier(),
            'Neural Networks': MLPClassifier()
        }
    else:  # Regression task
        task_type = "regression"
        algorithms = {
            'SVR': SVR(),
            'Decision Tree': DecisionTreeRegressor(),
            'Random Forest': RandomForestRegressor(),
            'KNN': KNeighborsRegressor(),
            'Linear Regression': LinearRegression(),
            'Neural Networks': MLPRegressor()
        }
   
    st.header("🤖 Algorithm Selection")
    selected_algos = st.multiselect("Select algorithms", list(algorithms.keys()))
   
    if len(selected_algos) > 0:
        with st.spinner('Training and evaluating algorithms...'):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            performances = {}
            for algo_name in selected_algos:
                start_time = time.time()
                model = algorithms[algo_name]
               
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    end_time = time.time()

                    model_size = get_model_size(model)

                    if task_type == "regression":
                        mse = mean_squared_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        performances[algo_name] = {
                            'MSE': mse,
                            'R-squared': r2,
                            'Time': end_time - start_time,
                            'Space': model_size
                        }
                    else:
                        acc = accuracy_score(y_test, y_pred)
                        performances[algo_name] = {
                            'Accuracy': acc,
                            'Time': end_time - start_time,
                            'Space': model_size
                        }
                except Exception as e:
                    st.warning(f"⚠️ Error occurred while fitting {algo_name}: {str(e)}")
                    continue

            time.sleep(2)  # Simulating evaluation process

        # Display results
        col4, col5 = st.columns([1, 1])
       
        with col4:
            st.header("📊 Performance Results")
            if performance:
                st.subheader("Performance Scores")
                for algo, perf in performances.items():
                    st.write(f"**{algo}**")
                    for k, v in perf.items():
                        if k not in ['Time', 'Space']:
                            st.write(f"- {k}: {v:.4f}")
                    st.write("")
            if time_complexity:
                st.subheader("⏱️ Time Complexity (seconds)")
                for algo, perf in performances.items():
                    st.write(f"**{algo}**: {perf['Time']:.4f}")
            if space_complexity:
                st.subheader("💾 Space Complexity (bytes)")
                for algo, perf in performances.items():
                    st.write(f"**{algo}**: {perf['Space']}")

        with col5:
            st.header("📈 Performance Charts")
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
           
            # Update matplotlib style for dark theme
            plt.style.use('dark_background')
           
            # Bar plot
            if task_type == "classification":
                sns.barplot(x=list(performances.keys()), y=[perf['Accuracy'] for perf in performances.values()], ax=ax1)
                ax1.set_ylabel('Accuracy')
                ax1.set_title('Classification Algorithm Accuracy Comparison')
            else:
                sns.barplot(x=list(performances.keys()), y=[perf['R-squared'] for perf in performances.values()], ax=ax1)
                ax1.set_ylabel('R-squared')
                ax1.set_title('Regression Algorithm R-squared Comparison')
            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
           
            # Line plot
            x = range(len(performances))
            if task_type == "classification":
                y = [perf['Accuracy'] for perf in performances.values()]
                ax2.set_ylabel('Accuracy')
            else:
                y = [perf['R-squared'] for perf in performances.values()]
                ax2.set_ylabel('R-squared')
            ax2.plot(x, y, marker='o')
            ax2.set_xticks(x)
            ax2.set_xticklabels(performances.keys(), rotation=45, ha='right')
            ax2.set_title('Performance Trend')
           
            plt.tight_layout()
            st.pyplot(fig)

        # Suggest the best algorithm
        st.header("🏆 Best Algorithm Suggestion")
        best_algo = suggest_best_algorithm(performances, task_type)
        st.write(f"Based on the analysis, the recommended algorithm is: **{best_algo}**")
   
    # Add a button to go to the "Advanced Analysis" page
    if st.button("Proceed to Advanced Analysis 🔬"):
        st.session_state.page = 'Advanced Analysis'
        st.rerun()

def show_advanced_analysis_page():
    st.title("🔬 Advanced Analysis")

    if 'cleaned_df' not in st.session_state:
        st.warning("⚠️ Please upload and clean your data first.")
        return

    st.header("🔍 Feature Importance and PCA Visualization")
    target_column = st.selectbox("Select target variable", st.session_state.cleaned_df.columns)
    X = st.session_state.cleaned_df.drop(columns=[target_column])
    y = st.session_state.cleaned_df[target_column]

    # Determine if it's a classification or regression task
    is_classification = is_categorical(y)

    if is_classification:
        le = LabelEncoder()
        y = le.fit_transform(y)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Fit the model
    model.fit(X, y)

    # Feature Importance
    importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({'feature': X.columns, 'importance': importance})
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Feature Importance")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x='importance', y='feature', data=feature_importance_df.head(10), ax=ax)
        ax.set_title("Top 10 Feature Importance")
        st.pyplot(fig)

    # PCA Visualization
    with col2:
        st.subheader("🔮 PCA Visualization")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
        df_pca['target'] = y

        fig, ax = plt.subplots(figsize=(8, 5))
        if is_classification:
            sns.scatterplot(x='PC1', y='PC2', hue='target', data=df_pca, ax=ax, palette='deep')
            ax.set_title("PCA Visualization (Classification)")
        else:
            scatter = ax.scatter(df_pca['PC1'], df_pca['PC2'], c=df_pca['target'], cmap='viridis')
            plt.colorbar(scatter)
            ax.set_title("PCA Visualization (Regression)")
       
        ax.set_xlabel("First Principal Component")
        ax.set_ylabel("Second Principal Component")
        st.pyplot(fig)

    # Explained Variance Ratio
    st.subheader("📉 Explained Variance Ratio")
    explained_variance_ratio = pca.explained_variance_ratio_
    st.write(f"Explained Variance Ratio: PC1 {explained_variance_ratio[0]:.2f}, PC2 {explained_variance_ratio[1]:.2f}")
    st.write(f"Total Explained Variance: {sum(explained_variance_ratio):.2f}")

    # Feature Correlation Heatmap
    st.subheader("🔥 Feature Correlation Heatmap")
    corr = X.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=False, cmap='coolwarm', ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    st.pyplot(fig)

    # Distribution of Target Variable
    st.subheader("📊 Distribution of Target Variable")
    fig, ax = plt.subplots(figsize=(10, 6))
    if is_classification:
        sns.countplot(x=y, ax=ax)
        ax.set_title("Distribution of Target Classes")
        ax.set_xlabel("Class")
    else:
        sns.histplot(y, kde=True, ax=ax)
        ax.set_title("Distribution of Target Variable")
        ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # Offer to go back to the main analysis page
    if st.button("🔙 Back to Main Analysis"):
        st.session_state.page = 'process'
        st.rerun()

if __name__ == "__main__":
    main()