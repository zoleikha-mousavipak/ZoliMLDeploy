import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Set the title of the dashboard
st.title("Dashboard & Machine Learning Classification")

# File uploader widget for CSV and Excel files
uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

# Check if a file has been uploaded
if uploaded_file is not None:
    try:
        # Read the file based on its extension
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Display a preview of the raw data
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        # Check if the 'failure' column exists in the dataset
        if 'failure' in df.columns:
            st.subheader("Failure Distribution")
            
            # Count the occurrences of each value in the failure column
            failure_counts = df['failure'].value_counts().reset_index()
            failure_counts.columns = ['failure', 'count']
            
            # Create a pie chart using Plotly
            fig = px.pie(
                failure_counts, 
                names='failure', 
                values='count', 
                title='Distribution of Failures',
                hole=0.3
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # --- Machine Learning Section ---
            st.divider()
            st.subheader("Machine Learning Classification")
            
            # Drop rows with missing values to prevent training errors
            df_ml = df.dropna()
            
            # Select feature columns (exclude 'failure' and non-numeric columns for simplicity)
            numeric_cols = df_ml.select_dtypes(include=['number']).columns.tolist()
            if 'failure' in numeric_cols:
                numeric_cols.remove('failure')
                
            selected_features = st.multiselect(
                "Select Feature Columns (X):", 
                numeric_cols, 
                default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
            )
            
            # Model selection dropdown
            #model_name = st.selectbox(
            #    "Select Classification Model:", 
            #    ["Random Forest", "Logistic Regression", "SVM", "KNN"]
            #)
            
            # Model selection using Radio Buttons
            model_name = st.radio(
                "Select Classification Model:", 
                ["Random Forest", "Logistic Regression", "SVM", "KNN"],
                horizontal=True  # Optional: displays options horizontally in a row
            )            
            
            # Train test split ratio slider
            test_size = st.slider("Test Set Size (%)", 10, 50, 20) / 100
            
            # Train button
            if st.button("Train Model"):
                if not selected_features:
                    st.warning("Please select at least one feature column.")
                else:
                    # Prepare X and y
                    X = df_ml[selected_features]
                    y = df_ml['failure']
                    
                    # Split the dataset into training and testing sets
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42
                    )
                    
                    # Initialize the selected model
                    if model_name == "Random Forest":
                        model = RandomForestClassifier(random_state=42)
                    elif model_name == "Logistic Regression":
                        model = LogisticRegression(max_iter=1000)
                    elif model_name == "SVM":
                        model = SVC()
                    elif model_name == "KNN":
                        model = KNN_model = KNeighborsClassifier()
                        model = KNN_model # assign to model variable
                        
                    # Train the model
                    model.fit(X_train, y_train)
                    
                    # Make predictions on the test set
                    y_pred = model.predict(X_test)
                    
                    # Calculate accuracy
                    acc = accuracy_score(y_test, y_pred)
                    
                    # Display results
                    st.success(f"Model ({model_name}) trained successfully!")
                    st.metric(label="Model Accuracy", value=f"{acc * 100:.2f}%")
                    
                    # Display classification report
                    st.text("Classification Report:")
                    report = classification_report(y_test, y_pred, output_dict=True)
                    st.dataframe(pd.DataFrame(report).transpose())
                    
        else:
            st.error("Error: The uploaded file does not contain a column named 'failure'.")
            
    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.info("Please upload a CSV or Excel file to start.")