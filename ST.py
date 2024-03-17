#!/usr/bin/env python
# coding: utf-8

## <a name="introduction">1. Introduction</a>

#### Business Problem
# Our objective is to identify factors that contribute to passenger satisfaction or dissatisfaction, so that the airline can address them to enhance customer service and improve customer retention.

### Preliminary Dependency Imports

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
#matplotlib.use('TkAgg')  # or any other backend that supports GUI rendering
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import CategoricalNB
from xgboost import XGBClassifier
import shap
import pickle
import lime.lime_tabular
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
st.cache_data




from matplotlib import rcParams

st.set_option('deprecation.showPyplotGlobalUse', False)



# # Streamlit APP



# Load the preprocessed data
@st.cache
def load_data():
    return pd.read_csv("preprocessed_data.csv")




# Visualize data using violin plots
def visualize_data(data):
    fig = plt.figure(figsize = (10,7))
    data.satisfaction.value_counts(normalize = True).plot(kind='bar', alpha = 0.9, rot=0)
    plt.title('Customer satisfaction')
    plt.ylabel('Percent')
    #plt.show()
    st.pyplot()

    categoricals = ["inflight_wifi_service", "departure_arrival_time_convenient", "ease_of_online_booking", 
                "gate_location", "food_and_drink", "online_boarding", "seat_comfort", "inflight_entertainment", 
                "on_board_service", "leg_room_service", "baggage_handling", "checkin_service", 
                "inflight_service", "cleanliness"]

    data.hist(column = categoricals, layout=(4,4), label='x', figsize = (20,20));
    st.pyplot()





    with sns.axes_style(style = 'ticks'):
        d = sns.histplot(x = "gender",  hue= 'satisfaction', data = data,  
                     stat = 'percent', multiple="dodge", palette = 'Set1')
    st.pyplot()




    with sns.axes_style(style = 'ticks'):
        d = sns.histplot(x = "customer_type",  hue= 'satisfaction', data = data, 
                     stat = 'percent', multiple="dodge", palette = 'Set1')

    st.pyplot()

    with sns.axes_style(style = 'ticks'):
        d = sns.histplot(x = "class",  hue= 'satisfaction', data = data,
                     stat = 'percent', multiple="dodge", palette = 'Set1')
    st.pyplot()
                
    with sns.axes_style(style = 'ticks'):
        d = sns.histplot(x = "type_of_travel",  hue= 'satisfaction', data = data,
                     stat = 'percent', multiple="dodge", palette = 'Set1')
    st.pyplot()


    with sns.axes_style('white'):
        g = sns.catplot(x = 'age', data = data,  
                    kind = 'count', hue = 'satisfaction', order = range(7, 80),
                    height = 8.27, aspect=18.7/8.27, legend = False,
                   palette = 'Set1')

    plt.legend(loc='upper right')
    st.pyplot()


def encode_data(data):
    score_cols = ["inflight_wifi_service", "departure_arrival_time_convenient", "ease_of_online_booking", 
              "gate_location","food_and_drink", "online_boarding", "seat_comfort", "inflight_entertainment", 
              "on_board_service","leg_room_service", "baggage_handling", "checkin_service", "inflight_service",
              "cleanliness"]
    df = data.copy()
    
    encoder = OrdinalEncoder()
    
    for j in score_cols:
        df[j] = encoder.fit_transform(df[[j]]) 
    
    
    df.was_flight_delayed.replace({'no': 0, 'yes' : 1}, inplace = True)
    df['satisfaction'].replace({'neutral or dissatisfied': 0, 'satisfied': 1},inplace = True)
    df.customer_type.replace({'disloyal customer': 0, 'loyal customer': 1}, inplace = True)
    df.type_of_travel.replace({'personal travel': 0, 'business travel': 1}, inplace = True)
    df.gender.replace({'male': 0, 'female' : 1}, inplace = True)
    
    encoded_df = pd.get_dummies(df, columns = ['class'])
    
    return encoded_df


import sklearn
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import CategoricalNB
import xgboost
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler


def train_and_evaluate_models(data):
    # Generate correlation heatmap
    train_corr = data.corr()[['satisfaction']]
    plt.figure(figsize=(10, 12))
    heatmap = sns.heatmap(train_corr.sort_values(by='satisfaction', ascending=False), 
                          vmin=-1, vmax=1, annot=True, cmap='Blues')
    heatmap.set_title('Feature Correlation with Target Variable', fontdict={'fontsize':14})
    st.pyplot(plt)

    # Pre-processing and scaling dataset for feature selection
    r_scaler = MinMaxScaler()
    r_scaler.fit(data)
    train_scaled = pd.DataFrame(r_scaler.transform(data), columns=data.columns)

    # Feature selection
    X = train_scaled.drop(columns=['satisfaction'])
    y = train_scaled['satisfaction']
    selector = SelectKBest(chi2, k=10)
    selector.fit(X, y)
    X_selected = selector.transform(X)

    # Splitting into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

    # Model activation and result plot function
    def get_model_metrics(model, X_train, X_test, y_train, y_test):
        # Fit the model on the training data and run predictions on test data
        model.fit(X_train,  y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        # Obtain training accuracy as a comparative metric using Sklearn's metrics package
        train_score = model.score(X_train, y_train)
        # Obtain testing accuracy as a comparative metric using Sklearn's metrics package
        accuracy = accuracy_score(y_test, y_pred)
        # Obtain precision from predictions using Sklearn's metrics package
        precision = precision_score(y_test, y_pred)
        # Obtain recall from predictions using Sklearn's metrics package
        recall = recall_score(y_test, y_pred)
        # Obtain f1 from predictions using Sklearn's metrics package
        f1 = f1_score(y_test, y_pred)
        # Obtain ROC score from predictions using Sklearn's metrics package
        roc = roc_auc_score(y_test, y_pred_proba)

        # Outputting the metrics of the model performance
        st.write("Accuracy on Training = {}".format(train_score))
        st.write("Accuracy on Test = {} • Precision = {}".format(accuracy, precision))
        st.write("Recall = {}".format(recall))
        st.write("F1 = {} • ROC Area under Curve = {}".format(f1, roc))

        # Plot confusion matrix
        plt.figure()
        cm = confusion_matrix(y_test, y_pred)
        cm_display = sns.heatmap(cm, annot=True, cmap='Blues')
        cm_display.set_title('Confusion Matrix')
        plt.show()

        # Plot ROC curve
        plt.figure()
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.show()

    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(),
        'Random Forest': RandomForestClassifier(),
        'AdaBoost': AdaBoostClassifier(),
        'Categorical Naive Bayes': CategoricalNB(),
        'Extreme Gradient Boost': XGBClassifier()
    }

    # Iterate through models and obtain metrics
    for model_name, model in models.items():
        st.write("Model: ", model_name)
        get_model_metrics(model, X_train, X_test, y_train, y_test)    


# Explain model predictions using SHAP and LIME
def explain_model_predictions(data):
    # Load pre-trained XGBoost model
    model = pickle.load(open("model_xgb.pkl", "rb"))
    
    # Initialize SHAP explainer
    explainer = shap.Explainer(model, feature_names=data.columns)
    shap_values = explainer(data)
    # Your code for SHAP visualization here
    # For example:
    get_ipython().run_line_magic('%time', '')
    shap.initjs()
    shap.summary_plot(shap_values, X_train, class_names=model_xgb.classes_)    
    # Initialize LIME explainer
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(data.values,
                                                            feature_names=data.columns,
                                                            class_names=["Not Satisfied", "Satisfied"],
                                                            discretize_continuous=False)




# Main function to run the Streamlit app
def main():
    # Set title and sidebar options
    st.title("Airline Passenger Satisfaction Prediction")
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Visualizations of Features", "Train and Evaluate Models", "Explain Model Predictions"])

    # Load data
    data = load_data()

    # Visualizations of Features page
    if page == "Visualizations of Features":
        st.header("Visualizations of Features")

        # For example:
        viz = visualize_data(data)
        st.write(viz)


        # Train and evaluate models page
    elif page == "Train and Evaluate Models":
        st.header("Train and Evaluate Models")
        a=train_and_evaluate_models(encode_data(data))
        st.write(a)

    # Explain model predictions page
    elif page == "Explain Model Predictions":
        st.header("Explain Model Predictions")
        explanation = explain_model_predictions(data)
        st.write(explanation)

if __name__ == "__main__":
    main()