# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 11:47:57 2025

@author: Ruba
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    cohen_kappa_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    log_loss,
)


import warnings
warnings.filterwarnings('ignore')

class IAClusterPredictor:
    def __init__(self):
        self.pipeline = None
        self.feature_importance = None
        self.models = {
            'random_forest': RandomForestClassifier(random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'logistic_regression': LogisticRegression(penalty="l2", solver="lbfgs", max_iter=2000, multi_class='auto', random_state=42),
            'svm': SVC(random_state=42, probability=True),
            'neural_network': MLPClassifier(random_state=42, max_iter=500)
        }
        
    def load_and_preprocess_data(self, filepath):
        """Load and preprocess the data"""
        print("Loading data...")
        df = pd.read_csv(filepath)
        print(f"Data shape: {df.shape}")
        
        # Remove columns that are not features or are redundant
        columns_to_remove = ['Educational_Other', 'Employment_Other', 'Cluster_Label', 
                           'Total_IA_Score', 'Predicted_Cluster']
        df_clean = df.drop(columns=[col for col in columns_to_remove if col in df.columns])
        
        # Define feature types
        categorical_features = ['Gender', 'Age_Group', 'Marital_Status', 'Educational_Level', 
                              'Employment_Status', 'Daily_App_Usage']
        
        # Create ordinal encodings for features that have natural ordering
        age_order = ['Less than 18', '18-24 years old', '25-34 years old', '35-44 years old', 
                    '45-54 years old', '55-64 years old', '65+ years old']
        
        education_order = ['High school', 'Diploma', "Bachelor's degree", "Master's degree", 
                         "Doctorate degree", 'Other (please specify)']
        
        app_usage_order = ['Less than 1 hour', '1-2 hours', '2-3 hours', '3-4 hours', 
                          '4-5 hours', 'More than 5 hours']
        
        # Apply ordinal encoding for ordered categorical variables
        if 'Age_Group' in df_clean.columns:
            age_encoder = OrdinalEncoder(categories=[age_order], handle_unknown='use_encoded_value', 
                                       unknown_value=-1)
            df_clean['Age_Group_Ordinal'] = age_encoder.fit_transform(df_clean[['Age_Group']]).ravel()
        
        if 'Educational_Level' in df_clean.columns:
            edu_encoder = OrdinalEncoder(categories=[education_order], handle_unknown='use_encoded_value', 
                                       unknown_value=-1)
            df_clean['Educational_Level_Ordinal'] = edu_encoder.fit_transform(df_clean[['Educational_Level']]).ravel()
        
        if 'Daily_App_Usage' in df_clean.columns:
            usage_encoder = OrdinalEncoder(categories=[app_usage_order], handle_unknown='use_encoded_value', 
                                         unknown_value=-1)
            df_clean['Daily_App_Usage_Ordinal'] = usage_encoder.fit_transform(df_clean[['Daily_App_Usage']]).ravel()
        
        # Update categorical features list (remove ordinal encoded ones)
        categorical_features_nominal = ['Gender', 'Marital_Status', 'Employment_Status']
        
        # Identify numerical features
        numerical_features = [col for col in df_clean.columns 
                            if col.startswith(('IA', 'CON', 'INT', 'AVL', 'AT', 'NR', 'DL')) 
                            and col not in categorical_features]
        
        # Add ordinal encoded features to numerical features
        ordinal_features = ['Age_Group_Ordinal', 'Educational_Level_Ordinal', 'Daily_App_Usage_Ordinal']
        numerical_features.extend([col for col in ordinal_features if col in df_clean.columns])
        
        print(f"Categorical features (nominal): {categorical_features_nominal}")
        print(f"Numerical features: {len(numerical_features)} features")
        print(f"Target distribution: \n{df_clean['Cluster_Number'].value_counts().sort_index()}")
        
        return df_clean, categorical_features_nominal, numerical_features
    
    def create_preprocessing_pipeline(self, categorical_features, numerical_features):
        """Create preprocessing pipeline"""
        # Preprocessing for numerical data
        numerical_transformer = StandardScaler()
        
        # Preprocessing for categorical data (one-hot encoding)
        categorical_transformer = Pipeline(steps=[
            ('encoder', pd.get_dummies)
        ])
        
        # Bundle preprocessing
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', 'passthrough', categorical_features)  # Will handle manually
            ])
        
        return preprocessor
    
    def prepare_features(self, df, categorical_features_nominal, numerical_features):
        """Prepare features for training"""
        X_numerical = df[numerical_features]
        
        # One-hot encode nominal categorical features
        X_categorical = pd.get_dummies(df[categorical_features_nominal], drop_first=True)
        
        # Combine features
        X = pd.concat([X_numerical, X_categorical], axis=1)
        y = df['Cluster_Number']
        
        print(f"Final feature matrix shape: {X.shape}")
        print(f"Feature columns: {list(X.columns)}")
        
        return X, y
    
    def train_and_evaluate_models(self, X, y):
        """Train and evaluate multiple models"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                          random_state=42, stratify=y)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        results = {}
        
        print("Training and evaluating models...")
        print("="*50)
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            train_pred = model.predict(X_train_scaled)
            test_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            train_accuracy = accuracy_score(y_train, train_pred)
            test_accuracy = accuracy_score(y_test, test_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            results[name] = {
                'model': model,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'test_predictions': test_pred
            }
            
            print(f"  Train Accuracy: {train_accuracy:.4f}")
            print(f"  Test Accuracy: {test_accuracy:.4f}")
            print(f"  CV Score: {cv_mean:.4f} (+/- {cv_std*2:.4f})")
            # Add Accuracy and AUC (macro OvR for multi-class)
            acc = accuracy_score(y_test, test_pred)
            auc = roc_auc_score(y_test, test_proba, multi_class="ovr", average="macro")
            
            print(f"Accuracy: {acc:.4f}")
            print(f"ROC-AUC (macro OvR): {auc:.4f}")
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['cv_mean'])
        best_model = results[best_model_name]['model']
        
        print(f"\n" + "="*50)
        print(f"Best Model: {best_model_name}")
        print(f"Best CV Score: {results[best_model_name]['cv_mean']:.4f}")
        
        # Detailed evaluation of best model
        print(f"\nDetailed evaluation for {best_model_name}:")
        print(classification_report(y_test, results[best_model_name]['test_predictions']))
        
        return results, best_model, scaler, X_test, y_test
    

    def hyperparameter_tuning(self, X, y, model_name='logistic_regression'):
        """
        Hyperparameter tuning with full metric readout for Logistic Regression.
        Uses RepeatedStratifiedKFold and scoring='f1_macro' for multi-class balance.
        """
        print(f"\nPerforming hyperparameter tuning for {model_name}...")
    
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)
    
        if model_name == 'logistic_regression':
            # Option 1: expanded LR grid
            param_grid = [
                {
                    'solver': ['lbfgs'],
                    'multi_class': ['auto', 'multinomial'],
                    'penalty': ['l2'],
                    'C': [0.05, 0.1, 0.2, 0.3162, 0.5, 1, 1.5, 2, 3],
                    'max_iter': [2000],
                    'class_weight': [None, 'balanced']
                },
                {
                    'solver': ['saga'],
                    'multi_class': ['auto', 'multinomial'],
                    'penalty': ['l1', 'l2', 'elasticnet'],
                    'l1_ratio': [0, 0.25, 0.5, 0.75, 1],  # used for elasticnet
                    'C': [0.05, 0.1, 0.2, 0.3162, 0.5, 1, 1.5, 2, 3],
                    'max_iter': [3000],
                    'class_weight': [None, 'balanced']
                }
            ]
            base_model = LogisticRegression(random_state=42)
            cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                scoring='f1_macro',   # use 'accuracy' if you prefer
                cv=cv,
                n_jobs=-1,
                refit=True,
                verbose=1
            )
        else:
            raise ValueError("This function is currently set up to tune 'logistic_regression' only.")
    
        # Fit grid
        grid_search.fit(X_train_scaled, y_train)
    
        # Best model & predictions
        best_model = grid_search.best_estimator_
        test_pred  = best_model.predict(X_test_scaled)
    
        # Safe proba & AUC
        if hasattr(best_model, "predict_proba"):
            test_proba = best_model.predict_proba(X_test_scaled)
            auc_macro = roc_auc_score(y_test, test_proba, average="macro", multi_class="ovr")
            ll = log_loss(y_test, test_proba)
        else:
            test_proba = None
            auc_macro = float('nan')
            ll = float('nan')
    
        # Core metrics
        acc = accuracy_score(y_test, test_pred)
        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
            y_test, test_pred, average='macro', zero_division=0
        )
        bal_acc = balanced_accuracy_score(y_test, test_pred)
        kappa   = cohen_kappa_score(y_test, test_pred)
        mcc     = matthews_corrcoef(y_test, test_pred)
        cm      = confusion_matrix(y_test, test_pred)
    
        # Prints
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV (f1_macro): {grid_search.best_score_:.4f}")
        print("\nDetailed classification report (tuned model):")
        print(classification_report(y_test, test_pred))
        print(f"Accuracy: {acc:.4f}")
        print(f"ROC-AUC (macro OvR): {auc_macro:.4f}")
        print(f"Balanced Accuracy: {bal_acc:.4f}")
        print(f"Cohen's Kappa: {kappa:.4f}")
        print(f"MCC: {mcc:.4f}")
        print(f"Log Loss: {ll:.4f}")
        print("Confusion Matrix:\n", cm)
    
        # Return everything you may want to reuse
        metrics_dict = {
            "best_params": grid_search.best_params_,
            "cv_f1_macro": grid_search.best_score_,
            "accuracy": acc,
            "precision_macro": prec_macro,
            "recall_macro": rec_macro,
            "f1_macro": f1_macro,
            "auc_macro_ovr": auc_macro,
            "balanced_accuracy": bal_acc,
            "cohen_kappa": kappa,
            "mcc": mcc,
            "log_loss": ll,
            "confusion_matrix": cm.tolist()
        }
        return best_model, scaler, metrics_dict
    




# Usage Example
if __name__ == "__main__":
    # Initialize predictor
    predictor = IAClusterPredictor()
    
    # Load and preprocess data
    df, categorical_features, numerical_features = predictor.load_and_preprocess_data(
        'Information_Assurance_Clustered.csv'
    )
    
    # Prepare features
    X, y = predictor.prepare_features(df, categorical_features, numerical_features)
    
    # Train and evaluate models
    results, best_model, scaler, X_test, y_test = predictor.train_and_evaluate_models(X, y)
    
    # Hyperparameter tuning for the best performing model
    tuned_model, tuned_scaler, best_params = predictor.hyperparameter_tuning(X, y, 'logistic_regression')
    
    # Feature importance
    feature_importance = predictor.get_feature_importance(tuned_model, X.columns, 'tree_based')
    
    # Save the final model
    predictor.save_model(tuned_model, tuned_scaler, 'ia_cluster_predictor')
    
    print("\n" + "="*60)
    print("SUMMARY:")
    print("="*60)
    print("1. Data preprocessed with ordinal encoding for ordered categorical variables")
    print("2. Multiple models trained and evaluated")
    print("3. Best model hyperparameters tuned")
    print("4. Feature importance analysis completed")
    print("5. Model saved for future predictions")
    
    # Example of making new predictions
    print("\nTo make predictions on new data:")
    print("1. Load the saved model: model = joblib.load('ia_cluster_predictor_model.pkl')")
    print("2. Load the scaler: scaler = joblib.load('ia_cluster_predictor_scaler.pkl')")
    print("3. Preprocess new data in the same way as training data")
    print("4. Use: predictions, probabilities = predictor.predict_new_data(model, scaler, new_data)")
    
    
'''
Loading data...
Data shape: (924, 45)
Categorical features (nominal): ['Gender', 'Marital_Status', 'Employment_Status']
Numerical features: 36 features
Target distribution: 
Cluster_Number
0    191
1    476
2    257
Name: count, dtype: int64
Final feature matrix shape: (924, 47)
Feature columns: ['IA1', 'IA2', 'IA3', 'IA4', 'IA5', 'CON1', 'CON2', 'CON3', 'CON4', 'INT1', 'INT2', 'INT3', 'AVL1', 'AVL2', 'AVL3', 'AT1', 'AT2', 'AT3', 'AT4', 'NR1', 'NR2', 'NR3', 'NR4', 'DL1', 'DL2', 'DL3', 'DL4', 'DL5', 'DL6', 'DL7', 'DL8', 'DL9', 'DL10', 'Age_Group_Ordinal', 'Educational_Level_Ordinal', 'Daily_App_Usage_Ordinal', 'Gender_Male', 'Marital_Status_Married', 'Marital_Status_Separated', 'Marital_Status_Single', 'Marital_Status_Widowed', 'Employment_Status_Employed in a public sector', 'Employment_Status_Other (please specify)', 'Employment_Status_Retired', 'Employment_Status_Self-employed(free lancer)', 'Employment_Status_Student', 'Employment_Status_Unemployed']
Training and evaluating models...
==================================================

Training random_forest...
  Train Accuracy: 1.0000
  Test Accuracy: 0.9135
  CV Score: 0.9310 (+/- 0.0264)

Training gradient_boosting...
  Train Accuracy: 1.0000
  Test Accuracy: 0.9405
  CV Score: 0.9012 (+/- 0.0478)

Training logistic_regression...
  Train Accuracy: 0.9973
  Test Accuracy: 0.9568
  CV Score: 0.9553 (+/- 0.0220)

Training svm...
  Train Accuracy: 0.9959
  Test Accuracy: 0.9297
  CV Score: 0.9486 (+/- 0.0252)

Training neural_network...
  Train Accuracy: 1.0000
  Test Accuracy: 0.9405
  CV Score: 0.9513 (+/- 0.0131)

==================================================
Best Model: logistic_regression
Best CV Score: 0.9553

Detailed evaluation for logistic_regression:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        38
           1       0.95      0.97      0.96        95
           2       0.94      0.90      0.92        52

    accuracy                           0.96       185
   macro avg       0.96      0.96      0.96       185
weighted avg       0.96      0.96      0.96       185


Performing hyperparameter tuning for random_forest...
Best parameters: {'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
Best CV score: 0.9337
Test accuracy: 0.9081

Top 15 Most Important Features:
   feature  importance
11    INT3    0.070215
15     AT1    0.065896
9     INT1    0.062805
19     NR1    0.060209
10    INT2    0.046496
18     AT4    0.046020
21     NR3    0.043694
7     CON3    0.042205
0      IA1    0.041725
13    AVL2    0.040896
20     NR2    0.038836
16     AT2    0.036509
12    AVL1    0.035658
4      IA5    0.034840
14    AVL3    0.034674
Model saved as ia_cluster_predictor_model.pkl
Scaler saved as ia_cluster_predictor_scaler.pkl

============================================================
SUMMARY:
============================================================
1. Data preprocessed with ordinal encoding for ordered categorical variables
2. Multiple models trained and evaluated
3. Best model hyperparameters tuned
4. Feature importance analysis completed
5. Model saved for future predictions

To make predictions on new data:
1. Load the saved model: model = joblib.load('ia_cluster_predictor_model.pkl')
2. Load the scaler: scaler = joblib.load('ia_cluster_predictor_scaler.pkl')
3. Preprocess new data in the same way as training data
4. Use: predictions, probabilities = predictor.predict_new_data(model, scaler, new_data)
'''

