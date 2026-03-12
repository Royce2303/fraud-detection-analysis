"""
JPMorgan Fraud Detection Case Study
Author: Royce Lobo
Date: March 2026

This analysis demonstrates a comprehensive approach to detecting fraudulent transactions
in digital banking using machine learning techniques.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, f1_score
import warnings
warnings.filterwarnings('ignore')

# Set style for visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class FraudDetectionAnalysis:
    """
    A comprehensive fraud detection system for digital banking transactions.
    """
    
    def __init__(self):
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.model = None
        
    def generate_synthetic_data(self, n_samples=100000):
        """
        Generate synthetic banking transaction data for analysis.
        In a real scenario, this would be replaced with actual transaction data.
        """
        np.random.seed(42)
        
        # Generate normal transactions (95%)
        n_normal = int(n_samples * 0.95)
        n_fraud = n_samples - n_normal
        
        # Normal transactions
        normal_amount = np.random.lognormal(4, 1.5, n_normal)
        normal_time = np.random.uniform(6, 23, n_normal)  # Business hours
        normal_frequency = np.random.poisson(2, n_normal)
        normal_location_change = np.random.binomial(1, 0.1, n_normal)
        normal_device_change = np.random.binomial(1, 0.05, n_normal)
        normal_velocity = np.random.gamma(2, 2, n_normal)
        
        # Fraudulent transactions
        fraud_amount = np.random.lognormal(6, 2, n_fraud)  # Higher amounts
        fraud_time = np.random.uniform(0, 6, n_fraud)  # Odd hours
        fraud_frequency = np.random.poisson(8, n_fraud)  # Higher frequency
        fraud_location_change = np.random.binomial(1, 0.7, n_fraud)
        fraud_device_change = np.random.binomial(1, 0.6, n_fraud)
        fraud_velocity = np.random.gamma(8, 3, n_fraud)  # Higher velocity
        
        # Combine data
        data = {
            'transaction_amount': np.concatenate([normal_amount, fraud_amount]),
            'transaction_hour': np.concatenate([normal_time, fraud_time]),
            'daily_frequency': np.concatenate([normal_frequency, fraud_frequency]),
            'location_change': np.concatenate([normal_location_change, fraud_location_change]),
            'device_change': np.concatenate([normal_device_change, fraud_device_change]),
            'transaction_velocity': np.concatenate([normal_velocity, fraud_velocity]),
            'is_fraud': np.concatenate([np.zeros(n_normal), np.ones(n_fraud)])
        }
        
        self.df = pd.DataFrame(data)
        
        # Add derived features
        self.df['amount_category'] = pd.cut(self.df['transaction_amount'], 
                                            bins=[0, 50, 200, 1000, np.inf],
                                            labels=['low', 'medium', 'high', 'very_high'])
        
        self.df['time_category'] = pd.cut(self.df['transaction_hour'],
                                          bins=[0, 6, 12, 18, 24],
                                          labels=['night', 'morning', 'afternoon', 'evening'])
        
        # Shuffle the dataset
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        
        print(f"Generated {n_samples} synthetic transactions")
        print(f"Fraud rate: {self.df['is_fraud'].mean()*100:.2f}%")
        
        return self.df
    
    def exploratory_analysis(self):
        """
        Perform comprehensive exploratory data analysis.
        """
        print("\n" + "="*60)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*60)
        
        # Basic statistics
        print("\nDataset Overview:")
        print(self.df.describe())
        
        print("\nFraud Distribution:")
        print(self.df['is_fraud'].value_counts())
        print(f"\nFraud Percentage: {self.df['is_fraud'].mean()*100:.2f}%")
        
        # Create visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Fraud Detection: Exploratory Data Analysis', fontsize=16, fontweight='bold')
        
        # 1. Transaction Amount Distribution
        axes[0, 0].hist(self.df[self.df['is_fraud']==0]['transaction_amount'], 
                       bins=50, alpha=0.7, label='Normal', color='green')
        axes[0, 0].hist(self.df[self.df['is_fraud']==1]['transaction_amount'], 
                       bins=50, alpha=0.7, label='Fraud', color='red')
        axes[0, 0].set_xlabel('Transaction Amount ($)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Transaction Amount Distribution')
        axes[0, 0].legend()
        axes[0, 0].set_xlim(0, 2000)
        
        # 2. Transaction Hour Distribution
        axes[0, 1].hist(self.df[self.df['is_fraud']==0]['transaction_hour'], 
                       bins=24, alpha=0.7, label='Normal', color='green')
        axes[0, 1].hist(self.df[self.df['is_fraud']==1]['transaction_hour'], 
                       bins=24, alpha=0.7, label='Fraud', color='red')
        axes[0, 1].set_xlabel('Hour of Day')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Transaction Time Distribution')
        axes[0, 1].legend()
        
        # 3. Daily Frequency
        axes[0, 2].hist(self.df[self.df['is_fraud']==0]['daily_frequency'], 
                       bins=20, alpha=0.7, label='Normal', color='green')
        axes[0, 2].hist(self.df[self.df['is_fraud']==1]['daily_frequency'], 
                       bins=20, alpha=0.7, label='Fraud', color='red')
        axes[0, 2].set_xlabel('Daily Transaction Frequency')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Transaction Frequency Distribution')
        axes[0, 2].legend()
        
        # 4. Location Change
        location_data = self.df.groupby(['location_change', 'is_fraud']).size().unstack()
        location_data.plot(kind='bar', ax=axes[1, 0], color=['green', 'red'])
        axes[1, 0].set_xlabel('Location Change')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Location Change Impact')
        axes[1, 0].set_xticklabels(['No Change', 'Changed'], rotation=0)
        axes[1, 0].legend(['Normal', 'Fraud'])
        
        # 5. Device Change
        device_data = self.df.groupby(['device_change', 'is_fraud']).size().unstack()
        device_data.plot(kind='bar', ax=axes[1, 1], color=['green', 'red'])
        axes[1, 1].set_xlabel('Device Change')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Device Change Impact')
        axes[1, 1].set_xticklabels(['No Change', 'Changed'], rotation=0)
        axes[1, 1].legend(['Normal', 'Fraud'])
        
        # 6. Correlation Heatmap
        numeric_cols = ['transaction_amount', 'transaction_hour', 'daily_frequency', 
                       'location_change', 'device_change', 'transaction_velocity', 'is_fraud']
        corr_matrix = self.df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[1, 2])
        axes[1, 2].set_title('Feature Correlation Matrix')
        
        plt.tight_layout()
        plt.savefig('/home/claude/jpmorgan_fraud_detection/eda_analysis.png', dpi=300, bbox_inches='tight')
        print("\n✓ EDA visualization saved: eda_analysis.png")
        
        return fig
    
    def prepare_data(self):
        """
        Prepare data for machine learning model.
        """
        print("\n" + "="*60)
        print("DATA PREPARATION")
        print("="*60)
        
        # Select features
        feature_cols = ['transaction_amount', 'transaction_hour', 'daily_frequency',
                       'location_change', 'device_change', 'transaction_velocity']
        
        X = self.df[feature_cols]
        y = self.df['is_fraud']
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Scale features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"Training set size: {len(self.X_train)}")
        print(f"Test set size: {len(self.X_test)}")
        print(f"Training fraud rate: {self.y_train.mean()*100:.2f}%")
        print(f"Test fraud rate: {self.y_test.mean()*100:.2f}%")
        
    def train_model(self):
        """
        Train Random Forest model for fraud detection.
        """
        print("\n" + "="*60)
        print("MODEL TRAINING")
        print("="*60)
        
        # Train Random Forest (can be compared with your Neural Network approach from IEEE paper)
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=4,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        print("Training Random Forest model...")
        self.model.fit(self.X_train, self.y_train)
        print("✓ Model training complete")
        
        # Feature importance
        feature_names = ['transaction_amount', 'transaction_hour', 'daily_frequency',
                        'location_change', 'device_change', 'transaction_velocity']
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        print(feature_importance)
        
        return feature_importance
    
    def evaluate_model(self):
        """
        Comprehensive model evaluation with multiple metrics.
        """
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        # Predictions
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # Classification Report
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred, target_names=['Normal', 'Fraud']))
        
        # Additional Metrics
        print(f"\nROC-AUC Score: {roc_auc_score(self.y_test, y_pred_proba):.4f}")
        print(f"F1 Score: {f1_score(self.y_test, y_pred):.4f}")
        
        # Create evaluation visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Evaluation', fontsize=16, fontweight='bold')
        
        # 1. Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        axes[0, 0].set_xticklabels(['Normal', 'Fraud'])
        axes[0, 0].set_yticklabels(['Normal', 'Fraud'])
        
        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, 
                       label=f'ROC curve (AUC = {roc_auc:.3f})')
        axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0, 1].set_xlim([0.0, 1.0])
        axes[0, 1].set_ylim([0.0, 1.05])
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('Receiver Operating Characteristic (ROC) Curve')
        axes[0, 1].legend(loc="lower right")
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(self.y_test, y_pred_proba)
        axes[1, 0].plot(recall, precision, color='green', lw=2)
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Precision-Recall Curve')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Feature Importance
        feature_names = ['transaction_amount', 'transaction_hour', 'daily_frequency',
                        'location_change', 'device_change', 'transaction_velocity']
        feature_importance = pd.Series(self.model.feature_importances_, index=feature_names).sort_values()
        feature_importance.plot(kind='barh', ax=axes[1, 1], color='steelblue')
        axes[1, 1].set_title('Feature Importance')
        axes[1, 1].set_xlabel('Importance Score')
        
        plt.tight_layout()
        plt.savefig('/home/claude/jpmorgan_fraud_detection/model_evaluation.png', dpi=300, bbox_inches='tight')
        print("\n✓ Model evaluation visualization saved: model_evaluation.png")
        
        return fig
    
    def business_insights(self):
        """
        Generate business insights from the analysis.
        """
        print("\n" + "="*60)
        print("BUSINESS INSIGHTS & RECOMMENDATIONS")
        print("="*60)
        
        # Calculate key metrics
        total_transactions = len(self.df)
        fraud_transactions = self.df['is_fraud'].sum()
        fraud_rate = (fraud_transactions / total_transactions) * 100
        
        avg_fraud_amount = self.df[self.df['is_fraud']==1]['transaction_amount'].mean()
        avg_normal_amount = self.df[self.df['is_fraud']==0]['transaction_amount'].mean()
        
        # High-risk patterns
        high_risk_hours = self.df[self.df['is_fraud']==1]['transaction_hour'].mode()[0]
        
        print(f"\n📊 KEY FINDINGS:")
        print(f"   • Total Transactions Analyzed: {total_transactions:,}")
        print(f"   • Fraudulent Transactions: {int(fraud_transactions):,}")
        print(f"   • Fraud Rate: {fraud_rate:.2f}%")
        print(f"   • Average Fraud Amount: ${avg_fraud_amount:.2f}")
        print(f"   • Average Normal Amount: ${avg_normal_amount:.2f}")
        print(f"   • High-Risk Hours: {int(high_risk_hours)}:00 - {int(high_risk_hours)+1}:00")
        
        print(f"\n🎯 BUSINESS RECOMMENDATIONS:")
        print(f"   1. Enhanced monitoring during high-risk hours (late night/early morning)")
        print(f"   2. Additional verification for transactions >$500")
        print(f"   3. Alert system for location/device changes with high-value transactions")
        print(f"   4. Real-time velocity checks for rapid successive transactions")
        print(f"   5. Customer education on secure banking practices")
        
        print(f"\n💡 MODEL PERFORMANCE:")
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        print(f"   • ROC-AUC Score: {roc_auc_score(self.y_test, y_pred_proba):.4f}")
        print(f"   • F1 Score: {f1_score(self.y_test, y_pred):.4f}")
        print(f"   • Estimated Annual Fraud Prevention: ${avg_fraud_amount * fraud_transactions * 0.85:,.2f}")
        
        # Create business insights visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Business Insights Dashboard', fontsize=16, fontweight='bold')
        
        # Fraud by time of day
        fraud_by_hour = self.df[self.df['is_fraud']==1].groupby(
            pd.cut(self.df[self.df['is_fraud']==1]['transaction_hour'], bins=6)
        ).size()
        axes[0].bar(range(len(fraud_by_hour)), fraud_by_hour.values, color='crimson', alpha=0.7)
        axes[0].set_xlabel('Time Period')
        axes[0].set_ylabel('Fraud Count')
        axes[0].set_title('Fraud Distribution by Time Period')
        axes[0].grid(True, alpha=0.3)
        
        # Fraud by amount category
        fraud_by_amount = self.df.groupby(['amount_category', 'is_fraud']).size().unstack()
        fraud_by_amount.plot(kind='bar', ax=axes[1], color=['green', 'red'], alpha=0.7)
        axes[1].set_xlabel('Transaction Amount Category')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Transaction Distribution by Amount Category')
        axes[1].legend(['Normal', 'Fraud'])
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig('/home/claude/jpmorgan_fraud_detection/business_insights.png', dpi=300, bbox_inches='tight')
        print("\n✓ Business insights visualization saved: business_insights.png")
        
        return fig


def main():
    """
    Main execution function for the fraud detection analysis.
    """
    print("="*60)
    print("JPMORGAN FRAUD DETECTION CASE STUDY")
    print("Author: Royce Lobo")
    print("="*60)
    
    # Initialize analysis
    analyzer = FraudDetectionAnalysis()
    
    # Step 1: Generate data
    analyzer.generate_synthetic_data(n_samples=100000)
    
    # Step 2: Exploratory Analysis
    analyzer.exploratory_analysis()
    
    # Step 3: Prepare data
    analyzer.prepare_data()
    
    # Step 4: Train model
    analyzer.train_model()
    
    # Step 5: Evaluate model
    analyzer.evaluate_model()
    
    # Step 6: Business insights
    analyzer.business_insights()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("\nGenerated Files:")
    print("  1. eda_analysis.png - Exploratory Data Analysis")
    print("  2. model_evaluation.png - Model Performance Metrics")
    print("  3. business_insights.png - Business Recommendations")
    print("\nNext Steps:")
    print("  • Create interactive Tableau/PowerBI dashboard")
    print("  • Deploy model as API endpoint")
    print("  • Integrate with real-time transaction monitoring system")
    print("="*60)


if __name__ == "__main__":
    main()
