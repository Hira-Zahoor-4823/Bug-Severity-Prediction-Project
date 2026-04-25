# DS4SE Project: Milestone 2: Feature Selection
# Group Leader: Hira Zahoor (22K-4823)
# Members:   Hafsa Salman (22K-5161)
#            Jaysha Iqbal (22K-5178)
#            Amna Mansoor (22K-5159)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif, chi2, SelectKBest, f_classif
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

class FeatureSelection:
    def __init__(self, data_path):
        """Initialize feature selection"""
        self.df = pd.read_csv(data_path)
        self.X = None
        self.y = None
        self.feature_names = None
        self.le_dict = {}
        
    def preprocess_data(self):
        """Prepare data for feature selection"""
        print("\n\nDATA PREPROCESSING FOR FEATURE SELECTION")
        
        df_processed = self.df.copy()
        
        # Target variable encoding
        severity_map = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
        self.y = df_processed['severity'].map(severity_map)
        
        # Select features for analysis
        categorical_features = ['app_name', 'module', 'bug_type', 'priority', 
                               'device_type', 'browser', 'os', 'reported_by']
        numeric_features = ['time_to_detect_sec', 'time_to_fix_sec', 'user_impact_score']
        boolean_features = ['screenshot_available', 'resolved']
        
        # Encode categorical features
        X_cat = pd.DataFrame()
        for col in categorical_features:
            le = LabelEncoder()
            X_cat[col] = le.fit_transform(df_processed[col])
            self.le_dict[col] = le
        
        # Encode boolean features
        X_bool = pd.DataFrame()
        for col in boolean_features:
            X_bool[col] = df_processed[col].astype(int)
        
        # Combine all features
        self.X = pd.concat([X_cat, X_bool, df_processed[numeric_features]], axis=1)
        self.feature_names = self.X.columns.tolist()
        
        print(f"Total Features: {len(self.feature_names)}")
        print(f"Features: {self.feature_names}")
        print(f"Target Variable (Severity): {len(self.y)} samples")
        
        return self.X, self.y
    
    def random_forest_importance(self):
        """Calculate feature importance using Random Forest"""
        print("\nRANDOM FOREST FEATURE IMPORTANCE")
        
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        
        rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print(f"\nModel Accuracy: {rf.score(X_test, y_test):.4f}")
        print("\nFeature Importance Ranking:")
        print("-" * 50)
        
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        for idx, (_, row) in enumerate(importance_df.iterrows(), 1):
            print(f"{idx:2d}. {row['Feature']:25s} : {row['Importance']:.6f}")
        
        return importance_df, rf, X_test, y_test
    
    def gradient_boosting_importance(self):
        """Calculate feature importance using Gradient Boosting"""
        print("\nGRADIENT BOOSTING FEATURE IMPORTANCE")
        
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        
        gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        gb.fit(X_train, y_train)
        
        importances = gb.feature_importances_
        
        print(f"\nModel Accuracy: {gb.score(X_test, y_test):.4f}")
        print("\nFeature Importance Ranking:")
        print("-" * 50)
        
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        for idx, (_, row) in enumerate(importance_df.iterrows(), 1):
            print(f"{idx:2d}. {row['Feature']:25s} : {row['Importance']:.6f}")
        
        return importance_df, gb
    
    def mutual_information_importance(self):
        """Calculate feature importance using Mutual Information"""
        print("\nMUTUAL INFORMATION FEATURE IMPORTANCE")
        
        mi_scores = mutual_info_classif(self.X, self.y, random_state=42)
        
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': mi_scores
        }).sort_values('Importance', ascending=False)
        
        print("\nMutual Information Scores:")
        print("-" * 50)
        for idx, (_, row) in enumerate(importance_df.iterrows(), 1):
            print(f"{idx:2d}. {row['Feature']:25s} : {row['Importance']:.6f}")
        
        return importance_df
    
    def statistical_importance(self):
        """Calculate feature importance using statistical tests"""
        print("\nSTATISTICAL FEATURE IMPORTANCE (F-Score)")
        
        f_scores = f_classif(self.X, self.y)[0]
        
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': f_scores
        }).sort_values('Importance', ascending=False)
        
        print("\nF-Scores:")
        print("-" * 50)
        for idx, (_, row) in enumerate(importance_df.iterrows(), 1):
            print(f"{idx:2d}. {row['Feature']:25s} : {row['Importance']:.6f}")
        
        return importance_df
    
    def select_top_features(self, n_features=10):
        """Select top N features based on consensus"""
        print(f"\nTOP {n_features} FEATURES (CONSENSUS RANKING)")
        
        # Get importance from all methods
        rf_imp, _, _, _ = self.random_forest_importance()
        gb_imp, _ = self.gradient_boosting_importance()
        mi_imp = self.mutual_information_importance()
        f_imp = self.statistical_importance()
        
        # Normalize importances to 0-1 range
        rf_norm = (rf_imp['Importance'] - rf_imp['Importance'].min()) / (rf_imp['Importance'].max() - rf_imp['Importance'].min())
        gb_norm = (gb_imp['Importance'] - gb_imp['Importance'].min()) / (gb_imp['Importance'].max() - gb_imp['Importance'].min())
        mi_norm = (mi_imp['Importance'] - mi_imp['Importance'].min()) / (mi_imp['Importance'].max() - mi_imp['Importance'].min())
        f_norm = (f_imp['Importance'] - f_imp['Importance'].min()) / (f_imp['Importance'].max() - f_imp['Importance'].min())
        
        # Calculate consensus score
        consensus_df = pd.DataFrame({
            'Feature': self.feature_names,
            'RF_Score': rf_norm.values,
            'GB_Score': gb_norm.values,
            'MI_Score': mi_norm.values,
            'F_Score': f_norm.values
        })
        
        consensus_df['Consensus_Score'] = consensus_df[['RF_Score', 'GB_Score', 'MI_Score', 'F_Score']].mean(axis=1)
        consensus_df = consensus_df.sort_values('Consensus_Score', ascending=False)
        
        print("\nConsensus Ranking:")
        print("-" * 80)
        print(f"{'Rank':<5} {'Feature':<25} {'RF':<8} {'GB':<8} {'MI':<8} {'F-Score':<8} {'Consensus':<10}")
        print("-" * 80)
        
        for idx, (_, row) in enumerate(consensus_df.head(n_features).iterrows(), 1):
            print(f"{idx:<5} {row['Feature']:<25} {row['RF_Score']:<8.4f} {row['GB_Score']:<8.4f} "
                  f"{row['MI_Score']:<8.4f} {row['F_Score']:<8.4f} {row['Consensus_Score']:<10.4f}")
        
        top_features = consensus_df.head(n_features)['Feature'].tolist()
        
        print(f"\nTop {n_features} Selected Features:")
        for idx, feature in enumerate(top_features, 1):
            print(f"  {idx}. {feature}")
        
        return consensus_df, top_features
    
    def generate_importance_visualizations(self, output_dir='plots'):
        """Generate feature importance visualizations"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Get importance scores
        rf_imp, _, _, _ = self.random_forest_importance()
        gb_imp, _ = self.gradient_boosting_importance()
        mi_imp = self.mutual_information_importance()
        f_imp = self.statistical_importance()
        consensus_df, _ = self.select_top_features(10)
        
        # 1. Random Forest Importance
        fig, ax = plt.subplots(figsize=(12, 8))
        top_rf = rf_imp.head(15)
        ax.barh(range(len(top_rf)), top_rf['Importance'].values, color='steelblue')
        ax.set_yticks(range(len(top_rf)))
        ax.set_yticklabels(top_rf['Feature'].values)
        ax.set_xlabel('Importance Score')
        ax.set_title('Top 15 Features - Random Forest Importance', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'{output_dir}/09_rf_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Gradient Boosting Importance
        fig, ax = plt.subplots(figsize=(12, 8))
        top_gb = gb_imp.head(15)
        ax.barh(range(len(top_gb)), top_gb['Importance'].values, color='coral')
        ax.set_yticks(range(len(top_gb)))
        ax.set_yticklabels(top_gb['Feature'].values)
        ax.set_xlabel('Importance Score')
        ax.set_title('Top 15 Features - Gradient Boosting Importance', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'{output_dir}/10_gb_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Mutual Information Importance
        fig, ax = plt.subplots(figsize=(12, 8))
        top_mi = mi_imp.head(15)
        ax.barh(range(len(top_mi)), top_mi['Importance'].values, color='seagreen')
        ax.set_yticks(range(len(top_mi)))
        ax.set_yticklabels(top_mi['Feature'].values)
        ax.set_xlabel('Importance Score')
        ax.set_title('Top 15 Features - Mutual Information', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'{output_dir}/11_mi_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Consensus Importance
        fig, ax = plt.subplots(figsize=(12, 8))
        top_consensus = consensus_df.head(15)
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_consensus)))
        ax.barh(range(len(top_consensus)), top_consensus['Consensus_Score'].values, color=colors)
        ax.set_yticks(range(len(top_consensus)))
        ax.set_yticklabels(top_consensus['Feature'].values)
        ax.set_xlabel('Consensus Score')
        ax.set_title('Top 15 Features - Consensus Ranking', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'{output_dir}/12_consensus_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Comparison of all methods (top 10)
        fig, ax = plt.subplots(figsize=(14, 8))
        top_features = consensus_df.head(10)['Feature'].tolist()
        x = np.arange(len(top_features))
        width = 0.2
        
        # Normalize scores for comparison
        rf_scores = [rf_imp[rf_imp['Feature'] == f]['Importance'].values[0] for f in top_features]
        gb_scores = [gb_imp[gb_imp['Feature'] == f]['Importance'].values[0] for f in top_features]
        mi_scores = [mi_imp[mi_imp['Feature'] == f]['Importance'].values[0] for f in top_features]
        f_scores = [f_imp[f_imp['Feature'] == f]['Importance'].values[0] for f in top_features]
        
        # Normalize
        rf_norm = (np.array(rf_scores) - min(rf_scores)) / (max(rf_scores) - min(rf_scores))
        gb_norm = (np.array(gb_scores) - min(gb_scores)) / (max(gb_scores) - min(gb_scores))
        mi_norm = (np.array(mi_scores) - min(mi_scores)) / (max(mi_scores) - min(mi_scores))
        f_norm = (np.array(f_scores) - min(f_scores)) / (max(f_scores) - min(f_scores))
        
        ax.bar(x - 1.5*width, rf_norm, width, label='Random Forest', color='steelblue')
        ax.bar(x - 0.5*width, gb_norm, width, label='Gradient Boosting', color='coral')
        ax.bar(x + 0.5*width, mi_norm, width, label='Mutual Information', color='seagreen')
        ax.bar(x + 1.5*width, f_norm, width, label='F-Score', color='gold')
        
        ax.set_xlabel('Features')
        ax.set_ylabel('Normalized Importance Score')
        ax.set_title('Feature Importance Comparison - Top 10 Features', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(top_features, rotation=45, ha='right')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f'{output_dir}/13_importance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nImportance visualizations saved to {output_dir}/")
    
    def generate_feature_report(self, output_file='feature_selection_report.txt'):
        """Generate comprehensive feature selection report"""
        with open(output_file, 'w') as f:
            f.write("\nFEATURE SELECTION AND IMPORTANCE ANALYSIS REPORT\n")
            
            # Overview
            f.write("ANALYSIS OVERVIEW:\n")
            f.write(f"Total Features Analyzed: {len(self.feature_names)}\n")
            f.write(f"Total Samples: {len(self.X)}\n")
            f.write(f"Target Classes: 4 (Low, Medium, High, Critical)\n\n")
            
            # Methods used
            f.write("FEATURE SELECTION METHODS USED:\n")
            f.write("1. Random Forest Classifier - Gini-based importance\n")
            f.write("2. Gradient Boosting Classifier - Information gain\n")
            f.write("3. Mutual Information - Information-theoretic approach\n")
            f.write("4. F-Score - Statistical significance\n\n")
            
            # Consensus ranking
            f.write("TOP 10 RECOMMENDED FEATURES (CONSENSUS RANKING):\n")
            consensus_df, top_features = self.select_top_features(10)
            for idx, feature in enumerate(top_features, 1):
                f.write(f"{idx}. {feature}\n")
            
            f.write("\nRECOMMENDATIONS:\n")
            f.write("1. Focus on the top 10 features for model building\n")
            f.write("2. Remove low-importance features to reduce dimensionality\n")
            f.write("3. Numeric features (time, impact) are crucial predictors\n")
            f.write("4. Consider feature interactions for enhanced performance\n")
        
        print(f"Feature selection report saved to {output_file}")

def main():
    # Initialize and run feature selection
    data_path = '../frontend_uiux_bug_dataset_cleaned.csv'
    fs = FeatureSelection(data_path)
    
    # Preprocess data
    fs.preprocess_data()
    
    # Run all feature importance methods
    fs.random_forest_importance()
    fs.gradient_boosting_importance()
    fs.mutual_information_importance()
    fs.statistical_importance()
    
    # Select top features
    consensus_df, top_features = fs.select_top_features(10)
    
    # Generate visualizations
    fs.generate_importance_visualizations('plots')
    
    # Generate report
    fs.generate_feature_report('feature_selection_report.txt')
    
    print("\nFEATURE SELECTION ANALYSIS COMPLETE")

if __name__ == "__main__":
    main()
