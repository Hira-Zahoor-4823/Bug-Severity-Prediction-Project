# DS4SE Project: Milestone 2: Analysis Scripts
# Group Leader: Hira Zahoor (22K-4823)
# Members:   Hafsa Salman (22K-5161)
#            Jaysha Iqbal (22K-5178)
#            Amna Mansoor (22K-5159)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

class StatisticalInsights:
    def __init__(self, data_path):
        """Initialize with dataset"""
        self.df = pd.read_csv(data_path)
        
        self.df['resolved'] = self.df['resolved'].map(
            {'True': True, 'False': False, True: True, False: False}
        )
        self.df['screenshot_available'] = self.df['screenshot_available'].map(
            {'True': True, 'False': False, True: True, False: False}
        )
        
        if 'bug_id' in self.df.columns:
            self.df = self.df.drop(columns=['bug_id'])
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        self.numeric_cols = [col for col in numeric_cols if col != 'bug_id']
        
        text_cols = ['steps_to_reproduce', 'expected_behavior', 'actual_behavior', 'tags', 'page_url']
        self.categorical_cols = [c for c in self.df.select_dtypes(include=['object']).columns if c not in text_cols]
        
    def basic_statistics(self):
        """Generate basic descriptive statistics"""
        print("BASIC STATISTICAL SUMMARY")
        print(f"\nDataset Shape: {self.df.shape}")
        print(f"Total Records: {len(self.df)}")
        print(f"Total Features: {len(self.df.columns)}")
        
        print("\nNUMERIC FEATURES STATISTICS:")
        print(self.df[self.numeric_cols].describe().round(2))
        
        print("\nMISSING VALUES:")
        missing = self.df.isnull().sum()
        if missing.sum() == 0:
            print("No missing values found!")
        else:
            print(missing[missing > 0])
        
        print("\nDATA TYPES:")
        print(self.df.dtypes)
        
    def categorical_analysis(self):
        """Analyze categorical features"""
        print("\nCATEGORICAL FEATURES ANALYSIS")
        
        for col in self.categorical_cols:
            print(f"\n{col}:")
            print(f"  Unique Values: {self.df[col].nunique()}")
            print(f"  Most Common: {self.df[col].mode()[0]} ({self.df[col].value_counts().iloc[0]} occurrences)")
            print(f"  Distribution:\n{self.df[col].value_counts()}")
    
    def severity_distribution(self):
        """Analyze severity distribution"""
        print("\nSEVERITY DISTRIBUTION ANALYSIS")
        
        severity_counts = self.df['severity'].value_counts()
        severity_pct = self.df['severity'].value_counts(normalize=True) * 100
        
        print("\nSeverity Count:")
        for severity, count in severity_counts.items():
            pct = severity_pct[severity]
            print(f"  {severity}: {count} ({pct:.2f}%)")
        
        # Severity by priority
        print("\nSeverity by Priority:")
        print(pd.crosstab(self.df['priority'], self.df['severity']))
        
        # Severity by device type
        print("\nSeverity by Device Type:")
        print(pd.crosstab(self.df['device_type'], self.df['severity']))
        
    def correlation_analysis(self):
        """Calculate and display correlations"""
        print("\nCORRELATION ANALYSIS")
        
        severity_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}
        df_corr = self.df.copy()
        df_corr['severity_numeric'] = df_corr['severity'].map(severity_map)
        df_corr['screenshot_available_numeric'] = df_corr['screenshot_available'].astype(int)
        df_corr['resolved_numeric'] = df_corr['resolved'].astype(int)
        
        corr_cols = ['time_to_detect_sec', 'time_to_fix_sec', 'user_impact_score', 
                     'severity_numeric', 'screenshot_available_numeric']
        correlation_matrix = df_corr[corr_cols].corr()
        
        print("\nCorrelation Matrix:")
        print(correlation_matrix.round(3))
        
        print("\nFeature Correlations with Severity:")
        severity_corr = correlation_matrix['severity_numeric'].sort_values(ascending=False)
        for feature, corr in severity_corr.items():
            if feature != 'severity_numeric':
                print(f"  {feature}: {corr:.4f}")
        
        return correlation_matrix, df_corr
    
    def time_analysis(self):
        """Analyze time-related metrics"""
        print("\nTIME METRICS ANALYSIS")
        
        print(f"\nTime to Detect (seconds):")
        print(f"  Mean: {self.df['time_to_detect_sec'].mean():.2f}")
        print(f"  Median: {self.df['time_to_detect_sec'].median():.2f}")
        print(f"  Std Dev: {self.df['time_to_detect_sec'].std():.2f}")
        print(f"  Min: {self.df['time_to_detect_sec'].min()}")
        print(f"  Max: {self.df['time_to_detect_sec'].max()}")
        
        print(f"\nTime to Fix (seconds):")
        print(f"  Mean: {self.df['time_to_fix_sec'].mean():.2f}")
        print(f"  Median: {self.df['time_to_fix_sec'].median():.2f}")
        print(f"  Std Dev: {self.df['time_to_fix_sec'].std():.2f}")
        print(f"  Min: {self.df['time_to_fix_sec'].min()}")
        print(f"  Max: {self.df['time_to_fix_sec'].max()}")
        
        correlation = self.df['time_to_detect_sec'].corr(self.df['time_to_fix_sec'])
        print(f"\nCorrelation between Time to Detect and Time to Fix: {correlation:.4f}")
        
        print("\nAverage Time to Fix by Severity:")
        print(self.df.groupby('severity')['time_to_fix_sec'].agg(['mean', 'median', 'std']).round(2))
    
    def user_impact_analysis(self):
        """Analyze user impact scores"""
        print("\nUSER IMPACT ANALYSIS")
        
        print(f"\nUser Impact Score Statistics:")
        print(f"  Mean: {self.df['user_impact_score'].mean():.2f}")
        print(f"  Median: {self.df['user_impact_score'].median():.2f}")
        print(f"  Std Dev: {self.df['user_impact_score'].std():.2f}")
        print(f"  Min: {self.df['user_impact_score'].min()}")
        print(f"  Max: {self.df['user_impact_score'].max()}")
        
        print("\nUser Impact Score by Severity:")
        print(self.df.groupby('severity')['user_impact_score'].agg(['mean', 'median', 'count']).round(2))
        
        print("\nUser Impact Score by Bug Type:")
        print(self.df.groupby('bug_type')['user_impact_score'].agg(['mean', 'count']).round(2))
    
    def resolution_analysis(self):
        """Analyze bug resolution patterns"""
        print("\nRESOLUTION ANALYSIS")
        
        resolved_count = self.df['resolved'].value_counts()
        resolved_pct = self.df['resolved'].value_counts(normalize=True) * 100
        
        print(f"\nResolution Status:")
        print(f"  Resolved: {resolved_count[True]} ({resolved_pct[True]:.2f}%)")
        print(f"  Unresolved: {resolved_count[False]} ({resolved_pct[False]:.2f}%)")
        
        print("\nResolution Rate by Severity:")
        resolution_by_severity = pd.crosstab(self.df['severity'], self.df['resolved'], normalize='index') * 100
        print(resolution_by_severity.round(2))
        
        print("\nResolution Rate by Priority:")
        resolution_by_priority = pd.crosstab(self.df['priority'], self.df['resolved'], normalize='index') * 100
        print(resolution_by_priority.round(2))
    
    def bug_type_analysis(self):
        """Analyze bug type distribution"""
        print("\nBUG TYPE ANALYSIS")
        
        bug_types = self.df['bug_type'].value_counts()
        print(f"\nBug Type Distribution:")
        for bug_type, count in bug_types.items():
            pct = (count / len(self.df)) * 100
            print(f"  {bug_type}: {count} ({pct:.2f}%)")
        
        print("\nAverage Time to Fix by Bug Type:")
        print(self.df.groupby('bug_type')['time_to_fix_sec'].mean().sort_values(ascending=False).round(2))
        
        print("\nSeverity Distribution by Bug Type:")
        print(pd.crosstab(self.df['bug_type'], self.df['severity']))
    
    def platform_analysis(self):
        """Analyze platform-related features"""
        print("\nPLATFORM ANALYSIS")
        
        print("\nDevice Type Distribution:")
        print(self.df['device_type'].value_counts())
        
        print("\nBrowser Distribution:")
        print(self.df['browser'].value_counts())
        
        print("\nOS Distribution:")
        print(self.df['os'].value_counts())
        
        print("\nSeverity by OS:")
        print(pd.crosstab(self.df['os'], self.df['severity']))
        
        print("\nAverage Time to Fix by Device Type:")
        print(self.df.groupby('device_type')['time_to_fix_sec'].mean().round(2))
    
    def hypothesis_testing(self):
        """Run statistical hypothesis tests"""
        print("\n")
        print("HYPOTHESIS TESTING - STATISTICAL SIGNIFICANCE ANALYSIS")
        
        from scipy import stats
        
        print("\n1. ANOVA - Time to Fix across Severity Levels")
        groups = [
            self.df[self.df['severity'] == s]['time_to_fix_sec'].values
            for s in ['Low', 'Medium', 'High', 'Critical']
        ]
        f_stat, p_value = stats.f_oneway(*groups)
        print(f"   F-statistic : {f_stat:.4f}")
        print(f"   p-value     : {p_value:.6f}")
        print(f"   Result      : {'*** SIGNIFICANT ***' if p_value < 0.05 else 'Not significant'} (α=0.05)")
        
        print("\n2. Chi-Square - Bug Type vs Severity Association")
        contingency = pd.crosstab(self.df['bug_type'], self.df['severity'])
        chi2, p, dof, expected = stats.chi2_contingency(contingency)
        print(f"   Chi2 statistic : {chi2:.4f}")
        print(f"   p-value        : {p:.6f}")
        print(f"   Degrees of freedom: {dof}")
        print(f"   Result         : {'*** SIGNIFICANT ***' if p < 0.05 else 'Not significant'} (α=0.05)")
        
        print("\n3. Chi-Square - Device Type vs Severity Association")
        contingency2 = pd.crosstab(self.df['device_type'], self.df['severity'])
        chi2_2, p2, dof2, _ = stats.chi2_contingency(contingency2)
        print(f"   Chi2 statistic : {chi2_2:.4f}")
        print(f"   p-value        : {p2:.6f}")
        print(f"   Degrees of freedom: {dof2}")
        print(f"   Result         : {'*** SIGNIFICANT ***' if p2 < 0.05 else 'Not significant'} (α=0.05)")
        
        print("\n4. Kruskal-Wallis - Time to Detect across Severity Levels")
        groups_detect = [
            self.df[self.df['severity'] == s]['time_to_detect_sec'].values
            for s in ['Low', 'Medium', 'High', 'Critical']
        ]
        h_stat, p_kw = stats.kruskal(*groups_detect)
        print(f"   H-statistic : {h_stat:.4f}")
        print(f"   p-value     : {p_kw:.6f}")
        print(f"   Result      : {'*** SIGNIFICANT ***' if p_kw < 0.05 else 'Not significant'} (α=0.05)")
        
    
    def generate_visualizations(self, output_dir='plots'):
        """Generate statistical visualizations"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        severity_counts = self.df['severity'].value_counts()
        axes[0].bar(severity_counts.index, severity_counts.values, color=['#ff6b6b', '#ffd93d', '#ff9ff3', '#ff1493'])
        axes[0].set_title('Severity Distribution', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Count')
        
        axes[1].pie(severity_counts.values, labels=severity_counts.index, autopct='%1.1f%%',
                   colors=['#ff6b6b', '#ffd93d', '#ff9ff3', '#ff1493'])
        axes[1].set_title('Severity Proportion', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/11_severity_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(self.df['time_to_detect_sec'], self.df['time_to_fix_sec'], 
                           c=self.df['user_impact_score'], cmap='viridis', alpha=0.6, s=50)
        ax.set_xlabel('Time to Detect (seconds)', fontsize=11)
        ax.set_ylabel('Time to Fix (seconds)', fontsize=11)
        ax.set_title('Time to Detect vs Time to Fix (colored by User Impact)', fontsize=14, fontweight='bold')
        cbar = plt.colorbar(scatter)
        cbar.set_label('User Impact Score')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/12_time_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        self.df.boxplot(column='user_impact_score', by='severity', ax=ax)
        ax.set_title('User Impact Score Distribution by Severity', fontsize=14, fontweight='bold')
        ax.set_xlabel('Severity')
        ax.set_ylabel('User Impact Score')
        plt.suptitle('')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/13_impact_by_severity.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bug_types = self.df['bug_type'].value_counts()
        ax.barh(bug_types.index, bug_types.values, color='steelblue')
        ax.set_xlabel('Count')
        ax.set_title('Bug Type Distribution', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/14_bug_type_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        resolution_data = pd.crosstab(self.df['severity'], self.df['resolved'])
        resolution_data.plot(kind='bar', ax=ax, color=['#ff6b6b', '#51cf66'])
        ax.set_title('Resolution Status by Severity', fontsize=14, fontweight='bold')
        ax.set_xlabel('Severity')
        ax.set_ylabel('Count')
        ax.legend(['Unresolved', 'Resolved'], loc='upper right')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/15_resolution_by_severity.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        device_severity = pd.crosstab(self.df['device_type'], self.df['severity'])
        device_severity.plot(kind='bar', ax=ax)
        ax.set_title('Severity Distribution by Device Type', fontsize=14, fontweight='bold')
        ax.set_xlabel('Device Type')
        ax.set_ylabel('Count')
        plt.xticks(rotation=45)
        plt.legend(title='Severity')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/16_device_severity.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        severity_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}
        df_corr = self.df.copy()
        df_corr['severity_numeric'] = df_corr['severity'].map(severity_map)
        df_corr['screenshot_available_numeric'] = df_corr['screenshot_available'].astype(int)
        df_corr['resolved_numeric'] = df_corr['resolved'].astype(int)
        
        corr_cols = ['time_to_detect_sec', 'time_to_fix_sec', 'user_impact_score', 
                     'severity_numeric', 'screenshot_available_numeric', 'resolved_numeric']
        correlation_matrix = df_corr[corr_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/17_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        priority_counts = self.df['priority'].value_counts()
        ax.bar(priority_counts.index, priority_counts.values, color='coral')
        ax.set_title('Priority Distribution', fontsize=14, fontweight='bold')
        ax.set_ylabel('Count')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/18_priority_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nVisualizations saved to {output_dir}/")
    
    def generate_report(self, output_file='statistical_report.txt'):
        """Generate comprehensive statistical report"""
        from datetime import datetime
        with open(output_file, 'w') as f:
            f.write("\nBUG SEVERITY PREDICTION - STATISTICAL INSIGHTS REPORT\n")
            f.write("\n\n")
            
            # Dataset overview
            f.write("DATASET OVERVIEW:\n")
            f.write(f"Total Records: {len(self.df)}\n")
            f.write(f"Total Features: {len(self.df.columns)}\n")
            f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # f.write("DATA LEAKAGE PREVENTION NOTE:\n")
            # f.write("- 'resolved' feature is EXCLUDED from ML model features\n")
            # f.write("- (Resolution status only available after bug processing, not at prediction time)\n")
            # f.write("- This report analyzes 'resolved' for exploratory insights only\n\n")
            
            f.write("KEY FINDINGS:\n")
            f.write(f"1. Severity Distribution:\n")
            severity_dist = self.df['severity'].value_counts()
            for sev, count in severity_dist.items():
                pct = (count / len(self.df)) * 100
                f.write(f"   - {sev}: {count} bugs ({pct:.2f}%)\n")
            
            f.write(f"\n2. Resolution Rate: {(self.df['resolved'].sum() / len(self.df) * 100):.2f}%\n")
            
            f.write(f"\n3. Average Metrics:\n")
            f.write(f"   - Time to Detect: {self.df['time_to_detect_sec'].mean():.2f} seconds\n")
            f.write(f"   - Time to Fix: {self.df['time_to_fix_sec'].mean():.2f} seconds\n")
            f.write(f"   - User Impact Score: {self.df['user_impact_score'].mean():.2f}/10\n")
            
            f.write(f"\n4. Most Common Bug Type: {self.df['bug_type'].mode()[0]}\n")
            f.write(f"\n5. Most Used Device: {self.df['device_type'].mode()[0]}\n")
            f.write(f"   Most Used Browser: {self.df['browser'].mode()[0]}\n")
            f.write(f"   Most Used OS: {self.df['os'].mode()[0]}\n")
        
        print(f"Report saved to {output_file}")

def main():
    data_path = '../frontend_uiux_bug_dataset_cleaned.csv'
    insights = StatisticalInsights(data_path)
    
    insights.basic_statistics()
    insights.categorical_analysis()
    insights.severity_distribution()
    insights.correlation_analysis()
    insights.time_analysis()
    insights.user_impact_analysis()
    insights.resolution_analysis()
    insights.bug_type_analysis()
    insights.platform_analysis()
    insights.hypothesis_testing()
    
    insights.generate_visualizations('plots')
    insights.generate_report('statistical_report.txt')
    
    print("\nSTATISTICAL ANALYSIS COMPLETE")

if __name__ == "__main__":
    main()