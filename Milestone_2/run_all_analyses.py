# DS4SE Project: Milestone 2: Analysis Scripts
# Group Leader: Hira Zahoor (22K-4823)
# Members:   Hafsa Salman (22K-5161)
#            Jaysha Iqbal (22K-5178)
#            Amna Mansoor (22K-5159)

import subprocess
import sys
import os
from datetime import datetime

def print_header(title):
    """Print formatted header"""
    print("\n")

def main():
    """Run all analysis modules"""
    print_header("BUG SEVERITY PREDICTION - MILESTONE 2 ANALYSIS")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # List of analysis scripts
    scripts = [
        ('statistical_insights.py', 'Statistical Insights and Exploratory Data Analysis'),
        ('feature_selection.py', 'Feature Selection and Importance Analysis'),
        ('clustering_analysis.py', 'Clustering and Cluster Analysis')
    ]
    
    completed = []
    failed = []
    
    # Execute each script
    for script_name, description in scripts:
        print_header(description)
        
        try:
            result = subprocess.run(
                [sys.executable, script_name],
                capture_output=True,
                text=True,
                timeout=600
            )
            
            if result.returncode == 0:
                print("✓ COMPLETED SUCCESSFULLY")
                completed.append(description)
            else:
                print(f"✗ FAILED with return code {result.returncode}")
                if result.stderr:
                    print(f"Error: {result.stderr}")
                failed.append(description)
                
        except subprocess.TimeoutExpired:
            print(f"✗ TIMEOUT: Script took too long to execute")
            failed.append(description)
        except Exception as e:
            print(f"✗ ERROR: {str(e)}")
            failed.append(description)
    
    # Print summary
    print_header("ANALYSIS SUMMARY")
    
    print("Completed Analyses:")
    for i, analysis in enumerate(completed, 1):
        print(f"  ✓ {i}. {analysis}")
    
    if failed:
        print(f"\nFailed Analyses:")
        for i, analysis in enumerate(failed, 1):
            print(f"  ✗ {i}. {analysis}")
    else:
        print("\n✓ All analyses completed successfully!")
    
    print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nGenerated Files:")
    print("  - statistical_report.txt")
    print("  - feature_selection_report.txt")
    print("  - clustering_report.txt")
    print("  - bug_dataset_with_kmeans_clusters.csv")
    print("  - bug_dataset_with_hierarchical_clusters.csv")
    print("  - bug_dataset_with_dbscan_clusters.csv")
    print("  - plots/ (directory with visualizations)")

if __name__ == "__main__":
    main()
