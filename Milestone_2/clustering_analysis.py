# DS4SE Project: Milestone 2: Bonus: Clustering Analysis
# Group Leader: Hira Zahoor (22K-4823)
# Members:   Hafsa Salman (22K-5161)
#            Jaysha Iqbal (22K-5178)
#            Amna Mansoor (22K-5159)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score

class ClusteringAnalysis:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)

        self.features = ['time_to_detect_sec', 'time_to_fix_sec', 'user_impact_score']
        self.X = self.df[self.features]

        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(self.X)

        self.n_clusters = self.find_optimal_clusters()
        print(f"\nOptimal number of clusters determined: {self.n_clusters}")
    
    def find_optimal_clusters(self, k_range=range(2, 11)):
        """
        Find optimal number of clusters using elbow method
        Tests k values from 2 to 10 and returns the elbow point
        """
        print("\nELBOW METHOD: Finding Optimal Number of Clusters")
        print("-" * 50)
        
        inertias = []
        silhouette_scores = []
        k_values = list(k_range)
        
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.X_scaled)
            inertias.append(kmeans.inertia_)
            score = silhouette_score(self.X_scaled, kmeans.labels_)
            silhouette_scores.append(score)
            print(f"k={k}: Inertia={kmeans.inertia_:.2f}, Silhouette Score={score:.4f}")
        
        n_points = len(inertias)
        all_coords = np.vstack((range(n_points), inertias)).T
        first_point = all_coords[0]
        last_point = all_coords[-1]
        
        line_vec = last_point - first_point
        line_vec_normalized = line_vec / np.sqrt(np.sum(line_vec**2))
        
        vec_from_first = all_coords - first_point
        scalar_product = np.sum(vec_from_first * line_vec_normalized, axis=1)
        point_to_line = vec_from_first - np.outer(scalar_product, line_vec_normalized)
        distances = np.sqrt(np.sum(point_to_line**2, axis=1))
        
        optimal_idx = np.argmax(distances) + k_values[0]
        
        # Plot elbow curve
        self.plot_elbow_curve(k_values, inertias, silhouette_scores, optimal_idx)
        
        print(f"\nElbow point identified at k={optimal_idx}")
        return optimal_idx
    
    def plot_elbow_curve(self, k_values, inertias, silhouette_scores, optimal_k, output_dir='plots'):
        """Plot elbow curve and silhouette scores"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Inertia plot
        axes[0].plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
        axes[0].axvline(x=optimal_k, color='r', linestyle='--', linewidth=2, label=f'Optimal k={optimal_k}')
        axes[0].set_xlabel('Number of Clusters (k)', fontsize=12)
        axes[0].set_ylabel('Inertia', fontsize=12)
        axes[0].set_title('Elbow Method: Inertia vs Number of Clusters', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Silhouette score plot
        axes[1].plot(k_values, silhouette_scores, 'go-', linewidth=2, markersize=8)
        axes[1].axvline(x=optimal_k, color='r', linestyle='--', linewidth=2, label=f'Optimal k={optimal_k}')
        axes[1].set_xlabel('Number of Clusters (k)', fontsize=12)
        axes[1].set_ylabel('Silhouette Score', fontsize=12)
        axes[1].set_title('Silhouette Score vs Number of Clusters', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/00_elbow_method.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Elbow curve saved to {output_dir}/00_elbow_method.png")

    def kmeans_clustering(self):
        print("\nKMEANS CLUSTERING")

        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(self.X_scaled)

        score = silhouette_score(self.X_scaled, labels)
        print(f"Silhouette Score: {score:.4f}")

        return labels

    def dbscan_clustering(self):
        print("\nDBSCAN CLUSTERING")

        dbscan = DBSCAN(eps=0.5, min_samples=5)
        labels = dbscan.fit_predict(self.X_scaled)

        if len(set(labels)) > 1:
            score = silhouette_score(self.X_scaled, labels)
            print(f"Silhouette Score: {score:.4f}")
        else:
            print("Silhouette Score not valid (single cluster)")

        return labels

    def hierarchical_clustering(self, sample_size=5000):
        print("\nHIERARCHICAL CLUSTERING (SAMPLED)")

        if len(self.X_scaled) > sample_size:
            indices = np.random.choice(len(self.X_scaled), sample_size, replace=False)
            X_sample = self.X_scaled[indices]
        else:
            indices = np.arange(len(self.X_scaled))
            X_sample = self.X_scaled

        hierarchical = AgglomerativeClustering(n_clusters=self.n_clusters, linkage='ward')
        labels = hierarchical.fit_predict(X_sample)

        print(f"Used {len(X_sample)} samples out of {len(self.X_scaled)}")

        return labels, indices

    def analyze_cluster_characteristics(self, labels, method_name, indices=None):
        print(f"\nCLUSTER ANALYSIS: {method_name}")

        if indices is not None:
            df_analysis = self.df.iloc[indices].copy()
        else:
            df_analysis = self.df.copy()

        df_analysis['Cluster'] = labels

        print("\nCluster Counts:")
        print(df_analysis['Cluster'].value_counts())

        print("\nCluster Means:")
        print(df_analysis.groupby('Cluster')[self.features].mean().round(2))

    def save_results(self, labels, filename):
        df_out = self.df.copy()
        df_out['Cluster'] = labels
        df_out.to_csv(filename, index=False)
        print(f"Saved: {filename}")


def main():
    data_path = '../frontend_uiux_bug_dataset_cleaned.csv'
    ca = ClusteringAnalysis(data_path)

    # KMeans
    kmeans_labels = ca.kmeans_clustering()
    ca.analyze_cluster_characteristics(kmeans_labels, "KMeans")
    ca.save_results(kmeans_labels, 'bug_dataset_with_kmeans_clusters.csv')

    # DBSCAN
    dbscan_labels = ca.dbscan_clustering()
    ca.analyze_cluster_characteristics(dbscan_labels, "DBSCAN")
    ca.save_results(dbscan_labels, 'bug_dataset_with_dbscan_clusters.csv')

    # Hierarchical 
    hierarchical_labels, indices = ca.hierarchical_clustering()

    ca.analyze_cluster_characteristics(hierarchical_labels, "Hierarchical", indices)

    df_sample = ca.df.iloc[indices].copy()
    df_sample['Cluster'] = hierarchical_labels
    df_sample.to_csv('bug_dataset_with_hierarchical_clusters.csv', index=False)
    print("Saved: bug_dataset_with_hierarchical_clusters.csv")

    print("\nCLUSTERING ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()