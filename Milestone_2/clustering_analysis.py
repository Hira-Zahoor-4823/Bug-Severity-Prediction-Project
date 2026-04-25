# DS4SE Project: Milestone 2: Bonus: Clustering Analysis
# Group Leader: Hira Zahoor (22K-4823)
# Members:   Hafsa Salman (22K-5161)
#            Jaysha Iqbal (22K-5178)
#            Amna Mansoor (22K-5159)

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score

class ClusteringAnalysis:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)

        # Select numeric features for clustering
        self.features = ['time_to_detect_sec', 'time_to_fix_sec', 'user_impact_score']
        self.X = self.df[self.features]

        # Scale data
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(self.X)

        self.n_clusters = 4

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