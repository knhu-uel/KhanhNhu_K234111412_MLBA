import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from flask import Flask, render_template, request
import json
from project_retail.connectors.connector import Connector

class CustomerClusterAnalysis:
    def __init__(self, database="salesdatabase"):
        """Initialize the customer cluster analysis with database connection"""
        self.conn = Connector(database=database)
        self.conn.connect()
        self.df_customers = None
        self.df_clustered = None
        self.cluster_labels = None
        self.n_clusters = None
        
    def load_customer_data(self):
        """Load customer data from MySQL database"""
        try:
            # Get all customer details
            sql_customers = "SELECT * FROM customer"
            self.df_customers = self.conn.queryDataset(sql_customers)
            
            # Get customer data with spending scores for clustering
            # Using the exact query format from the working test file
            sql_clustering = (
                "SELECT DISTINCT customer.CustomerID, customer.Name, customer.Gender, "
                "customer.Age, customer_spend_score.Annual_Income, customer_spend_score.Spending_Score "
                "FROM customer, customer_spend_score "
                "WHERE customer.CustomerID = customer_spend_score.CustomerID"
            )
            self.df_clustered = self.conn.queryDataset(sql_clustering)
            
            if self.df_clustered is not None and not self.df_clustered.empty:
                self.df_clustered.columns = ['CustomerID', 'Name', 'Gender', 'Age', 'Annual_Income', 'Spending_Score']
                print(f"Loaded {len(self.df_clustered)} customers for clustering analysis")
                return True
            else:
                print("No customer data found")
                return False
                
        except Exception as e:
            print(f"Error loading customer data: {e}")
            return False
    
    def perform_clustering(self, features=['Age', 'Spending_Score'], n_clusters=4, scale_data=False):
        """Perform K-means clustering on customer data"""
        if self.df_clustered is None or self.df_clustered.empty:
            print("No customer data loaded. Please load data first.")
            return False
            
        try:
            # Prepare feature matrix
            X = self.df_clustered[features].values
            
            # Scale data if requested
            if scale_data:
                scaler = StandardScaler()
                X = scaler.fit_transform(X)
            
            # Perform K-means clustering
            kmeans = KMeans(
                n_clusters=n_clusters,
                init='k-means++',
                max_iter=500,
                random_state=42
            )
            
            self.cluster_labels = kmeans.fit_predict(X)
            self.n_clusters = n_clusters
            
            # Add cluster labels to dataframe
            self.df_clustered['Cluster'] = self.cluster_labels
            
            print(f"Clustering completed with {n_clusters} clusters using features: {features}")
            return True
            
        except Exception as e:
            print(f"Error performing clustering: {e}")
            return False
    
    def get_customers_by_cluster(self, cluster_id):
        """Retrieve detailed customer information for a specific cluster"""
        if self.df_clustered is None or 'Cluster' not in self.df_clustered.columns:
            print("Clustering not performed. Please perform clustering first.")
            return None
            
        try:
            # Get customer IDs in the specified cluster
            cluster_customers = self.df_clustered[self.df_clustered['Cluster'] == cluster_id]
            customer_ids = cluster_customers['CustomerID'].tolist()
            
            if not customer_ids:
                print(f"No customers found in cluster {cluster_id}")
                return None
            
            # Get detailed customer information from the main customer table
            customer_ids_str = ','.join(map(str, customer_ids))
            sql = f"SELECT * FROM customer WHERE CustomerID IN ({customer_ids_str})"
            detailed_customers = self.conn.queryDataset(sql)
            
            # Merge with cluster information
            if detailed_customers is not None and not detailed_customers.empty:
                # Add cluster information
                cluster_info = cluster_customers[['CustomerID', 'Age', 'Annual_Income', 'Spending_Score', 'Cluster']]
                detailed_with_cluster = detailed_customers.merge(cluster_info, on='CustomerID', how='left')
                return detailed_with_cluster
            
            return None
            
        except Exception as e:
            print(f"Error retrieving customers for cluster {cluster_id}: {e}")
            return None
    
    def display_cluster_summary_console(self):
        """Display cluster summary on console"""
        if self.df_clustered is None or 'Cluster' not in self.df_clustered.columns:
            print("Clustering not performed. Please perform clustering first.")
            return
            
        print("\n" + "="*80)
        print("CUSTOMER CLUSTER ANALYSIS SUMMARY")
        print("="*80)
        
        for cluster_id in range(self.n_clusters):
            cluster_data = self.df_clustered[self.df_clustered['Cluster'] == cluster_id]
            print(f"\nCluster {cluster_id + 1}:")
            print(f"  Number of customers: {len(cluster_data)}")
            print(f"  Average Age: {cluster_data['Age'].mean():.1f}")
            print(f"  Average Annual Income: ${cluster_data['Annual_Income'].mean():.2f}")
            print(f"  Average Spending Score: {cluster_data['Spending_Score'].mean():.1f}")
    
    def display_customers_by_cluster_console(self, cluster_id=None):
        """Display detailed customer list for each cluster on console"""
        if self.df_clustered is None or 'Cluster' not in self.df_clustered.columns:
            print("Clustering not performed. Please perform clustering first.")
            return
            
        clusters_to_display = [cluster_id] if cluster_id is not None else range(self.n_clusters)
        
        for cid in clusters_to_display:
            customers = self.get_customers_by_cluster(cid)
            
            if customers is not None and not customers.empty:
                print(f"\n{'='*80}")
                print(f"CLUSTER {cid} DETAILS")
                print(f"{'='*80}")
                
                # Display cluster summary
                cluster_data = self.df_clustered[self.df_clustered['Cluster'] == cid]
                print(f"Number of customers: {len(cluster_data)}")
                print(f"Average age: {cluster_data['Age'].mean():.1f}")
                print(f"Average income: ${cluster_data['Annual_Income'].mean():.2f}")
                print(f"Average spending score: {cluster_data['Spending_Score'].mean():.1f}")
                
                print(f"\nCustomer Details:")
                print("-" * 80)
                print(f"{'ID':<8} {'Name':<20} {'Gender':<8} {'Age':<5} {'Income':<10} {'Score':<6}")
                print("-" * 80)
                
                for _, customer in customers.iterrows():
                    # Handle potential column name conflicts from merge
                    age = customer.get('Age_y', customer.get('Age', 'N/A'))
                    print(f"{customer['CustomerID']:<8} {customer['Name']:<20} "
                          f"{customer['Gender']:<8} {age:<5} ${customer['Annual_Income']:<9.0f} "
                          f"{customer['Spending_Score']:<6}")
            else:
                print(f"\nCluster {cid}: No customers found")

# Flask Web Application for displaying clusters
app = Flask(__name__)
cluster_analysis = None

@app.route('/')
def index():
    """Main page showing cluster overview"""
    global cluster_analysis
    
    if cluster_analysis is None or cluster_analysis.df_clustered is None:
        return render_template('error.html', message="No clustering data available. Please run clustering first.")
    
    # Prepare cluster summary data
    cluster_summary = []
    for cluster_id in range(cluster_analysis.n_clusters):
        cluster_data = cluster_analysis.df_clustered[cluster_analysis.df_clustered['Cluster'] == cluster_id]
        summary = {
            'cluster_id': cluster_id + 1,
            'count': len(cluster_data),
            'avg_age': round(cluster_data['Age'].mean(), 1),
            'avg_income': round(cluster_data['Annual_Income'].mean(), 2),
            'avg_score': round(cluster_data['Spending_Score'].mean(), 1)
        }
        cluster_summary.append(summary)
    
    return render_template('cluster_overview.html', clusters=cluster_summary)

@app.route('/cluster/<int:cluster_id>')
def cluster_details(cluster_id):
    """Display detailed customer list for a specific cluster"""
    global cluster_analysis
    
    if cluster_analysis is None:
        return render_template('error.html', message="No clustering data available.")
    
    # Convert to 0-based index
    cluster_index = cluster_id - 1
    
    if cluster_index < 0 or cluster_index >= cluster_analysis.n_clusters:
        return render_template('error.html', message=f"Invalid cluster ID: {cluster_id}")
    
    customers = cluster_analysis.get_customers_by_cluster(cluster_index)
    
    if customers is None or customers.empty:
        return render_template('error.html', message=f"No customers found in cluster {cluster_id}")
    
    # Convert DataFrame to list of dictionaries for template
    customer_list = []
    for _, customer in customers.iterrows():
        # Handle potential column name conflicts from merge
        age = customer.get('Age_y', customer.get('Age', 'N/A'))
        customer_dict = {
            'id': customer['CustomerID'],
            'name': customer.get('Name', 'N/A'),
            'gender': customer.get('Gender', 'N/A'),
            'age': age,
            'income': customer.get('Annual_Income', 'N/A'),
            'score': customer.get('Spending_Score', 'N/A'),
            'email': 'N/A',  # Not available in current schema
            'address': 'N/A',  # Not available in current schema
            'city': 'N/A',  # Not available in current schema
            'state': 'N/A',  # Not available in current schema
            'country': 'N/A',  # Not available in current schema
            'postal_code': 'N/A'  # Not available in current schema
        }
        customer_list.append(customer_dict)
    
    return render_template('cluster_details.html', 
                         cluster_id=cluster_id, 
                         customers=customer_list,
                         customer_count=len(customer_list))

def start_web_server(host='localhost', port=5000, debug=True):
    """Start the Flask web server"""
    print(f"Starting web server at http://{host}:{port}")
    print("Available routes:")
    print(f"  - Cluster Overview: http://{host}:{port}/")
    print(f"  - Cluster Details: http://{host}:{port}/cluster/<cluster_id>")
    app.run(host=host, port=port, debug=debug)

def display_customers_web(analysis_instance, host='localhost', port=5000):
    """Function to display customers on web interface"""
    global cluster_analysis
    cluster_analysis = analysis_instance
    start_web_server(host, port, debug=False)

# Example usage functions
def run_clustering_scenario_1():
    """Scenario 1: Age and Spending Score clustering with 4 clusters"""
    print("\n" + "="*80)
    print("SCENARIO 1: Age and Spending Score Clustering (4 clusters)")
    print("="*80)
    
    analysis = CustomerClusterAnalysis()
    
    if analysis.load_customer_data():
        if analysis.perform_clustering(features=['Age', 'Spending_Score'], n_clusters=4):
            analysis.display_cluster_summary_console()
            analysis.display_customers_by_cluster_console()
            return analysis
    return None

def run_clustering_scenario_2():
    """Scenario 2: Age, Income, and Spending Score clustering with 5 clusters (scaled)"""
    print("\n" + "="*80)
    print("SCENARIO 2: Age, Income, and Spending Score Clustering (5 clusters, scaled)")
    print("="*80)
    
    analysis = CustomerClusterAnalysis()
    
    if analysis.load_customer_data():
        if analysis.perform_clustering(features=['Age', 'Annual_Income', 'Spending_Score'], 
                                     n_clusters=5, scale_data=True):
            analysis.display_cluster_summary_console()
            analysis.display_customers_by_cluster_console()
            return analysis
    return None

def run_clustering_scenario_3():
    """Scenario 3: Income and Spending Score clustering with 3 clusters"""
    print("\n" + "="*80)
    print("SCENARIO 3: Income and Spending Score Clustering (3 clusters)")
    print("="*80)
    
    analysis = CustomerClusterAnalysis()
    
    if analysis.load_customer_data():
        if analysis.perform_clustering(features=['Annual_Income', 'Spending_Score'], n_clusters=3):
            analysis.display_cluster_summary_console()
            analysis.display_customers_by_cluster_console()
            return analysis
    return None

if __name__ == "__main__":
    # Run different clustering scenarios
    print("Running Customer Cluster Analysis...")
    
    # Scenario 1: Console display
    scenario1 = run_clustering_scenario_1()
    
    # Scenario 2: Console display
    scenario2 = run_clustering_scenario_2()
    
    # Scenario 3: Console display
    scenario3 = run_clustering_scenario_3()
    
    # Web display example (uncomment to run web server)
    # if scenario1:
    #     print("\nStarting web interface for Scenario 1...")
    #     display_customers_web(scenario1)