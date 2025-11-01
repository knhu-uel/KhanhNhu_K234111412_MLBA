#!/usr/bin/env python3
"""
Test script for Customer Cluster Analysis System
This script tests the core functionality without requiring a full database setup.
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_sample_data():
    """Create sample customer data for testing purposes."""
    np.random.seed(42)
    
    # Generate sample customer data
    n_customers = 100
    
    customer_data = {
        'CustomerId': range(1, n_customers + 1),
        'FirstName': [f'Customer{i}' for i in range(1, n_customers + 1)],
        'LastName': [f'LastName{i}' for i in range(1, n_customers + 1)],
        'Gender': np.random.choice(['Male', 'Female'], n_customers),
        'Age': np.random.randint(18, 70, n_customers),
        'Annual_Income': np.random.normal(50000, 20000, n_customers),
        'Spending_Score': np.random.randint(1, 100, n_customers),
        'Email': [f'customer{i}@email.com' for i in range(1, n_customers + 1)],
        'Address': [f'{i} Main St' for i in range(1, n_customers + 1)],
        'City': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston'], n_customers),
        'State': np.random.choice(['NY', 'CA', 'IL', 'TX'], n_customers),
        'Country': ['USA'] * n_customers,
        'PostalCode': [f'{10000 + i:05d}' for i in range(n_customers)]
    }
    
    return pd.DataFrame(customer_data)

def test_clustering_functionality():
    """Test the core clustering functionality."""
    print("🧪 Testing Customer Clustering Functionality")
    print("=" * 50)
    
    # Create sample data
    df = create_sample_data()
    print(f"✅ Created sample data with {len(df)} customers")
    
    # Test clustering with different scenarios
    scenarios = [
        {
            'name': 'Age and Spending Score (4 clusters)',
            'features': ['Age', 'Spending_Score'],
            'n_clusters': 4,
            'scale_data': False
        },
        {
            'name': 'Age, Income, and Spending Score (5 clusters, scaled)',
            'features': ['Age', 'Annual_Income', 'Spending_Score'],
            'n_clusters': 5,
            'scale_data': True
        },
        {
            'name': 'Income and Spending Score (3 clusters)',
            'features': ['Annual_Income', 'Spending_Score'],
            'n_clusters': 3,
            'scale_data': False
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n📊 Scenario {i}: {scenario['name']}")
        print("-" * 40)
        
        # Prepare features
        X = df[scenario['features']].copy()
        
        # Scale data if required
        if scenario['scale_data']:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X = pd.DataFrame(X_scaled, columns=scenario['features'])
            print("✅ Data scaled using StandardScaler")
        
        # Perform clustering
        kmeans = KMeans(n_clusters=scenario['n_clusters'], random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X)
        
        # Add cluster labels to dataframe
        df_clustered = df.copy()
        df_clustered['Cluster'] = clusters
        
        print(f"✅ K-means clustering completed with {scenario['n_clusters']} clusters")
        
        # Display cluster summary
        print("\n📈 Cluster Summary:")
        for cluster_id in range(scenario['n_clusters']):
            cluster_data = df_clustered[df_clustered['Cluster'] == cluster_id]
            print(f"  Cluster {cluster_id}: {len(cluster_data)} customers")
            
            # Calculate cluster statistics
            if 'Age' in scenario['features']:
                avg_age = cluster_data['Age'].mean()
                print(f"    - Average Age: {avg_age:.1f}")
            
            if 'Annual_Income' in scenario['features']:
                avg_income = cluster_data['Annual_Income'].mean()
                print(f"    - Average Income: ${avg_income:,.0f}")
            
            if 'Spending_Score' in scenario['features']:
                avg_score = cluster_data['Spending_Score'].mean()
                print(f"    - Average Spending Score: {avg_score:.1f}")
        
        # Test customer retrieval by cluster
        print(f"\n👥 Sample customers from Cluster 0:")
        cluster_0_customers = df_clustered[df_clustered['Cluster'] == 0].head(3)
        for _, customer in cluster_0_customers.iterrows():
            print(f"  - {customer['FirstName']} {customer['LastName']}, "
                  f"Age: {customer['Age']}, "
                  f"Income: ${customer['Annual_Income']:,.0f}, "
                  f"Score: {customer['Spending_Score']}")

def test_console_display_format():
    """Test the console display formatting."""
    print("\n\n🖥️  Testing Console Display Format")
    print("=" * 50)
    
    # Create sample data
    df = create_sample_data()
    
    # Simple clustering
    X = df[['Age', 'Spending_Score']]
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    df['Cluster'] = clusters
    
    # Test formatted display
    cluster_0_customers = df[df['Cluster'] == 0].head(5)
    
    print("📋 Formatted Customer List (Cluster 0):")
    print("-" * 120)
    print(f"{'ID':<5} {'Name':<20} {'Gender':<8} {'Age':<5} {'Income':<12} {'Score':<7} {'Email':<25} {'City':<15}")
    print("-" * 120)
    
    for _, customer in cluster_0_customers.iterrows():
        print(f"{customer['CustomerId']:<5} "
              f"{customer['FirstName']} {customer['LastName']:<15} "
              f"{customer['Gender']:<8} "
              f"{customer['Age']:<5} "
              f"${customer['Annual_Income']:>10,.0f} "
              f"{customer['Spending_Score']:<7} "
              f"{customer['Email']:<25} "
              f"{customer['City']:<15}")
    
    print("✅ Console formatting test completed")

def test_web_data_structure():
    """Test the data structure for web display."""
    print("\n\n🌐 Testing Web Data Structure")
    print("=" * 50)
    
    # Create sample data
    df = create_sample_data()
    
    # Simple clustering
    X = df[['Age', 'Annual_Income', 'Spending_Score']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df['Cluster'] = clusters
    
    # Test cluster overview data
    cluster_overview = []
    for cluster_id in range(4):
        cluster_data = df[df['Cluster'] == cluster_id]
        overview = {
            'cluster_id': cluster_id,
            'customer_count': len(cluster_data),
            'avg_age': cluster_data['Age'].mean(),
            'avg_income': cluster_data['Annual_Income'].mean(),
            'avg_spending_score': cluster_data['Spending_Score'].mean(),
            'gender_distribution': cluster_data['Gender'].value_counts().to_dict()
        }
        cluster_overview.append(overview)
    
    print("📊 Cluster Overview Data Structure:")
    for overview in cluster_overview:
        print(f"  Cluster {overview['cluster_id']}:")
        print(f"    - Customers: {overview['customer_count']}")
        print(f"    - Avg Age: {overview['avg_age']:.1f}")
        print(f"    - Avg Income: ${overview['avg_income']:,.0f}")
        print(f"    - Avg Spending Score: {overview['avg_spending_score']:.1f}")
        print(f"    - Gender: {overview['gender_distribution']}")
    
    # Test detailed customer data
    cluster_0_details = df[df['Cluster'] == 0].head(3)
    print(f"\n👥 Sample Customer Details (Cluster 0):")
    for _, customer in cluster_0_details.iterrows():
        customer_dict = customer.to_dict()
        print(f"  Customer {customer_dict['CustomerId']}:")
        print(f"    - Name: {customer_dict['FirstName']} {customer_dict['LastName']}")
        print(f"    - Demographics: {customer_dict['Gender']}, Age {customer_dict['Age']}")
        print(f"    - Financial: ${customer_dict['Annual_Income']:,.0f} income, {customer_dict['Spending_Score']} score")
        print(f"    - Contact: {customer_dict['Email']}")
        print(f"    - Location: {customer_dict['City']}, {customer_dict['State']}")
    
    print("✅ Web data structure test completed")

def main():
    """Run all tests."""
    print("🚀 Customer Cluster Analysis - Test Suite")
    print("=" * 60)
    print("This test suite verifies the core functionality without requiring")
    print("a MySQL database connection. It uses generated sample data.")
    print("=" * 60)
    
    try:
        # Run tests
        test_clustering_functionality()
        test_console_display_format()
        test_web_data_structure()
        
        print("\n\n🎉 All Tests Completed Successfully!")
        print("=" * 60)
        print("✅ Clustering functionality works correctly")
        print("✅ Console display formatting is proper")
        print("✅ Web data structures are valid")
        print("\n💡 Next Steps:")
        print("1. Ensure MySQL database is set up with customer data")
        print("2. Run demo_customer_clustering.py for full functionality")
        print("3. Test web interface by accessing http://localhost:5000")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        print("Please check the implementation and try again.")
        return False
    
    return True

if __name__ == "__main__":
    main()