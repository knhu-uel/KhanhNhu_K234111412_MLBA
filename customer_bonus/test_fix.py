#!/usr/bin/env python3
"""
Simple test script to verify the customer cluster analysis fixes
"""

from customer_bonus.customer_cluster_analysis import CustomerClusterAnalysis

def test_basic_functionality():
    """Test basic clustering functionality"""
    print("Testing Customer Cluster Analysis...")
    
    # Initialize the analysis
    analysis = CustomerClusterAnalysis()
    
    # Test data loading
    print("\n1. Testing data loading...")
    if analysis.load_customer_data():
        print("✓ Data loading successful")
        print(f"  Loaded {len(analysis.df_clustered)} customers")
        print(f"  Columns: {analysis.df_clustered.columns.tolist()}")
    else:
        print("✗ Data loading failed")
        return False
    
    # Test clustering
    print("\n2. Testing clustering...")
    if analysis.perform_clustering(features=['Age', 'Spending_Score'], n_clusters=4):
        print("✓ Clustering successful")
        print(f"  Created {analysis.n_clusters} clusters")
    else:
        print("✗ Clustering failed")
        return False
    
    # Test console display
    print("\n3. Testing console display...")
    try:
        analysis.display_cluster_summary_console()
        print("✓ Console summary display successful")
    except Exception as e:
        print(f"✗ Console summary display failed: {e}")
        return False
    
    # Test detailed display for one cluster
    print("\n4. Testing detailed cluster display...")
    try:
        analysis.display_customers_by_cluster_console(cluster_id=0)
        print("✓ Detailed cluster display successful")
    except Exception as e:
        print(f"✗ Detailed cluster display failed: {e}")
        return False
    
    print("\n✓ All tests passed! The customer cluster analysis is working correctly.")
    return True

if __name__ == "__main__":
    test_basic_functionality()