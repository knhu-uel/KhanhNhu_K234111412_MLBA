"""
Customer Cluster Analysis Demonstration Script

This script demonstrates how to use the customer clustering analysis functions
for both console and web display in different clustering scenarios.

Author: Machine Learning Project
Date: 2024
"""

from customer_bonus.customer_cluster_analysis import (
    CustomerClusterAnalysis, 
    display_customers_web,
    run_clustering_scenario_1,
    run_clustering_scenario_2,
    run_clustering_scenario_3
)

def display_menu():
    """Display the main menu options"""
    print("\n" + "="*80)
    print("CUSTOMER CLUSTER ANALYSIS - DEMONSTRATION MENU")
    print("="*80)
    print("1. Console Display - Age & Spending Score Clustering (4 clusters)")
    print("2. Console Display - Age, Income & Spending Score Clustering (5 clusters, scaled)")
    print("3. Console Display - Income & Spending Score Clustering (3 clusters)")
    print("4. Web Display - Age & Spending Score Clustering (4 clusters)")
    print("5. Web Display - Age, Income & Spending Score Clustering (5 clusters, scaled)")
    print("6. Web Display - Income & Spending Score Clustering (3 clusters)")
    print("7. Custom Clustering Configuration")
    print("8. Compare All Scenarios (Console)")
    print("9. Exit")
    print("="*80)

def console_display_scenario_1():
    """Console display for scenario 1: Age and Spending Score clustering"""
    print("\n🖥️  CONSOLE DISPLAY - SCENARIO 1")
    analysis = run_clustering_scenario_1()
    if analysis:
        print("\n✅ Scenario 1 completed successfully!")
        input("\nPress Enter to continue...")
    else:
        print("\n❌ Failed to run scenario 1")

def console_display_scenario_2():
    """Console display for scenario 2: Age, Income, and Spending Score clustering"""
    print("\n🖥️  CONSOLE DISPLAY - SCENARIO 2")
    analysis = run_clustering_scenario_2()
    if analysis:
        print("\n✅ Scenario 2 completed successfully!")
        input("\nPress Enter to continue...")
    else:
        print("\n❌ Failed to run scenario 2")

def console_display_scenario_3():
    """Console display for scenario 3: Income and Spending Score clustering"""
    print("\n🖥️  CONSOLE DISPLAY - SCENARIO 3")
    analysis = run_clustering_scenario_3()
    if analysis:
        print("\n✅ Scenario 3 completed successfully!")
        input("\nPress Enter to continue...")
    else:
        print("\n❌ Failed to run scenario 3")

def web_display_scenario_1():
    """Web display for scenario 1: Age and Spending Score clustering"""
    print("\n🌐 WEB DISPLAY - SCENARIO 1")
    print("Setting up Age & Spending Score clustering for web display...")
    
    analysis = CustomerClusterAnalysis()
    if analysis.load_customer_data():
        if analysis.perform_clustering(features=['Age', 'Spending_Score'], n_clusters=4):
            print("\n✅ Clustering completed successfully!")
            print("🚀 Starting web server...")
            print("📱 Open your browser and go to: http://localhost:5000")
            print("⏹️  Press Ctrl+C to stop the web server")
            
            try:
                display_customers_web(analysis, host='localhost', port=5000)
            except KeyboardInterrupt:
                print("\n🛑 Web server stopped by user")
        else:
            print("\n❌ Failed to perform clustering")
    else:
        print("\n❌ Failed to load customer data")

def web_display_scenario_2():
    """Web display for scenario 2: Age, Income, and Spending Score clustering"""
    print("\n🌐 WEB DISPLAY - SCENARIO 2")
    print("Setting up Age, Income & Spending Score clustering for web display...")
    
    analysis = CustomerClusterAnalysis()
    if analysis.load_customer_data():
        if analysis.perform_clustering(features=['Age', 'Annual_Income', 'Spending_Score'], 
                                     n_clusters=5, scale_data=True):
            print("\n✅ Clustering completed successfully!")
            print("🚀 Starting web server...")
            print("📱 Open your browser and go to: http://localhost:5001")
            print("⏹️  Press Ctrl+C to stop the web server")
            
            try:
                display_customers_web(analysis, host='localhost', port=5001)
            except KeyboardInterrupt:
                print("\n🛑 Web server stopped by user")
        else:
            print("\n❌ Failed to perform clustering")
    else:
        print("\n❌ Failed to load customer data")

def web_display_scenario_3():
    """Web display for scenario 3: Income and Spending Score clustering"""
    print("\n🌐 WEB DISPLAY - SCENARIO 3")
    print("Setting up Income & Spending Score clustering for web display...")
    
    analysis = CustomerClusterAnalysis()
    if analysis.load_customer_data():
        if analysis.perform_clustering(features=['Annual_Income', 'Spending_Score'], n_clusters=3):
            print("\n✅ Clustering completed successfully!")
            print("🚀 Starting web server...")
            print("📱 Open your browser and go to: http://localhost:5002")
            print("⏹️  Press Ctrl+C to stop the web server")
            
            try:
                display_customers_web(analysis, host='localhost', port=5002)
            except KeyboardInterrupt:
                print("\n🛑 Web server stopped by user")
        else:
            print("\n❌ Failed to perform clustering")
    else:
        print("\n❌ Failed to load customer data")

def custom_clustering():
    """Allow user to configure custom clustering parameters"""
    print("\n⚙️  CUSTOM CLUSTERING CONFIGURATION")
    print("="*50)
    
    analysis = CustomerClusterAnalysis()
    if not analysis.load_customer_data():
        print("❌ Failed to load customer data")
        return
    
    # Show available features
    print("Available features:")
    print("1. Age")
    print("2. Annual_Income")
    print("3. Spending_Score")
    
    # Get feature selection
    print("\nSelect features for clustering (comma-separated numbers, e.g., 1,3):")
    feature_input = input("Features: ").strip()
    
    feature_map = {
        '1': 'Age',
        '2': 'Annual_Income', 
        '3': 'Spending_Score'
    }
    
    try:
        selected_features = [feature_map[f.strip()] for f in feature_input.split(',')]
        print(f"Selected features: {selected_features}")
    except KeyError:
        print("❌ Invalid feature selection")
        return
    
    # Get number of clusters
    try:
        n_clusters = int(input("Number of clusters (2-10): "))
        if n_clusters < 2 or n_clusters > 10:
            print("❌ Number of clusters must be between 2 and 10")
            return
    except ValueError:
        print("❌ Invalid number of clusters")
        return
    
    # Get scaling option
    scale_input = input("Scale data? (y/n): ").strip().lower()
    scale_data = scale_input in ['y', 'yes']
    
    # Get display option
    display_input = input("Display on (c)onsole or (w)eb? ").strip().lower()
    
    # Perform clustering
    print(f"\n🔄 Performing clustering with {n_clusters} clusters...")
    if analysis.perform_clustering(features=selected_features, n_clusters=n_clusters, scale_data=scale_data):
        print("✅ Clustering completed!")
        
        if display_input in ['c', 'console']:
            analysis.display_cluster_summary_console()
            analysis.display_customers_by_cluster_console()
            input("\nPress Enter to continue...")
        elif display_input in ['w', 'web']:
            print("🚀 Starting web server...")
            print("📱 Open your browser and go to: http://localhost:5003")
            print("⏹️  Press Ctrl+C to stop the web server")
            
            try:
                display_customers_web(analysis, host='localhost', port=5003)
            except KeyboardInterrupt:
                print("\n🛑 Web server stopped by user")
        else:
            print("❌ Invalid display option")
    else:
        print("❌ Failed to perform clustering")

def compare_all_scenarios():
    """Compare all clustering scenarios on console"""
    print("\n📊 COMPARING ALL CLUSTERING SCENARIOS")
    print("="*80)
    
    scenarios = [
        ("Scenario 1: Age & Spending Score (4 clusters)", run_clustering_scenario_1),
        ("Scenario 2: Age, Income & Spending Score (5 clusters, scaled)", run_clustering_scenario_2),
        ("Scenario 3: Income & Spending Score (3 clusters)", run_clustering_scenario_3)
    ]
    
    results = []
    
    for name, scenario_func in scenarios:
        print(f"\n🔄 Running {name}...")
        analysis = scenario_func()
        if analysis:
            results.append((name, analysis))
            print(f"✅ {name} completed")
        else:
            print(f"❌ {name} failed")
    
    print(f"\n📈 COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Scenario':<50} {'Clusters':<10} {'Customers':<12}")
    print("-"*80)
    
    for name, analysis in results:
        if analysis.df_clustered is not None:
            n_clusters = analysis.n_clusters
            n_customers = len(analysis.df_clustered)
            print(f"{name:<50} {n_clusters:<10} {n_customers:<12}")
    
    input("\nPress Enter to continue...")

def main():
    """Main function to run the demonstration"""
    print("🎯 Customer Cluster Analysis Demonstration")
    print("This script demonstrates customer clustering with console and web display options")
    
    while True:
        display_menu()
        
        try:
            choice = input("\nSelect an option (1-9): ").strip()
            
            if choice == '1':
                console_display_scenario_1()
            elif choice == '2':
                console_display_scenario_2()
            elif choice == '3':
                console_display_scenario_3()
            elif choice == '4':
                web_display_scenario_1()
            elif choice == '5':
                web_display_scenario_2()
            elif choice == '6':
                web_display_scenario_3()
            elif choice == '7':
                custom_clustering()
            elif choice == '8':
                compare_all_scenarios()
            elif choice == '9':
                print("\n👋 Thank you for using Customer Cluster Analysis!")
                print("Goodbye! 🎉")
                break
            else:
                print("\n❌ Invalid option. Please select 1-9.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Program interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            print("Please try again.")

if __name__ == "__main__":
    main()