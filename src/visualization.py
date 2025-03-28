import matplotlib.pyplot as plt
import pandas as pd
import os

def generate_visualizations():
    """Generate key visualizations from the data"""
    # Load data
    data = pd.read_csv('data/raw/telecom_churn.csv')
    
    # Create plots directory
    os.makedirs('frontend/static/plots', exist_ok=True)
    plot_paths = []
    
    # 1. Churn by International Plan
    plt.figure(figsize=(8, 6))
    data.groupby('International plan')['Churn'].mean().plot(kind='bar')
    plt.title('Churn Rate by International Plan')
    plt.ylabel('Churn Rate')
    intl_plot_path = 'frontend/static/plots/churn_by_intl_plan.png'
    plt.savefig(intl_plot_path)
    plt.close()
    plot_paths.append(intl_plot_path)
    
    # 2. Customer Service Calls vs Churn
    plt.figure(figsize=(8, 6))
    data['Customer service calls'].hist(by=data['Churn'], bins=8, alpha=0.5)
    plt.suptitle('Customer Service Calls Distribution by Churn Status')
    plt.xlabel('Number of Customer Service Calls')
    service_plot_path = 'frontend/static/plots/service_calls_dist.png'
    plt.savefig(service_plot_path)
    plt.close()
    plot_paths.append(service_plot_path)
    
    # 3. Day Minutes vs Churn
    plt.figure(figsize=(8, 6))
    data.boxplot(column='Total day minutes', by='Churn')
    plt.title('Day Minutes Distribution by Churn Status')
    plt.suptitle('')
    plt.ylabel('Total Day Minutes')
    minutes_plot_path = 'frontend/static/plots/day_minutes_boxplot.png'
    plt.savefig(minutes_plot_path)
    plt.close()
    plot_paths.append(minutes_plot_path)
    
    return plot_paths