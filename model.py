# Import necessary libraries
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import sqlite3
import os

# Load your dataset (adjust the path as needed)
data_path = "/kaggle/input/fin-data/data.csv"
df = pd.read_csv(data_path)

# Clean the data
df_cleaned = df.dropna(subset=['net_profit', 'operating_activity', 'equity_capital', 'sales'])
df_cleaned = df_cleaned.fillna(0)
df_cleaned = df_cleaned.drop_duplicates(subset=['company_id', 'year'], keep='first')

def parse_year(year_str):
    """Parse year strings in 'Month Year' (e.g., 'Dec 2012') or 'Year' (e.g., '2014') formats."""
    try:
        if ' ' in year_str:
            return datetime.strptime(year_str, '%b %Y')
        else:
            return datetime.strptime(year_str, '%Y')
    except ValueError:
        return None

def prepare_data(df):
    """Prepare data with features for modeling."""
    df = df.copy()
    
    # Parse and sort by year
    df['year_datetime'] = df['year'].apply(parse_year)
    df = df.dropna(subset=['year_datetime'])
    df = df.sort_values(['company_id', 'year_datetime'])
    
    # Add lagged features for each parameter
    for param in ['net_profit', 'operating_activity', 'equity_capital', 'sales']:
        df[f'{param}_lag1'] = df.groupby('company_id')[param].shift(1)
        df[f'{param}_lag2'] = df.groupby('company_id')[param].shift(2)
    
    # Convert numeric columns to float
    numeric_cols = df.columns.drop(['company_id', 'year', 'year_datetime'])
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Handle infinities and NaNs
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    return df

def train_and_evaluate_model(df, param, features):
    """Train a LightGBM model for a given parameter and evaluate its performance."""
    X = df[features].fillna(0)
    y = df[param]
    
    # Split data chronologically (80% train, 20% test)
    split_point = int(len(X) * 0.8)
    X_train, y_train = X.iloc[:split_point], y.iloc[:split_point]
    X_test, y_test = X.iloc[split_point:], y.iloc[split_point:]
    
    if len(X_test) < 1:
        print(f"Not enough data to train model for {param}")
        return None, None, None
    
    # Train LightGBM model with GPU support if available
    try:
        model = lgb.LGBMRegressor(objective='regression', random_state=42, device='gpu')
        model.fit(X_train, y_train)
    except:
        # Fallback to CPU if GPU is not available
        print(f"GPU training failed for {param}, falling back to CPU")
        model = lgb.LGBMRegressor(objective='regression', random_state=42)
        model.fit(X_train, y_train)
    
    # Make predictions and evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    metrics = {}
    metrics['mse'] = mean_squared_error(y_test, y_pred_test)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['r2'] = r2_score(y_test, y_pred_test)
    
    # Calculate MAPE if no zeros in actual values
    if not np.any(y_test == 0):
        metrics['mape'] = mean_absolute_percentage_error(y_test, y_pred_test) * 100
    else:
        # Custom MAPE calculation that handles zeros gracefully
        non_zero_mask = y_test != 0
        if np.any(non_zero_mask):
            mape = np.mean(np.abs((y_test[non_zero_mask] - y_pred_test[non_zero_mask]) / y_test[non_zero_mask])) * 100
            metrics['mape'] = mape
        else:
            metrics['mape'] = np.nan
    
    # Calculate mean absolute error
    metrics['mae'] = np.mean(np.abs(y_test - y_pred_test))
    
    # Print evaluation metrics
    print(f"\nModel for {param} evaluation:")
    print(f"  Test MSE: {metrics['mse']:.2f}")
    print(f"  Test RMSE: {metrics['rmse']:.2f}")
    print(f"  Test R²: {metrics['r2']:.4f}")
    if 'mape' in metrics and not np.isnan(metrics['mape']):
        print(f"  Test MAPE: {metrics['mape']:.2f}%")
    print(f"  Test MAE: {metrics['mae']:.2f}")
    
    # Add evaluation of scale
    data_range = y.max() - y.min()
    metrics['rmse_to_range_ratio'] = metrics['rmse'] / data_range if data_range > 0 else np.nan
    print(f"  RMSE/Range ratio: {metrics['rmse_to_range_ratio']:.4f}")
    
    # Give interpretation
    if metrics['r2'] > 0.8:
        print("  Interpretation: Excellent fit")
    elif metrics['r2'] > 0.6:
        print("  Interpretation: Good fit")
    elif metrics['r2'] > 0.4:
        print("  Interpretation: Moderate fit")
    else:
        print("  Interpretation: Poor fit, consider model improvements")
    
    # Print feature importance
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({'Feature': features, 'Importance': importance})
    feature_importance = feature_importance.sort_values('Importance', ascending=False).head(5)
    print("\n  Top 5 important features:")
    for i, row in feature_importance.iterrows():
        print(f"    {row['Feature']}: {row['Importance']:.4f}")
    
    return model, metrics, y_pred_test

def plot_predictions(company_df, models, predictions, company_id):
    """Plot actual vs predicted values for each parameter with different colors for predictions."""
    params = ['net_profit', 'operating_activity', 'equity_capital', 'sales']
    actual_colors = {'net_profit': 'blue', 'operating_activity': 'green', 
                    'equity_capital': 'red', 'sales': 'purple'}
    pred_colors = {'net_profit': 'lightblue', 'operating_activity': 'lightgreen', 
                  'equity_capital': 'salmon', 'sales': 'orchid'}
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, param in enumerate(params):
        if param in models and models[param] is not None:
            # Get prediction features
            X_company = company_df.drop(columns=[param, 'company_id', 'year', 'year_datetime'] + 
                                      [p for p in params if p != param]).fillna(0)
            
            # Get actual values and generate predictions
            y_actual = company_df[param]
            y_pred = models[param].predict(X_company)
            
            # Get years for plotting
            years = company_df['year'].tolist()
            
            # Plot on appropriate subplot
            ax = axes[i]
            
            # Plot actual data
            ax.plot(years, y_actual, label=f'Actual', 
                   color=actual_colors[param], marker='o', linewidth=2)
            
            # Plot predicted data
            ax.plot(years, y_pred, label=f'Predicted', 
                   color=pred_colors[param], marker='x', linestyle='--', linewidth=2)
            
            # Calculate and display R² on the plot
            r2 = np.corrcoef(y_actual, y_pred)[0, 1]**2
            ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes, 
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Format the plot
            ax.set_title(f"{param.replace('_', ' ').title()}")
            ax.set_xlabel("Year")
            ax.set_ylabel("Value")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            # Add some margins to prevent cutoff
            ax.margins(0.1)
    
    plt.suptitle(f"Financial Predictions for {company_id}", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    return fig

def forecast_future(company_df, models, periods=2):
    """Generate future forecasts for the next few periods."""
    params = ['net_profit', 'operating_activity', 'equity_capital', 'sales']
    
    # Get the last date in the dataset
    last_date = company_df['year_datetime'].max()
    
    # Initialize forecast dataframe with the last record
    last_record = company_df.iloc[-1:].copy()
    forecasts = [last_record]
    
    # Generate forecasts for specified periods
    for i in range(periods):
        # Create a new future record based on the previous one
        future_record = forecasts[-1].copy()
        
        # Increment date for next year
        next_date = datetime(last_date.year + i + 1, 1, 1)
        future_record['year_datetime'] = next_date
        future_record['year'] = next_date.strftime('%Y')
        
        # Update lagged features based on previous predictions
        for param in params:
            if param in models and models[param] is not None:
                # Update lag1 and lag2 features for next prediction
                future_record[f'{param}_lag1'] = future_record[param]
                future_record[f'{param}_lag2'] = future_record[f'{param}_lag1']
                
                # Prepare features for this parameter
                X_future = future_record.drop(columns=[param, 'company_id', 'year', 'year_datetime'] + 
                                            [p for p in params if p != param]).fillna(0)
                
                # Generate prediction
                future_record[param] = models[param].predict(X_future)[0]
        
        forecasts.append(future_record)
    
    # Combine all forecasts (excluding the first which is the last actual record)
    future_df = pd.concat(forecasts[1:], ignore_index=True)
    return future_df

def plot_with_forecast(company_df, models, future_df, company_id):
    """Plot historical data with future forecasts."""
    params = ['net_profit', 'operating_activity', 'equity_capital', 'sales']
    actual_colors = {'net_profit': 'blue', 'operating_activity': 'green', 
                    'equity_capital': 'red', 'sales': 'purple'}
    pred_colors = {'net_profit': 'lightblue', 'operating_activity': 'lightgreen', 
                  'equity_capital': 'salmon', 'sales': 'orchid'}
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, param in enumerate(params):
        if param in models and models[param] is not None:
            # Get prediction features
            X_company = company_df.drop(columns=[param, 'company_id', 'year', 'year_datetime'] + 
                                      [p for p in params if p != param]).fillna(0)
            
            # Get actual values and generate predictions
            y_actual = company_df[param]
            y_pred = models[param].predict(X_company)
            
            # Get years for plotting
            historical_years = company_df['year'].tolist()
            forecast_years = future_df['year'].tolist()
            all_years = historical_years + forecast_years
            
            # Get forecast values
            forecast_values = future_df[param].tolist()
            
            # Plot on appropriate subplot
            ax = axes[i]
            
            # Plot historical actual data
            ax.plot(historical_years, y_actual, label=f'Actual', 
                   color=actual_colors[param], marker='o', linewidth=2)
            
            # Plot historical predicted data
            ax.plot(historical_years, y_pred, label=f'Historical Predicted', 
                   color=pred_colors[param], marker='x', linestyle='--', linewidth=2)
            
            # Add vertical line to separate historical from forecast
            ax.axvline(x=historical_years[-1], color='gray', linestyle='-', alpha=0.5)
            
            # Plot forecast
            ax.plot(forecast_years, forecast_values, label=f'Forecast', 
                   color='black', marker='d', linestyle='-.', linewidth=2)
            
            # Add shaded area for confidence interval (simplified approach)
            # Here we use a fixed percentage range for illustration
            confidence = 0.1  # 10% confidence interval
            upper_bound = [val * (1 + confidence) for val in forecast_values]
            lower_bound = [val * (1 - confidence) for val in forecast_values]
            ax.fill_between(forecast_years, lower_bound, upper_bound, color='gray', alpha=0.2)
            
            # Format the plot
            ax.set_title(f"{param.replace('_', ' ').title()}")
            ax.set_xlabel("Year")
            ax.set_ylabel("Value")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            # Add some margins to prevent cutoff
            ax.margins(0.1)
    
    plt.suptitle(f"Financial Predictions and Forecast for {company_id}", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    return fig

def setup_database():
    """Create SQLite database and tables if they don't exist."""
    # Database file path
    db_path = "company_financial_data.db"
    
    # Connect to SQLite database (creates file if it doesn't exist)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create companies table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS companies (
        company_id TEXT PRIMARY KEY,
        first_year TEXT,
        last_year TEXT,
        years_count INTEGER,
        avg_net_profit REAL,
        avg_operating_activity REAL,
        avg_equity_capital REAL,
        avg_sales REAL,
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create forecasts table to store prediction results
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS forecasts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        company_id TEXT,
        forecast_year TEXT,
        net_profit REAL,
        operating_activity REAL,
        equity_capital REAL,
        sales REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (company_id) REFERENCES companies(company_id)
    )
    ''')
    
    # Create performance metrics table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS model_performance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        company_id TEXT,
        parameter TEXT,
        r2_score REAL,
        rmse REAL,
        mape REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (company_id) REFERENCES companies(company_id)
    )
    ''')
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    
    print(f"Database setup complete. Database file: {db_path}")
    return db_path

def store_company_data(db_path, company_id, company_df, future_df, metrics):
    """Store company data, forecasts, and model performance in SQLite database."""
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Extract company summary data
    params = ['net_profit', 'operating_activity', 'equity_capital', 'sales']
    
    # Calculate summary statistics
    summary = {
        'company_id': company_id,
        'first_year': company_df['year'].min(),
        'last_year': company_df['year'].max(),
        'years_count': len(company_df['year'].unique())
    }
    
    # Calculate averages for financial parameters
    for param in params:
        summary[f'avg_{param}'] = company_df[param].mean()
    
    # Store in companies table (UPSERT - insert or update if exists)
    try:
        cursor.execute('''
        INSERT INTO companies (company_id, first_year, last_year, years_count, 
                             avg_net_profit, avg_operating_activity, avg_equity_capital, avg_sales)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(company_id) DO UPDATE SET
            first_year = excluded.first_year,
            last_year = excluded.last_year,
            years_count = excluded.years_count,
            avg_net_profit = excluded.avg_net_profit,
            avg_operating_activity = excluded.avg_operating_activity,
            avg_equity_capital = excluded.avg_equity_capital,
            avg_sales = excluded.avg_sales,
            last_updated = CURRENT_TIMESTAMP
        ''', (
            summary['company_id'],
            summary['first_year'],
            summary['last_year'],
            summary['years_count'],
            summary['avg_net_profit'],
            summary['avg_operating_activity'],
            summary['avg_equity_capital'],
            summary['avg_sales']
        ))
        
        # Store forecast data
        # First delete existing forecasts for this company (to avoid duplicates)
        cursor.execute('''
        DELETE FROM forecasts WHERE company_id = ?
        ''', (company_id,))
        
        # Insert new forecasts
        for _, row in future_df.iterrows():
            cursor.execute('''
            INSERT INTO forecasts (company_id, forecast_year, net_profit, operating_activity, equity_capital, sales)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                company_id,
                row['year'],
                row['net_profit'],
                row['operating_activity'],
                row['equity_capital'],
                row['sales']
            ))
        
        # Store model performance metrics
        # First delete existing metrics for this company
        cursor.execute('''
        DELETE FROM model_performance WHERE company_id = ?
        ''', (company_id,))
        
        # Insert new metrics
        for param in params:
            if param in metrics and metrics[param] is not None:
                cursor.execute('''
                INSERT INTO model_performance (company_id, parameter, r2_score, rmse, mape)
                VALUES (?, ?, ?, ?, ?)
                ''', (
                    company_id,
                    param,
                    metrics[param].get('r2', None),
                    metrics[param].get('rmse', None),
                    metrics[param].get('mape', None)
                ))
        
        # Commit changes
        conn.commit()
        print(f"Successfully stored data for {company_id} in the database.")
        
    except Exception as e:
        conn.rollback()
        print(f"Error storing data in database: {e}")
    
    finally:
        # Close connection
        conn.close()

def list_stored_companies(db_path):
    """List all companies stored in the database with key statistics."""
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Query the companies table
        cursor.execute('''
        SELECT company_id, first_year, last_year, years_count, 
               avg_net_profit, avg_operating_activity, avg_equity_capital, avg_sales
        FROM companies
        ORDER BY company_id
        ''')
        
        # Fetch all results
        companies = cursor.fetchall()
        
        if companies:
            # Print as a formatted table
            print("\nStored Companies in Database:")
            print("-" * 100)
            print(f"{'ID':<10} {'First Year':<12} {'Last Year':<12} {'Years':<8} {'Avg Net Profit':<15} {'Avg Operating':<15} {'Avg Equity':<15} {'Avg Sales':<15}")
            print("-" * 100)
            
            for company in companies:
                print(f"{company[0]:<10} {company[1]:<12} {company[2]:<12} {company[3]:<8} {company[4]:<15.2f} {company[5]:<15.2f} {company[6]:<15.2f} {company[7]:<15.2f}")
            
            print("-" * 100)
        else:
            print("No companies found in the database.")
            
    except Exception as e:
        print(f"Error querying database: {e}")
    
    finally:
        # Close connection
        conn.close()

def get_company_details(db_path, company_id):
    """Retrieve and display detailed information for a specific company."""
    # Connect to database
    conn = sqlite3.connect(db_path)
    
    try:
        # Query company data
        company_df = pd.read_sql_query(
            "SELECT * FROM companies WHERE company_id = ?", 
            conn, 
            params=(company_id,)
        )
        
        if company_df.empty:
            print(f"No data found for company {company_id}")
            return
        
        # Query forecasts
        forecasts_df = pd.read_sql_query(
            "SELECT * FROM forecasts WHERE company_id = ? ORDER BY forecast_year", 
            conn, 
            params=(company_id,)
        )
        
        # Query model performance
        performance_df = pd.read_sql_query(
            "SELECT * FROM model_performance WHERE company_id = ?", 
            conn, 
            params=(company_id,)
        )
        
        # Display company details
        print(f"\n--- Detailed Information for {company_id} ---")
        print(f"Years of data: {company_df['first_year'].iloc[0]} to {company_df['last_year'].iloc[0]} ({company_df['years_count'].iloc[0]} years)")
        print(f"Last updated: {company_df['last_updated'].iloc[0]}")
        print("\nAverage Financial Metrics:")
        print(f"  Net Profit: {company_df['avg_net_profit'].iloc[0]:.2f}")
        print(f"  Operating Activity: {company_df['avg_operating_activity'].iloc[0]:.2f}")
        print(f"  Equity Capital: {company_df['avg_equity_capital'].iloc[0]:.2f}")
        print(f"  Sales: {company_df['avg_sales'].iloc[0]:.2f}")
        
        # Display forecasts
        if not forecasts_df.empty:
            print("\nForecasts:")
            for _, row in forecasts_df.iterrows():
                print(f"  Year {row['forecast_year']}:")
                print(f"    Net Profit: {row['net_profit']:.2f}")
                print(f"    Operating Activity: {row['operating_activity']:.2f}")
                print(f"    Equity Capital: {row['equity_capital']:.2f}")
                print(f"    Sales: {row['sales']:.2f}")
        
        # Display model performance
        if not performance_df.empty:
            print("\nModel Performance:")
            for _, row in performance_df.iterrows():
                print(f"  {row['parameter']}:")
                print(f"    R² Score: {row['r2_score']:.4f}")
                print(f"    RMSE: {row['rmse']:.2f}")
                if row['mape'] is not None:
                    print(f"    MAPE: {row['mape']:.2f}%")
                
    except Exception as e:
        print(f"Error retrieving company details: {e}")
    
    finally:
        # Close connection
        conn.close()


# Main execution
if __name__ == "__main__":
    # Set up the database
    db_path = setup_database()
    
    # Prepare the data
    prepared_df = prepare_data(df_cleaned)
    print("Sample of Prepared Data:")
    print(prepared_df.head())

    # Define features and parameters
    params = ['net_profit', 'operating_activity', 'equity_capital', 'sales']
    features = [col for col in prepared_df.columns if col not in params + ['company_id', 'year', 'year_datetime']]

    # Train models for each parameter
    models = {}
    metrics = {}
    test_predictions = {}
    
    for param in params:
        print(f"\nTraining model for {param}...")
        models[param], metrics[param], test_predictions[param] = train_and_evaluate_model(prepared_df, param, features)
        
    # Display data range for each parameter to put MSE in context
    print("\nData ranges to help interpret MSE values:")
    for param in params:
        data_range = prepared_df[param].max() - prepared_df[param].min()
        mean_value = prepared_df[param].mean()
        print(f"  {param}: Range = {data_range:.2f}, Mean = {mean_value:.2f}")
        if param in metrics and metrics[param] is not None:
            relative_error = np.sqrt(metrics[param]['mse']) / mean_value * 100 if mean_value != 0 else np.nan
            print(f"    RMSE as % of mean value: {relative_error:.2f}%")

    # Display companies already in the database
    list_stored_companies(db_path)
    
    # Get user input for company to analyze
    company_id = input("\nEnter company ID (e.g., ABB): ").strip().upper()
    company_df = prepared_df[prepared_df['company_id'] == company_id]
    
    if company_df.empty:
        print(f"No data found for {company_id}")
    else:
        # Plot predictions for historical data
        fig = plot_predictions(company_df, models, test_predictions, company_id)
        
        # Save the plot to file
        plot_file = f"{company_id}_predictions.png"
        fig.savefig(plot_file)
        plt.close(fig)
        print(f"Predictions plot saved to {plot_file}")
        
        # Generate future forecasts
        future_periods = 2  # Adjust as needed
        future_df = forecast_future(company_df, models, periods=future_periods)
        
        print(f"\nForecast for {company_id} for the next {future_periods} years:")
        print(future_df[['year'] + params])
        
        # Plot with forecast
        forecast_fig = plot_with_forecast(company_df, models, future_df, company_id)
        
        # Save the forecast plot to file
        forecast_plot_file = f"{company_id}_forecast.png"
        forecast_fig.savefig(forecast_plot_file)
        plt.close(forecast_fig)
        print(f"Forecast plot saved to {forecast_plot_file}")
        
        # Store company data in SQLite database
        store_company_data(db_path, company_id, company_df, future_df, metrics)
        
        # Display detailed information about the company from the database
        get_company_details(db_path, company_id)
        
        # Ask if user wants to see another company
        while True:
            another = input("\nWould you like to view another company? (y/n): ").strip().lower()
            if another == 'n':
                break
            elif another == 'y':
                # List companies in database
                list_stored_companies(db_path)
                
                # Get new company ID
                company_id = input("\nEnter company ID (e.g., ABB): ").strip().upper()
                company_df = prepared_df[prepared_df['company_id'] == company_id]
                
                if company_df.empty:
                    print(f"No data found for {company_id}")
                else:
                    # Process new company
                    fig = plot_predictions(company_df, models, test_predictions, company_id)
                    plt.close(fig)
                    
                    future_df = forecast_future(company_df, models, periods=future_periods)
                    forecast_fig = plot_with_forecast(company_df, models, future_df, company_id)
                    plt.close(forecast_fig)
                    
                    store_company_data(db_path, company_id, company_df, future_df, metrics)
                    get_company_details(db_path, company_id)
            else:
                print("Invalid input. Please enter 'y' or 'n'.")