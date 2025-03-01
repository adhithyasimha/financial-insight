import pandas as pd
import requests
import time
import os
from dotenv import load_dotenv
from utils import read_excel_to_dataframe


# Function to read Excel file to DataFrame
def read_excel_to_dataframe(file_path):
    try:
        df = pd.read_excel(file_path)
        return df
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return None

# Function to fetch data from API for a specific company
def fetch_company_data(company_id, api_key):
    base_url = "https://stockticker.tech/server/api/company.php"
    params = {"id": company_id, "api_key": api_key}

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {company_id}: {e}")
        return None

# Function to save all data in single CSV files
def save_to_csv(data, company_id, output_dir, all_dataframes):
    if data and isinstance(data, dict):
        if 'balance_sheet' in data:
            df_balance = pd.DataFrame(data['balance_sheet'])
            df_balance.insert(0, "Company_ID", company_id)  # Add Company ID column
            all_dataframes['balance'].append(df_balance)

        if 'profit_loss' in data:
            df_profit_loss = pd.DataFrame(data['profit_loss'])
            df_profit_loss.insert(0, "Company_ID", company_id)
            all_dataframes['profit_loss'].append(df_profit_loss)

        if 'cash_flow' in data:
            df_cash_flow = pd.DataFrame(data['cash_flow'])
            df_cash_flow.insert(0, "Company_ID", company_id)
            all_dataframes['cash_flow'].append(df_cash_flow)

        df_raw = pd.DataFrame([data])
        df_raw.insert(0, "Company_ID", company_id)
        all_dataframes['raw'].append(df_raw)

# Main function to process all companies
def process_companies(excel_file_path, api_key, output_dir="company_data"):
    companies_df = read_excel_to_dataframe(excel_file_path)
    if companies_df is None or companies_df.empty:
        print("No companies found in the Excel file or file could not be read.")
        return

    os.makedirs(output_dir, exist_ok=True)
    total_companies = len(companies_df)
    print(f"Found {total_companies} companies in the Excel file.")

    all_dataframes = {
        'balance': [],
        'profit_loss': [],
        'cash_flow': [],
        'raw': []
    }

    for index, row in companies_df.iterrows():
        if index == 0 and isinstance(row[0], str) and "company" in row[0].lower():
            continue

        company_id = str(row[0]).strip()
        print(f"Processing company {index+1}/{total_companies}: {company_id}")

        company_data = fetch_company_data(company_id, api_key)
        if company_data:
            save_to_csv(company_data, company_id, output_dir, all_dataframes)

        time.sleep(1)

    # Save all collected dataframes into single CSV files
    if all_dataframes['balance']:
        pd.concat(all_dataframes['balance']).to_csv(f"{output_dir}/all_balance_sheets.csv", index=False)

    if all_dataframes['profit_loss']:
        pd.concat(all_dataframes['profit_loss']).to_csv(f"{output_dir}/all_profit_loss.csv", index=False)

    if all_dataframes['cash_flow']:
        pd.concat(all_dataframes['cash_flow']).to_csv(f"{output_dir}/all_cash_flows.csv", index=False)

    if all_dataframes['raw']:
        pd.concat(all_dataframes['raw']).to_csv(f"{output_dir}/all_raw_data.csv", index=False)

    print("All companies processed successfully!")

# Example usage
if __name__ == "__main__":
    print(os.getenv('API_KEY')) 
    api_key = os.getenv('API_KEY')
    excel_file_path = "data/companies.xlsx"
    process_companies(excel_file_path, api_key)
