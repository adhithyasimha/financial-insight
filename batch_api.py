import pandas as pd
import requests
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
EXCEL_FILE_PATH = "data/companies.xlsx"
OUTPUT_DIR = "company_data"

# Function to read Excel file to DataFrame
def read_excel_to_dataframe(file_path):
    try:
        df = pd.read_excel(file_path)
        df.rename(columns={'companies': 'id', 'Unnamed: 1': 'company_name'}, inplace=True)
        df = df.iloc[1:].reset_index(drop=True)
        df.index += 1
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

# Function to process and save data to CSV
def save_to_csv(data, company_id, all_data):
    if data and isinstance(data, dict) and "data" in data:
        data_section = data["data"]
        
        # Extract required financial topics
        cashflow = data_section.get("cashflow", [])
        balancesheet = data_section.get("balancesheet", [])
        profitloss = data_section.get("profitandloss", [])
        
        # Convert to DataFrames
        cashflow_df = pd.DataFrame(cashflow).drop(columns=['id'], errors='ignore') if cashflow else pd.DataFrame(columns=['company_id', 'year'])
        balancesheet_df = pd.DataFrame(balancesheet).drop(columns=['id'], errors='ignore') if balancesheet else pd.DataFrame(columns=['company_id', 'year'])
        profitloss_df = pd.DataFrame(profitloss).drop(columns=['id'], errors='ignore') if profitloss else pd.DataFrame(columns=['company_id', 'year'])
        
        # Add company_id if missing (ensure consistency)
        for df in [cashflow_df, balancesheet_df, profitloss_df]:
            if not df.empty and 'company_id' not in df.columns:
                df['company_id'] = company_id
            elif df.empty:
                df['company_id'] = [company_id] * len(df)
                df['year'] = [None] * len(df)  # Placeholder to allow merging
        
        # Merge only if at least one DataFrame has data
        if not (cashflow_df.empty and balancesheet_df.empty and profitloss_df.empty):
            # Start with the first non-empty DataFrame
            merge_base = None
            for base_df in [cashflow_df, balancesheet_df, profitloss_df]:
                if not base_df.empty:
                    merge_base = base_df
                    break
            
            if merge_base is not None:
                for other_df in [cashflow_df, balancesheet_df, profitloss_df]:
                    if other_df is not merge_base and not other_df.empty:
                        merge_base = merge_base.merge(other_df, on=['company_id', 'year'], how='outer')
                all_data.append(merge_base)
            else:
                print(f"No valid data to merge for {company_id}")
        else:
            print(f"No financial data available for {company_id}")

# Main function to process companies and save to single CSV
def process_companies(excel_file_path, api_key, output_dir=OUTPUT_DIR):
    companies_df = read_excel_to_dataframe(excel_file_path)
    if companies_df is None or companies_df.empty:
        print("No companies found in the Excel file or file could not be read.")
        return

    os.makedirs(output_dir, exist_ok=True)
    total_companies = len(companies_df)
    print(f"Found {total_companies} companies in the Excel file.")

    all_data = []

    for index, row in companies_df.iterrows():
        if index == 0 and isinstance(row[0], str) and "company" in row[0].lower():
            continue

        company_id = str(row[0]).strip()
        print(f"Processing company {index}/{total_companies}: {company_id}")

        company_data = fetch_company_data(company_id, api_key)
        if company_data:
            save_to_csv(company_data, company_id, all_data)

        time.sleep(1)  # Avoid overwhelming the API

    # Combine all data into a single DataFrame and save to CSV
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        final_df.to_csv(f"{output_dir}/data.csv", index=False)
        print(f"All data saved to {output_dir}/data.csv")
    else:
        print("No valid data to save.")

# Example usage
if __name__ == "__main__":
    api_key = os.getenv('API') or "ghfkffu6378382826hhdjgk"  # Fallback if .env not set
    process_companies(EXCEL_FILE_PATH, api_key)