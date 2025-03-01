import pandas as pd

def read_excel_to_dataframe(file_path):
    try:
        # Read the Excel file into a DataFrame
        df = pd.read_excel(file_path)

        # Rename columns
        df.rename(columns={'companies': 'id', 'Unnamed: 1': 'company_name'}, inplace=True)

        # Remove the first row containing column headers
        df = df.iloc[1:].reset_index(drop=True)

        # Start index from 1
        df.index += 1

        return df
    
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return None

# Example usage
file_path = 'data/companies.xlsx'
dataframe = read_excel_to_dataframe(file_path)

if dataframe is not None:
    print(dataframe.head())
else:
    print("Failed to read the Excel file.")
