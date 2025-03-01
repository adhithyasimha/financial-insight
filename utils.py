import pandas as pd

def read_excel_to_dataframe(file_path):
    try:
        # Read the Excel file into a DataFrame
        df = pd.read_excel(file_path)
        return df
    
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return None

