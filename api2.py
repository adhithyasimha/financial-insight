import streamlit as st
import pandas as pd
import os

# Path to the Excel file
EXCEL_FILE_PATH = "data/companies.xlsx"

def load_excel_to_dataframe(file_path):
    """Load the Excel file into a pandas DataFrame"""
    return pd.read_excel(file_path)

def find_company_name_column(df):
    """Find the column that likely contains company names"""
    possible_names = ['company_name', 'companyname', 'company', 'name', 'business', 'business_name', 
                      'organization', 'organisation', 'corp', 'corporation']
    
    df_columns_lower = [col.lower() for col in df.columns]
    
    for name in possible_names:
        for i, col in enumerate(df_columns_lower):
            if name in col:
                return df.columns[i]
    
    return df.columns[0] if df.columns else None

def get_company_matches(dataframe, search_column, search_term, max_results=10):
    """Return company names that contain the search term"""
    if not search_term:
        return []
    
    search_term = search_term.lower()
    mask = dataframe[search_column].astype(str).str.lower().str.contains(search_term, na=False)
    matches = dataframe[mask][search_column].dropna().unique().tolist()
    
    return matches[:max_results]  # Limit number of results

# Streamlit app
def main():
    st.title("Company Search")

    if not os.path.exists(EXCEL_FILE_PATH):
        st.error(f"Error: The companies database file '{EXCEL_FILE_PATH}' was not found.")
        return

    try:
        companies_df = load_excel_to_dataframe(EXCEL_FILE_PATH)
        if companies_df.empty:
            st.error("The Excel file is empty. Please check the file.")
            return

        company_name_column = find_company_name_column(companies_df)
        if not company_name_column:
            st.error("Could not detect a company name column in the data.")
            return

        # Search input
        search_term = st.text_input("Search for a company:")

        # Fetch matching companies
        matches = get_company_matches(companies_df, company_name_column, search_term)

        # Display clickable options
        if matches:
            st.subheader("Select a company:")
            for company in matches:
                if st.button(company):
                    st.session_state.selected_company = company  # Store selection in session state
        
        # Show details of selected company
        if "selected_company" in st.session_state:
            selected_company = st.session_state.selected_company
            st.success(f"Selected Company: {selected_company}")

    except Exception as e:
        st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
