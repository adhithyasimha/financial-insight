import streamlit as st
import pandas as pd
import requests
import os

# Constants
EXCEL_FILE_PATH = "data/companies.xlsx"
API_URL = "https://stockticker.tech/server/api/company.php?id={}&api_key={}"
API_KEY = "ghfkffu6378382826hhdjgk"

def read_excel_to_dataframe(file_path):
    """Read the Excel file and return a DataFrame with cleaned column names."""
    try:
        df = pd.read_excel(file_path)

        # Rename columns
        df.rename(columns={'companies': 'id', 'Unnamed: 1': 'company_name'}, inplace=True)

        # Remove the first row containing column headers
        df = df.iloc[1:].reset_index(drop=True)

        # Start index from 1
        df.index += 1

        return df
    
    except Exception as e:
        st.error(f"❌ Error reading Excel file: {e}")
        return None

def find_company_name_column(df):
    """Find the column that likely contains company names."""
    possible_names = ['company_name', 'companyname', 'company', 'name', 'business', 'business_name', 
                      'organization', 'organisation', 'corp', 'corporation']
    
    df_columns_lower = [col.lower() for col in df.columns]
    
    for name in possible_names:
        for i, col in enumerate(df_columns_lower):
            if name in col:
                return df.columns[i]
    
    return df.columns[0] if df.columns else None

def find_company_id_column(df):
    """Find the column that likely contains company IDs."""
    possible_ids = ['id', 'company_id', 'cid', 'identifier', 'code']
    
    df_columns_lower = [col.lower() for col in df.columns]
    
    for id_name in possible_ids:
        for i, col in enumerate(df_columns_lower):
            if id_name in col:
                return df.columns[i]
    
    return None

def get_company_matches(dataframe, search_column, search_term, max_results=10):
    """Return company names that contain the search term."""
    if not search_term:
        return []
    
    search_term = search_term.lower()
    mask = dataframe[search_column].astype(str).str.lower().str.contains(search_term, na=False)
    matches = dataframe[mask][search_column].dropna().unique().tolist()
    
    return matches[:max_results]

def fetch_company_details(company_id):
    """Fetch company details from the API."""
    url = API_URL.format(company_id, API_KEY)
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"❌ Failed to fetch company details. Status code: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"❌ API request failed: {e}")
        return None

# Search page
def search_page():
    st.title("🔍 Company Search")

    if not os.path.exists(EXCEL_FILE_PATH):
        st.error(f"❌ Error: The companies database file '{EXCEL_FILE_PATH}' was not found.")
        return

    try:
        companies_df = read_excel_to_dataframe(EXCEL_FILE_PATH)
        if companies_df is None or companies_df.empty:
            st.error("❌ The Excel file is empty or could not be read. Please check the file.")
            return

        company_name_column = find_company_name_column(companies_df)
        company_id_column = find_company_id_column(companies_df)
        
        if not company_name_column:
            st.error("❌ Could not detect a company name column in the data.")
            return

        # Store DataFrame in session state for use across pages
        st.session_state.companies_df = companies_df
        st.session_state.company_name_column = company_name_column
        st.session_state.company_id_column = company_id_column

        # Search input
        search_term = st.text_input("🔎 Search for a company:")

        # Fetch matching companies
        matches = get_company_matches(companies_df, company_name_column, search_term)

        # Display clickable options
        if matches:
            st.subheader("📌 Select a company:")
            for company in matches:
                if st.button(company):
                    st.session_state.selected_company = company
                    st.session_state.page = "company_details"  # Switch to details page
                    st.rerun()  # Rerun the app to reflect page change

    except Exception as e:
        st.error(f"❌ Error: {e}")

def company_details_page():
    st.title("🏢 Company Details")
    
    if "selected_company" not in st.session_state:
        st.error("⚠️ No company selected. Please go back to search.")
        if st.button("🔙 Back to Search"):
            st.session_state.page = "search"
            st.rerun()
        return

    selected_company = st.session_state.selected_company
    companies_df = st.session_state.companies_df
    company_name_column = st.session_state.company_name_column
    company_id_column = st.session_state.company_id_column

    # Find the company row
    company_row = companies_df[companies_df[company_name_column] == selected_company].iloc[0]
    company_id = company_row[company_id_column] if company_id_column and company_id_column in company_row else "Not available"
    
    st.write(f"**📌 Company Name:** {selected_company}")
    st.write(f"**🆔 Company ID:** {company_id}")

    # Fetch API data
    if company_id != "Not available":
        company_data = fetch_company_details(company_id)
        if company_data and "company" in company_data:
            company_info = company_data["company"]

            # Extract and display logo
            logo_url = company_info.get("company_logo")
            if logo_url:
                st.image(logo_url, caption=selected_company, use_column_width=True)
            else:
                st.warning("⚠️ Company logo not found.")

    if st.button("🔙 Back to Search"):
        st.session_state.page = "search"
        st.rerun()


# Main app logic
def main():
    # Initialize page state
    if "page" not in st.session_state:
        st.session_state.page = "search"

    # Page routing
    if st.session_state.page == "search":
        search_page()
    elif st.session_state.page == "company_details":
        company_details_page()

if __name__ == "__main__":
    main()
