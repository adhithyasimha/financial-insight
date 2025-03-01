import streamlit as st
import pandas as pd
import requests
import os
from datetime import datetime

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

    company_row = companies_df[companies_df[company_name_column] == selected_company].iloc[0]
    company_id = company_row[company_id_column] if company_id_column and company_id_column in company_row else "Not available"
    
    st.write(f"**📌 Company Name:** {selected_company}")
    st.write(f"**🆔 Company ID:** {company_id}")

    if company_id != "Not available":
        company_data = fetch_company_details(company_id)
        if company_data and "company" in company_data:
            company_info = company_data["company"]

            logo_url = company_info.get("company_logo")
            if logo_url:
                st.image(logo_url, caption=selected_company, use_container_width=True)
            else:
                st.warning("⚠️ Company logo not found.")

            st.write(f"**ℹ️ About:** {company_info.get('about_company', 'Not available')}")
            st.write(f"**🌐 Website:** [{company_info.get('website', 'Not available')}]({company_info.get('website', 'Not available')})")
            st.write(f"**📊 NSE Profile:** [{company_info.get('nse_profile', 'Not available')}]({company_info.get('nse_profile', 'Not available')})")
            st.write(f"**📈 BSE Profile:** [{company_info.get('bse_profile', 'Not available')}]({company_info.get('bse_profile', 'Not available')})")
            st.write(f"**💰 Face Value:** {company_info.get('face_value', 'Not available')}")
            st.write(f"**📖 Book Value:** {company_info.get('book_value', 'Not available')}")
            st.write(f"**📈 ROCE:** {company_info.get('roce_percentage', 'Not available')}%")
            st.write(f"**📊 ROE:** {company_info.get('roe_percentage', 'Not available')}%")

            # Cash flow
            if "data" in company_data and "cashflow" in company_data["data"]:
                cashflow_data = company_data["data"]["cashflow"]
                cashflow_df = pd.DataFrame(cashflow_data)
                cashflow_df = cashflow_df.drop(columns=['id', 'company_id'], errors='ignore')
                
                def parse_year(year_str):
                    try:
                        return datetime.strptime(year_str, '%b %Y')
                    except ValueError:
                        return None  # For "TTM" or invalid formats
                cashflow_df['year_datetime'] = cashflow_df['year'].apply(parse_year)
                cashflow_df = cashflow_df.sort_values('year_datetime')
                cashflow_df_display = cashflow_df.drop(columns=['year_datetime'], errors='ignore')

                st.subheader("💸 Yearly Cash Flow")
                st.dataframe(cashflow_df_display)
                
                st.subheader("📉 Net Cash Flow Trend")
                cashflow_df['net_cash_flow'] = pd.to_numeric(cashflow_df['net_cash_flow'], errors='coerce').fillna(0)
                st.line_chart(cashflow_df[['year', 'net_cash_flow']].set_index('year'))

            # Pros and cons
            proscons_data = company_data.get("prosandcons") or (company_data.get("data", {}).get("prosandcons") if "data" in company_data else None)
            if proscons_data:
                proscons_df = pd.DataFrame(proscons_data)
                proscons_df = proscons_df.drop(columns=['id', 'company_id'], errors='ignore')
                proscons_df['cons'] = proscons_df['cons'].replace("NULL", "Not available")

                st.subheader("👍 Pros and Cons")
                st.dataframe(proscons_df.style.set_properties(**{
                    'text-align': 'left',
                    'white-space': 'pre-wrap'
                }))
            else:
                st.warning("⚠️ No Pros and Cons data available.")

            # Balance sheet
            balancesheet_data = company_data.get("balancesheet") or (company_data.get("data", {}).get("balancesheet") if "data" in company_data else None)
            if balancesheet_data:
                balancesheet_df = pd.DataFrame(balancesheet_data)
                balancesheet_df = balancesheet_df.drop(columns=['id', 'company_id'], errors='ignore')
                
                balancesheet_df['year_datetime'] = balancesheet_df['year'].apply(parse_year)
                balancesheet_df = balancesheet_df.sort_values('year_datetime')
                balancesheet_df_display = balancesheet_df.drop(columns=['year_datetime'], errors='ignore')

                st.subheader("📒 Balance Sheet")
                st.dataframe(balancesheet_df_display)
                
                st.subheader("📊 Total Assets Trend")
                balancesheet_df['total_assets'] = pd.to_numeric(balancesheet_df['total_assets'], errors='coerce').fillna(0)
                st.line_chart(balancesheet_df[['year', 'total_assets']].set_index('year'))
            else:
                st.warning("⚠️ No Balance Sheet data available.")

            # Profit and loss
            profitloss_data = company_data.get("profitandloss") or (company_data.get("data", {}).get("profitandloss") if "data" in company_data else None)
            if profitloss_data:
                profitloss_df = pd.DataFrame(profitloss_data)
                profitloss_df = profitloss_df.drop(columns=['id', 'company_id'], errors='ignore')
                
                profitloss_df['year_datetime'] = profitloss_df['year'].apply(parse_year)
                profitloss_df = profitloss_df.sort_values('year_datetime')
                profitloss_df_display = profitloss_df.drop(columns=['year_datetime'], errors='ignore')

                st.subheader("📈 Profit and Loss")
                st.dataframe(profitloss_df_display)
                
                st.subheader("📊 Net Profit Trend")
                profitloss_df['net_profit'] = pd.to_numeric(profitloss_df['net_profit'], errors='coerce').fillna(0)
                st.line_chart(profitloss_df[['year', 'net_profit']].set_index('year'))
            else:
                st.warning("⚠️ No Profit and Loss data available.")

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
