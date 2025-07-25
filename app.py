import streamlit as st
import pandas as pd
import numpy as np
import io
import re
from datetime import datetime

st.set_page_config(
    page_title="Portfolio Analyzer",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Portfolio Analyzer")

def clean_numeric(value):
    if pd.isna(value) or value is None:
        return 0.0
    
    if isinstance(value, (int, float)):
        return float(value)
    
    str_val = str(value).strip()
    
    if not str_val or str_val.lower() in ['nan', 'none', '']:
        return 0.0
    
    is_negative = str_val.startswith('(') and str_val.endswith(')')
    if is_negative:
        str_val = str_val[1:-1]
    
    str_val = re.sub(r'[^\d.-]', '', str_val)
    
    if not str_val:
        return 0.0
    
    try:
        numeric_val = float(str_val)
        return -numeric_val if is_negative else numeric_val
    except ValueError:
        return 0.0

def find_account_sections(df):
    account_patterns = [
        'Individual_401', 'Contributory', 'Joint_Tenant', 'Individual',
        'Account Total', 'Cash & Cash'
    ]
    
    account_sections = []
    current_account = "Unknown"
    
    for idx, row in df.iterrows():
        first_col = str(row.iloc[0]) if pd.notna(row.iloc[0]) else ""
        
        for pattern in account_patterns:
            if pattern.lower() in first_col.lower():
                current_account = first_col
                break
        
        if len(first_col) > 0 and not any(word in first_col.lower() for word in ['account total', 'total']):
            if 'cash' in first_col.lower() or any(word in first_col.lower() for word in ['investments', 'sweep', 'core']):
                account_sections.append({
                    'row_index': idx,
                    'account': current_account,
                    'symbol': first_col,
                    'row_data': row
                })
            elif 'cash' not in first_col.lower():
                account_sections.append({
                    'row_index': idx,
                    'account': current_account,
                    'symbol': first_col,
                    'row_data': row
                })
    
    return account_sections

def create_aggregated_positions(sections):
    positions = []
    
    for section in sections:
        row_data = section['row_data']
        
        try:
            position = {
                'Account': section['account'],
                'Symbol': str(row_data.iloc[0]) if pd.notna(row_data.iloc[0]) else '',
                'Description': str(row_data.iloc[1]) if len(row_data) > 1 and pd.notna(row_data.iloc[1]) else '',
                'Quantity': clean_numeric(row_data.iloc[2]) if len(row_data) > 2 else 0,
                'Price': clean_numeric(row_data.iloc[3]) if len(row_data) > 3 else 0,
                'Market_Value': clean_numeric(row_data.iloc[6]) if len(row_data) > 6 else 0,
                'Day_Change_Dollar': clean_numeric(row_data.iloc[7]) if len(row_data) > 7 else 0,
                'Day_Change_Percent': clean_numeric(row_data.iloc[8]) if len(row_data) > 8 else 0,
            }
            
            if position['Symbol'] and position['Market_Value'] != 0:
                positions.append(position)
                
        except Exception:
            continue
    
    df = pd.DataFrame(positions)
    
    if len(df) > 0:
        numeric_columns = ['Quantity', 'Price', 'Market_Value', 'Day_Change_Dollar', 'Day_Change_Percent']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        text_columns = ['Account', 'Symbol', 'Description']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str)
    
    return df

def classify_sector(symbol, description=""):
    symbol = symbol.upper()
    desc_upper = description.upper()
    
    # Cash detection
    cash_indicators = ['CASH', 'MONEY', 'SWEEP', 'SETTLEMENT', 'CORE', 'INVESTMENTS']
    
    if 'CASH & CASH INVESTMENTS' in desc_upper or any(indicator in symbol for indicator in cash_indicators) or any(indicator in desc_upper for indicator in cash_indicators):
        return 'Cash & Cash Equivalents'
    elif 'ETF' in desc_upper or 'FUND' in desc_upper:
        return 'ETF'
    else:
        return 'Other'

# File upload
uploaded_file = st.file_uploader(
    "Choose your portfolio CSV or Excel file",
    type=['csv', 'xlsx', 'xls']
)

if uploaded_file is not None:
    try:
        # Load file
        if uploaded_file.name.lower().endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.success(f"✅ File loaded successfully! Shape: {df.shape}")
        
        # Process data
        with st.spinner("Processing portfolio data..."):
            sections = find_account_sections(df)
            agg_df = create_aggregated_positions(sections)
            
            if len(agg_df) > 0:
                agg_df = agg_df[agg_df['Symbol'] != '']
                agg_df = agg_df[agg_df['Market_Value'] > 0]
                
                # Add sector classification
                agg_df['Sector'] = agg_df.apply(lambda row: classify_sector(row['Symbol'], row['Description']), axis=1)
                
                # Create consolidated view
                symbol_agg = agg_df.groupby('Symbol').agg({
                    'Market_Value': 'sum',
                    'Quantity': 'sum', 
                    'Day_Change_Dollar': 'sum',
                    'Account': lambda x: ', '.join(sorted(set(x))),
                    'Description': 'first',
                    'Sector': 'first'
                }).reset_index()
                
                symbol_agg['Account_Count'] = symbol_agg['Account'].apply(lambda x: len(x.split(', ')))
                symbol_agg = symbol_agg.sort_values('Market_Value', ascending=False)
                
                # Calculate metrics
                total_value = symbol_agg['Market_Value'].sum()
                cash_positions = symbol_agg[symbol_agg['Sector'] == 'Cash & Cash Equivalents']
                cash_value = cash_positions['Market_Value'].sum()
                cash_percentage = (cash_value / total_value * 100) if total_value > 0 else 0
                
                symbol_agg['Weight_Pct'] = (symbol_agg['Market_Value'] / total_value * 100).round(2)
                
                # Display key metrics
                st.header("📊 Portfolio Overview")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Portfolio Value", f"${total_value:,.0f}")
                
                with col2:
                    st.metric("Unique Positions", f"{len(symbol_agg)}")
                
                with col3:
                    largest_weight = symbol_agg.iloc[0]['Weight_Pct']
                    largest_symbol = symbol_agg.iloc[0]['Symbol']
                    st.metric("Largest Position", f"{largest_weight:.1f}%", f"{largest_symbol}")
                
                with col4:
                    st.metric("Cash Allocation", f"{cash_percentage:.1f}%")
                
                # Display cash positions if detected
                if len(cash_positions) > 0:
                    with st.expander("💰 Cash Positions Detected"):
                        for _, pos in cash_positions.iterrows():
                            st.write(f"**{pos['Symbol']}**: ${pos['Market_Value']:,.2f} ({pos['Weight_Pct']:.1f}%)")
                
                # Top Holdings
                st.header("🏆 Top 10 Holdings")
                top_10 = symbol_agg.head(10)[['Symbol', 'Description', 'Market_Value', 'Weight_Pct', 'Sector']]
                top_10.columns = ['Symbol', 'Description', 'Market Value', 'Weight %', 'Sector']
                top_10['Market Value'] = top_10['Market Value'].apply(lambda x: f"${x:,.0f}")
                top_10['Weight %'] = top_10['Weight %'].apply(lambda x: f"{x:.1f}%")
                st.dataframe(top_10, use_container_width=True)
                
                # Sector Analysis
                st.header("🏭 Sector Analysis")
                sector_summary = agg_df.groupby('Sector')['Market_Value'].sum().sort_values(ascending=False)
                sector_pcts = (sector_summary / total_value * 100).round(1)
                
                sector_df = pd.DataFrame({
                    'Sector': sector_summary.index,
                    'Value': [f"${x:,.0f}" for x in sector_summary.values],
                    'Percentage': [f"{x:.1f}%" for x in sector_pcts.values]
                })
                st.dataframe(sector_df, use_container_width=True)
                
                # Download Excel
                st.header("📥 Download Analysis")
                
                # Create Excel file
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    # Summary sheet
                    summary_data = {
                        'Metric': ['Total Portfolio Value', 'Unique Positions', 'Cash Percentage', 'Largest Position'],
                        'Value': [f"${total_value:,.2f}", len(symbol_agg), f"{cash_percentage:.1f}%", f"{largest_symbol} ({largest_weight:.1f}%)"]
                    }
                    pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
                    
                    # Consolidated positions
                    symbol_agg.to_excel(writer, sheet_name='Consolidated_Positions', index=False)
                    
                    # Individual positions
                    agg_df.to_excel(writer, sheet_name='Individual_Positions', index=False)
                
                excel_data = output.getvalue()
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M')
                filename = f"portfolio_analysis_{timestamp}.xlsx"
                
                st.download_button(
                    label="📥 Download Excel Analysis",
                    data=excel_data,
                    file_name=filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
                st.success("✅ Analysis complete! Your portfolio has been processed successfully.")
                
            else:
                st.error("❌ No valid positions found in the uploaded file")
                
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.error("Please check that your file format matches the expected broker export format.")

else:
    st.info("👆 Please upload your portfolio CSV or Excel file to begin analysis")

st.markdown("---")
st.markdown("**Portfolio Analyzer** - Upload your broker's export file to get instant analysis")
