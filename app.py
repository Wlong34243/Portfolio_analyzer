"""
Portfolio Analyzer Pro - Streamlit Web App
Professional portfolio analysis with risk assessment and options strategies
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import re
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Portfolio Analyzer Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .success-box {
        background: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analyzed_data' not in st.session_state:
    st.session_state.analyzed_data = None
if 'show_analysis' not in st.session_state:
    st.session_state.show_analysis = False

# Utility functions
@st.cache_data
def clean_numeric(value):
    """Clean numeric values - remove $ signs, commas, parentheses"""
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
    """Find where each account section starts"""
    account_patterns = [
        'Individual_401', 'Contributory', 'Joint_Tenant', 'Individual',
        'Account Total'
    ]
    
    account_sections = []
    current_account = "Unknown"
    
    for idx, row in df.iterrows():
        first_col = str(row.iloc[0]) if pd.notna(row.iloc[0]) else ""
        
        # Check for account headers
        for pattern in account_patterns:
            if pattern.lower() in first_col.lower():
                current_account = first_col
                break
        
        # Include positions AND cash positions
        if len(first_col) > 0 and not any(word in first_col.lower() for word in ['account', 'total']):
            # Include both regular positions and cash positions
            if (first_col.isalpha() and len(first_col) <= 6) or 'cash & cash investments' in first_col.lower():
                account_sections.append({
                    'row_index': idx,
                    'account': current_account,
                    'symbol': first_col,
                    'row_data': row
                })
    
    return account_sections

def create_aggregated_positions(sections, original_df):
    """Create a clean aggregated positions DataFrame"""
    positions = []
    
    for section in sections:
        row_data = section['row_data']
        symbol = section['symbol']
        
        try:
            # Handle cash positions specially
            if 'cash & cash investments' in symbol.lower():
                position = {
                    'Account': section['account'],
                    'Symbol': 'CASH',
                    'Description': 'Cash & Cash Investments',
                    'Quantity': 1,
                    'Price': clean_numeric(row_data.iloc[6]) if len(row_data) > 6 else 0,
                    'Market_Value': clean_numeric(row_data.iloc[6]) if len(row_data) > 6 else 0,
                    'Day_Change_Dollar': clean_numeric(row_data.iloc[7]) if len(row_data) > 7 else 0,
                    'Day_Change_Percent': clean_numeric(row_data.iloc[8]) if len(row_data) > 8 else 0,
                }
            else:
                # Handle regular securities
                position = {
                    'Account': section['account'],
                    'Symbol': symbol,
                    'Description': str(row_data.iloc[1]) if len(row_data) > 1 and pd.notna(row_data.iloc[1]) else '',
                    'Quantity': clean_numeric(row_data.iloc[2]) if len(row_data) > 2 else 0,
                    'Price': clean_numeric(row_data.iloc[3]) if len(row_data) > 3 else 0,
                    'Market_Value': clean_numeric(row_data.iloc[6]) if len(row_data) > 6 else 0,
                    'Day_Change_Dollar': clean_numeric(row_data.iloc[7]) if len(row_data) > 7 else 0,
                    'Day_Change_Percent': clean_numeric(row_data.iloc[8]) if len(row_data) > 8 else 0,
                }
            
            # Only include positions with market value
            if position['Market_Value'] != 0:
                positions.append(position)
                
        except Exception as e:
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
    """Classify securities by sector"""
    tech_symbols = ['AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA', 'CRM', 'ADBE', 'AMD', 'INTC', 'CSCO', 'PANW', 'ZS', 'DELL', 'AVGO']
    energy_symbols = ['XOM', 'ET', 'EPD', 'NEE', 'COP', 'SLB']
    health_symbols = ['UNH', 'JNJ', 'PFE']
    financial_symbols = ['KEY', 'BX', 'ARCC']
    reit_symbols = ['O', 'IIPR', 'ADC']
    comm_symbols = ['VZ', 'DIS']
    industrial_symbols = ['MMM', 'ETN', 'HMC']
    consumer_symbols = ['BABA']
    crypto_symbols = ['GBTC']
    
    etf_patterns = ['ETF', 'FUND', 'INDEX', 'SPDR', 'VANGUARD', 'ISHARES', 'INVESCO', 'SCHWAB', 'SELECT']
    
    symbol = symbol.upper()
    desc_upper = description.upper()
    
    # Handle cash specially
    if symbol == 'CASH':
        return 'Cash & Equivalents'
    
    if any(pattern in desc_upper for pattern in etf_patterns) or symbol in ['VTI', 'VEA', 'VEU', 'XLF', 'SCHD', 'IFRA', 'OIH', 'PPA']:
        if any(word in desc_upper for word in ['FINANCIAL', 'BANK']):
            return 'ETF-Financial'
        elif any(word in desc_upper for word in ['TECH', 'NASDAQ', 'QQQ']):
            return 'ETF-Technology'
        elif any(word in desc_upper for word in ['ENERGY', 'OIL']):
            return 'ETF-Energy'
        elif any(word in desc_upper for word in ['INFRAST', 'UTILITY']):
            return 'ETF-Infrastructure'
        elif any(word in desc_upper for word in ['DIVIDEND', 'INCOME']):
            return 'ETF-Dividend'
        elif any(word in desc_upper for word in ['INTERNATIONAL', 'WORLD', 'DEVELOPED']):
            return 'ETF-International'
        elif any(word in desc_upper for word in ['DEFENSE', 'AEROSPACE']):
            return 'ETF-Defense'
        else:
            return 'ETF-Broad Market'
    
    if symbol in tech_symbols:
        return 'Technology'
    elif symbol in energy_symbols:
        return 'Energy'
    elif symbol in health_symbols:
        return 'Healthcare'
    elif symbol in financial_symbols:
        return 'Financial Services'
    elif symbol in reit_symbols:
        return 'Real Estate'
    elif symbol in comm_symbols:
        return 'Communication'
    elif symbol in industrial_symbols:
        return 'Industrial'
    elif symbol in consumer_symbols:
        return 'Consumer Discretionary'
    elif symbol in crypto_symbols:
        return 'Cryptocurrency'
    else:
        return 'Other'

def calculate_portfolio_metrics(consolidated_df, total_portfolio_value):
    """Calculate comprehensive portfolio metrics using consolidated positions"""
    
    metrics = {
        'total_portfolio_value': total_portfolio_value,
        'position_count': len(consolidated_df),
        'average_position_size': total_portfolio_value / len(consolidated_df) if len(consolidated_df) > 0 else 0,
        'largest_position': consolidated_df['Market_Value'].max(),
        'smallest_position': consolidated_df['Market_Value'].min(),
        'top_10_concentration': consolidated_df.nlargest(10, 'Market_Value')['Market_Value'].sum() / total_portfolio_value * 100,
        'top_5_concentration': consolidated_df.nlargest(5, 'Market_Value')['Market_Value'].sum() / total_portfolio_value * 100,
        'single_largest_weight': consolidated_df['Market_Value'].max() / total_portfolio_value * 100,
        'cash_percentage': consolidated_df[consolidated_df['Symbol'] == 'CASH']['Market_Value'].sum() / total_portfolio_value * 100
    }
    
    return metrics

def analyze_options_opportunities(symbol_agg):
    """Analyze options opportunities for large positions"""
    large_positions = symbol_agg[(symbol_agg['Consolidated_Weight'] > 3) & (symbol_agg['Symbol'] != 'CASH')].copy()
    options_data = []
    
    for _, position in large_positions.head(10).iterrows():
        symbol = position['Symbol']
        
        if len(symbol) <= 4 and symbol.isalpha():
            position_value = position['Market_Value']
            weight = position['Consolidated_Weight']
            
            estimated_price = position_value / position.get('Quantity', 1) if position.get('Quantity', 0) > 0 else 100
            total_shares = position.get('Quantity', position_value / estimated_price)
            round_lots = max(1, int(total_shares / 100))
            
            estimated_premium = estimated_price * 0.02  # 2% estimate
            monthly_yield = (estimated_premium * round_lots * 100) / position_value * 100
            
            options_data.append({
                'Symbol': symbol,
                'Strategy': 'Covered Call',
                'Position_Value': position_value,
                'Position_Weight_%': weight,
                'Est_Monthly_Yield_%': monthly_yield,
                'Est_Annual_Yield_%': monthly_yield * 12,
                'Round_Lots': round_lots
            })
    
    return pd.DataFrame(options_data) if options_data else pd.DataFrame()

def create_excel_workbook(agg_df, symbol_agg, metrics, sector_analysis, options_df=None):
    """Create comprehensive Excel workbook"""
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Portfolio Summary
        summary_data = {
            'Metric': [
                'Total Portfolio Value',
                'Total Consolidated Positions', 
                'Average Position Size',
                'Largest Position Value',
                'Largest Position %',
                'Top 5 Concentration %',
                'Top 10 Concentration %',
                'Cash Percentage %'
            ],
            'Value': [
                f"${metrics['total_portfolio_value']:,.2f}",
                metrics['position_count'],
                f"${metrics['average_position_size']:,.2f}",
                f"${metrics['largest_position']:,.2f}",
                f"{metrics['single_largest_weight']:.2f}%",
                f"{metrics['top_5_concentration']:.2f}%",
                f"{metrics['top_10_concentration']:.2f}%",
                f"{metrics['cash_percentage']:.2f}%"
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Portfolio_Summary', index=False)
        
        # Individual and consolidated positions
        agg_df.sort_values('Market_Value', ascending=False).to_excel(writer, sheet_name='Individual_Positions', index=False)
        symbol_agg.to_excel(writer, sheet_name='Consolidated_Positions', index=False)
        
        # Sector analysis
        sector_export = sector_analysis.copy()
        sector_export['Sector'] = sector_export.index
        sector_export = sector_export[['Sector', 'Market_Value', 'Percentage']]
        sector_export.to_excel(writer, sheet_name='Sector_Analysis', index=False)
        
        # Options opportunities
        if options_df is not None and not options_df.empty:
            options_df.to_excel(writer, sheet_name='Options_Opportunities', index=False)
    
    output.seek(0)
    return output

# Main application
def main():
    # Header
    st.markdown('<div class="main-header">📊 Portfolio Analyzer Pro</div>', unsafe_allow_html=True)
    st.markdown("### Professional Portfolio Analysis & Options Strategy Tool")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Upload Portfolio Data")
        uploaded_file = st.file_uploader(
            "Choose your portfolio CSV file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload your positions file from your broker"
        )
        
        st.markdown("---")
        st.header("🎯 Analysis Options")
        
        include_options = st.checkbox("Include Options Analysis", value=True)
        include_risk = st.checkbox("Include Risk Analysis", value=True)
        show_charts = st.checkbox("Show Interactive Charts", value=True)
        
        st.markdown("---")
        st.info("📊 **Features:**\n- Complete portfolio aggregation\n- Sector analysis\n- Risk assessment\n- Options opportunities\n- Excel export")
    
    # Main content
    if uploaded_file is not None:
        try:
            # Load and process file
            with st.spinner("🔄 Processing your portfolio data..."):
                # Read file
                if uploaded_file.name.lower().endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                # Process data
                sections = find_account_sections(df)
                agg_df = create_aggregated_positions(sections, df)
                
                if len(agg_df) > 0:
                    # Filter and clean data
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
                    
                    # Calculate total portfolio value from consolidated positions
                    total_portfolio_value = symbol_agg['Market_Value'].sum()
                    
                    # Calculate metrics using consolidated positions
                    metrics = calculate_portfolio_metrics(symbol_agg, total_portfolio_value)
                    
                    # Add position weights
                    agg_df['Position_Weight'] = (agg_df['Market_Value'] / total_portfolio_value * 100).round(2)
                    symbol_agg['Consolidated_Weight'] = (symbol_agg['Market_Value'] / total_portfolio_value * 100).round(2)
                    symbol_agg['Account_Count'] = symbol_agg['Account'].apply(lambda x: len(x.split(', ')))
                    symbol_agg = symbol_agg.sort_values('Market_Value', ascending=False)
                    
                    # Store in session state
                    st.session_state.analyzed_data = {
                        'agg_df': agg_df,
                        'symbol_agg': symbol_agg,
                        'metrics': metrics,
                        'total_portfolio_value': total_portfolio_value
                    }
                    st.session_state.show_analysis = True
            
            # Success message
            st.markdown('<div class="success-box">✅ <strong>Analysis Complete!</strong> Your portfolio has been successfully processed.</div>', unsafe_allow_html=True)
            
            # Display results
            if st.session_state.show_analysis and st.session_state.analyzed_data:
                data = st.session_state.analyzed_data
                agg_df = data['agg_df']
                symbol_agg = data['symbol_agg']
                metrics = data['metrics']
                
                # Portfolio Summary
                st.header("📊 Portfolio Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Total Value",
                        f"${metrics['total_portfolio_value']:,.0f}",
                        delta=f"{symbol_agg['Day_Change_Dollar'].sum():+,.0f}"
                    )
                
                with col2:
                    cash_value = symbol_agg[symbol_agg['Symbol'] == 'CASH']['Market_Value'].sum()
                    st.metric(
                        "Cash & Equivalents",
                        f"${cash_value:,.0f}",
                        delta=f"{metrics['cash_percentage']:.1f}%"
                    )
                
                with col3:
                    largest_position = symbol_agg.iloc[0]
                    st.metric(
                        "Largest Position",
                        f"{largest_position['Symbol']}",
                        delta=f"{metrics['single_largest_weight']:.1f}%"
                    )
                
                with col4:
                    st.metric(
                        "Total Positions",
                        f"{metrics['position_count']}",
                        delta=f"Top 5: {metrics['top_5_concentration']:.1f}%"
                    )
                
                st.markdown("---")
                
                # Top Holdings
                st.header("🏆 Top Holdings")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    top_10 = symbol_agg.head(10)
                    st.dataframe(
                        top_10[['Symbol', 'Description', 'Market_Value', 'Consolidated_Weight', 'Account_Count']].style.format({
                            'Market_Value': '${:,.0f}',
                            'Consolidated_Weight': '{:.2f}%'
                        }),
                        height=400
                    )
                
                with col2:
                    if show_charts and len(top_10) > 0:
                        try:
                            # Top 10 pie chart
                            fig_pie = px.pie(
                                top_10, 
                                values='Market_Value', 
                                names='Symbol',
                                title="Top 10 Holdings"
                            )
                            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                            fig_pie.update_layout(height=400)
                            st.plotly_chart(fig_pie, use_container_width=True)
                        except Exception as chart_error:
                            st.warning(f"Chart display issue: {chart_error}")
                            st.info("Data is still available in the table.")
                
                # Sector Analysis
                st.header("🏭 Sector Allocation")
                
                sector_analysis = symbol_agg.groupby('Sector').agg({'Market_Value': 'sum'}).round(2)
                sector_analysis['Percentage'] = (sector_analysis['Market_Value'] / total_portfolio_value * 100).round(1)
                sector_analysis = sector_analysis.sort_values('Market_Value', ascending=False)
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.dataframe(
                        sector_analysis.style.format({
                            'Market_Value': '${:,.0f}',
                            'Percentage': '{:.1f}%'
                        }),
                        height=400
                    )
                
                with col2:
                    if show_charts and not sector_analysis.empty:
                        try:
                            # Create sector allocation chart using Plotly
                            sector_chart_data = sector_analysis.reset_index()
                            
                            fig_sector = px.bar(
                                sector_chart_data, 
                                x='Sector', 
                                y='Market_Value',
                                title="Portfolio by Sector",
                                color='Market_Value',
                                color_continuous_scale='Blues',
                                labels={'Market_Value': 'Market Value ($)', 'Sector': 'Sector'}
                            )
                            
                            fig_sector.update_layout(
                                xaxis_tickangle=45,
                                height=400,
                                showlegend=False
                            )
                            
                            st.plotly_chart(fig_sector, use_container_width=True)
                            
                        except Exception as chart_error:
                            st.warning(f"Chart display issue: {chart_error}")
                            st.info("Data is still available in the table above.")
                
                # Options Analysis
                if include_options:
                    st.header("📊 Options Opportunities")
                    
                    options_df = analyze_options_opportunities(symbol_agg)
                    
                    if not options_df.empty:
                        st.dataframe(
                            options_df.style.format({
                                'Position_Value': '${:,.0f}',
                                'Position_Weight_%': '{:.2f}%',
                                'Est_Monthly_Yield_%': '{:.2f}%',
                                'Est_Annual_Yield_%': '{:.2f}%'
                            }),
                            height=300
                        )
                        
                        st.info("💡 **Options Strategy Tips:**\n- Use covered calls on positions >5% for income generation\n- Consider protective puts for concentrated holdings >10%\n- Monitor time decay and roll positions before expiration")
                    else:
                        st.info("No significant positions (>3%) suitable for options strategies found.")
                
                # Risk Analysis
                if include_risk:
                    st.header("⚠️ Risk Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Concentration Risks")
                        concentrated = symbol_agg[symbol_agg['Consolidated_Weight'] > 5]
                        
                        if not concentrated.empty:
                            for _, pos in concentrated.iterrows():
                                severity = "🔴 High" if pos['Consolidated_Weight'] > 10 else "🟡 Medium"
                                st.write(f"{severity}: **{pos['Symbol']}** - {pos['Consolidated_Weight']:.1f}% of portfolio")
                        else:
                            st.success("✅ No significant concentration risks detected")
                    
                    with col2:
                        st.subheader("Consolidation Opportunities")
                        multi_account = symbol_agg[symbol_agg['Account_Count'] > 1]
                        
                        if not multi_account.empty:
                            for _, pos in multi_account.head(5).iterrows():
                                st.write(f"🔄 **{pos['Symbol']}**: ${pos['Market_Value']:,.0f} across {pos['Account_Count']} accounts")
                        else:
                            st.info("No multi-account positions found")
                
                # Export Section
                st.header("💾 Export Analysis")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("📊 Generate Excel Report", type="primary"):
                        with st.spinner("Creating Excel workbook..."):
                            excel_data = create_excel_workbook(
                                agg_df, 
                                symbol_agg, 
                                metrics, 
                                sector_analysis, 
                                options_df if include_options else None
                            )
                            
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
                            filename = f"portfolio_analysis_{timestamp}.xlsx"
                            
                            st.download_button(
                                label="⬇️ Download Excel Report",
                                data=excel_data,
                                file_name=filename,
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                            
                            st.success("✅ Excel report generated successfully!")
                
                with col2:
                    csv_data = symbol_agg.to_csv(index=False)
                    st.download_button(
                        label="📄 Download CSV (Consolidated)",
                        data=csv_data,
                        file_name=f"consolidated_positions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
                
                with col3:
                    json_data = {
                        'portfolio_summary': metrics,
                        'top_holdings': symbol_agg.head(10).to_dict('records'),
                        'sector_allocation': sector_analysis.to_dict(),
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    st.download_button(
                        label="🔧 Download JSON (API)",
                        data=pd.io.json.dumps(json_data, indent=2),
                        file_name=f"portfolio_data_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                        mime="application/json"
                    )
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please ensure your file is in the correct format with the expected columns.")
    
    else:
        # Welcome screen
        st.info("👆 **Get Started**: Upload your portfolio CSV file using the sidebar to begin analysis.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 📊 **Features**")
            st.markdown("""
            - Complete portfolio aggregation
            - Risk and concentration analysis
            - Sector allocation breakdown
            - Options strategy opportunities
            - Multi-format exports
            """)
        
        with col2:
            st.markdown("### 🎯 **Analysis Types**")
            st.markdown("""
            - Individual vs consolidated positions
            - Account-level breakdowns
            - Technical insights
            - Performance tracking
            - Interactive visualizations
            """)
        
        with col3:
            st.markdown("### 💼 **Export Options**")
            st.markdown("""
            - Comprehensive Excel workbooks
            - CSV data files
            - JSON for API integration
            - Professional reports
            - Custom formatting
            """)

if __name__ == "__main__":
    main()
