import streamlit as st
import pandas as pd
import numpy as np
import io
import re
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Portfolio Analyzer Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1e3d59;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

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
            if 'cash' in first_col.lower() or any(word in first_col.lower() for word in ['cash', 'investments', 'sweep', 'core']):
                account_sections.append({
                    'row_index': idx,
                    'account': current_account,
                    'symbol': first_col,
                    'row_data': row
                })
            elif not any(word in first_col.lower() for word in ['cash']):
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
    tech_symbols = ['AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA', 'CRM', 'ADBE']
    energy_symbols = ['XOM', 'ET', 'EPD', 'NEE', 'COP', 'SLB']
    health_symbols = ['UNH', 'JNJ', 'PFE']
    financial_symbols = ['KEY', 'BX', 'ARCC']
    
    symbol = symbol.upper()
    desc_upper = description.upper()
    
    # Comprehensive cash detection
    cash_indicators = [
        'CASH', 'MONEY', 'SWEEP', 'SETTLEMENT', 'CORE', 'FDIC', 'BANK',
        'SAVINGS', 'CHECKING', 'DEPOSIT', 'MMDA', 'MM', 'OVERNIGHT',
        'INVESTMENTS'
    ]
    
    if 'CASH & CASH INVESTMENTS' in desc_upper or \
       'CASH & CASH INVESTMENTS' in symbol or \
       any(indicator in symbol for indicator in cash_indicators) or \
       any(indicator in desc_upper for indicator in cash_indicators):
        return 'Cash & Cash Equivalents'
    elif 'ETF' in desc_upper or 'FUND' in desc_upper:
        return 'ETF'
    elif symbol in tech_symbols:
        return 'Technology'
    elif symbol in energy_symbols:
        return 'Energy'
    elif symbol in health_symbols:
        return 'Healthcare'
    elif symbol in financial_symbols:
        return 'Financial Services'
    else:
        return 'Other'

def calculate_portfolio_metrics(consolidated_df):
    """Calculate comprehensive portfolio metrics from consolidated positions"""
    total_value = consolidated_df['Market_Value'].sum()
    sorted_df = consolidated_df.sort_values('Market_Value', ascending=False)
    
    cash_positions = consolidated_df[consolidated_df['Sector'] == 'Cash & Cash Equivalents']
    cash_value = cash_positions['Market_Value'].sum()
    
    metrics = {
        'total_portfolio_value': total_value,
        'position_count': len(consolidated_df),
        'average_position_size': total_value / len(consolidated_df) if len(consolidated_df) > 0 else 0,
        'largest_position': sorted_df['Market_Value'].iloc[0],
        'largest_position_symbol': sorted_df['Symbol'].iloc[0],
        'top_10_concentration': sorted_df.head(10)['Market_Value'].sum() / total_value * 100,
        'top_5_concentration': sorted_df.head(5)['Market_Value'].sum() / total_value * 100,
        'single_largest_weight': sorted_df['Market_Value'].iloc[0] / total_value * 100,
        'cash_percentage': (cash_value / total_value * 100) if total_value > 0 else 0
    }
    
    return metrics

def get_technical_analysis(symbols, period='3mo'):
    """Fetch and analyze technical data for symbols"""
    tech_data = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, symbol in enumerate(symbols):
        try:
            status_text.text(f'Analyzing {symbol}... ({i+1}/{len(symbols)})')
            progress_bar.progress((i + 1) / len(symbols))
            
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            if len(hist) > 20:
                current_price = hist['Close'].iloc[-1]
                
                # Technical indicators
                sma_20 = hist['Close'].rolling(window=20).mean().iloc[-1]
                sma_50 = hist['Close'].rolling(window=50).mean().iloc[-1] if len(hist) >= 50 else sma_20
                
                # RSI calculation
                delta = hist['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = (100 - (100 / (1 + rs))).iloc[-1]
                
                # Support and resistance
                recent_data = hist.tail(20)
                resistance = recent_data['High'].max()
                support = recent_data['Low'].min()
                
                # Generate signal
                if rsi < 30:
                    signal = "OVERSOLD"
                elif rsi > 70:
                    signal = "OVERBOUGHT"
                elif current_price > sma_20:
                    signal = "BULLISH"
                else:
                    signal = "BEARISH"
                
                tech_data.append({
                    'Symbol': symbol,
                    'Current_Price': round(current_price, 2),
                    'RSI': round(rsi, 1),
                    'SMA_20': round(sma_20, 2),
                    'SMA_50': round(sma_50, 2),
                    'Price_vs_SMA20': "Above" if current_price > sma_20 else "Below",
                    'Support': round(support, 2),
                    'Resistance': round(resistance, 2),
                    'Signal': signal,
                    'Volume': int(hist['Volume'].iloc[-1]) if len(hist) > 0 else 0
                })
        except Exception as e:
            st.warning(f"Could not fetch data for {symbol}: {str(e)}")
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    return tech_data

def analyze_options_strategies(symbol_agg, min_weight=3):
    """Analyze options strategies for large positions"""
    large_positions = symbol_agg[
        (symbol_agg['Consolidated_Weight'] > min_weight) & 
        (symbol_agg['Sector'] != 'Cash & Cash Equivalents')
    ].copy()
    
    options_data = []
    
    for _, position in large_positions.iterrows():
        symbol = position['Symbol']
        
        if len(symbol) <= 4 and symbol.isalpha():
            position_value = position['Market_Value']
            weight = position['Consolidated_Weight']
            quantity = position.get('Quantity', 0)
            
            if quantity > 0:
                estimated_price = position_value / quantity
            else:
                estimated_price = 100
            
            # Covered call analysis
            monthly_premium_rate = 0.02  # 2% monthly estimate
            round_lots = max(1, int(quantity / 100))
            
            if round_lots > 0:
                monthly_income = position_value * monthly_premium_rate
                annual_income = monthly_income * 12
                
                options_data.append({
                    'Symbol': symbol,
                    'Strategy': 'Covered Call',
                    'Position_Value': round(position_value, 2),
                    'Position_Weight': round(weight, 1),
                    'Current_Price_Est': round(estimated_price, 2),
                    'Suggested_Strike': round(estimated_price * 1.05, 2),
                    'Round_Lots': round_lots,
                    'Monthly_Income_Est': round(monthly_income, 2),
                    'Monthly_Yield_Est': round((monthly_income / position_value) * 100, 2),
                    'Annual_Income_Est': round(annual_income, 2),
                    'Annual_Yield_Est': round((annual_income / position_value) * 100, 2),
                    'Recommended_Expiration': '30-45 days'
                })
    
    return options_data

def create_risk_analysis(symbol_agg, agg_df, metrics):
    """Create comprehensive risk analysis"""
    risk_data = []
    
    # Position concentration risks
    high_concentration = symbol_agg[symbol_agg['Consolidated_Weight'] > 10]
    for _, pos in high_concentration.iterrows():
        risk_level = 'CRITICAL' if pos['Consolidated_Weight'] > 20 else 'HIGH' if pos['Consolidated_Weight'] > 15 else 'MEDIUM'
        risk_data.append({
            'Risk_Type': 'Position Concentration',
            'Item': pos['Symbol'],
            'Current_Value': f"${pos['Market_Value']:,.0f}",
            'Current_Weight': f"{pos['Consolidated_Weight']:.1f}%",
            'Risk_Level': risk_level,
            'Recommendation': f"Reduce to <10% of portfolio (sell ~${pos['Market_Value']*0.3:,.0f})" if pos['Consolidated_Weight'] > 10 else "Monitor closely"
        })
    
    # Sector concentration risks
    sector_analysis = agg_df.groupby('Sector')['Market_Value'].sum()
    sector_weights = (sector_analysis / metrics['total_portfolio_value'] * 100)
    
    for sector, weight in sector_weights.items():
        if weight > 30:
            risk_level = 'HIGH' if weight > 50 else 'MEDIUM'
            risk_data.append({
                'Risk_Type': 'Sector Concentration',
                'Item': sector,
                'Current_Value': f"${sector_analysis[sector]:,.0f}",
                'Current_Weight': f"{weight:.1f}%",
                'Risk_Level': risk_level,
                'Recommendation': f"Diversify across sectors (target <25%)"
            })
    
    # Low diversification warning
    if len(symbol_agg) < 15:
        risk_data.append({
            'Risk_Type': 'Low Diversification',
            'Item': 'Total Portfolio',
            'Current_Value': f"{len(symbol_agg)} positions",
            'Current_Weight': 'N/A',
            'Risk_Level': 'MEDIUM',
            'Recommendation': 'Consider adding more positions for better diversification'
        })
    
    # Cash allocation analysis
    if metrics['cash_percentage'] > 10:
        risk_data.append({
            'Risk_Type': 'High Cash Allocation',
            'Item': 'Cash & Cash Equivalents',
            'Current_Value': f"${sector_analysis.get('Cash & Cash Equivalents', 0):,.0f}",
            'Current_Weight': f"{metrics['cash_percentage']:.1f}%",
            'Risk_Level': 'LOW',
            'Recommendation': 'Consider investing excess cash for better returns'
        })
    elif metrics['cash_percentage'] < 2:
        risk_data.append({
            'Risk_Type': 'Low Cash Reserves',
            'Item': 'Cash & Cash Equivalents',
            'Current_Value': f"${sector_analysis.get('Cash & Cash Equivalents', 0):,.0f}",
            'Current_Weight': f"{metrics['cash_percentage']:.1f}%",
            'Risk_Level': 'MEDIUM',
            'Recommendation': 'Consider maintaining 3-5% cash for opportunities'
        })
    
    if not risk_data:
        risk_data = [{
            'Risk_Type': 'Portfolio Health',
            'Item': 'Overall Assessment',
            'Current_Value': 'Well Diversified',
            'Current_Weight': 'N/A',
            'Risk_Level': 'LOW',
            'Recommendation': 'Portfolio shows good diversification - continue monitoring'
        }]
    
    return risk_data

def create_excel_report(summary_df, consolidated_export, individual_export, sector_summary, 
                       account_summary, tech_df, options_df, risk_df):
    """Create Excel report and return as bytes"""
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        summary_df.to_excel(writer, sheet_name='Portfolio_Summary', index=False)
        consolidated_export.to_excel(writer, sheet_name='Consolidated_Positions', index=False)
        individual_export.to_excel(writer, sheet_name='Individual_Positions', index=False)
        sector_summary.to_excel(writer, sheet_name='Sector_Analysis', index=False)
        account_summary.to_excel(writer, sheet_name='Account_Analysis', index=False)
        tech_df.to_excel(writer, sheet_name='Technical_Analysis', index=False)
        options_df.to_excel(writer, sheet_name='Options_Strategies', index=False)
        risk_df.to_excel(writer, sheet_name='Risk_Analysis', index=False)
    
    return output.getvalue()

# Main Streamlit App
def main():
    st.markdown('<h1 class="main-header">📊 Portfolio Analyzer Pro</h1>', unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Analysis Settings")
    
    # Analysis parameters
    tech_positions = st.sidebar.slider(
        "📈 Positions for Technical Analysis", 
        min_value=5, max_value=20, value=10, step=1,
        help="Number of top positions to analyze technically"
    )
    
    options_min_weight = st.sidebar.slider(
        "💼 Min. Position Weight for Options", 
        min_value=1.0, max_value=10.0, value=3.0, step=0.5,
        help="Minimum portfolio weight (%) to consider for options strategies"
    )
    
    tech_period = st.sidebar.selectbox(
        "📊 Technical Analysis Period",
        ["1mo", "3mo", "6mo", "1y"],
        index=1,
        help="Historical period for technical analysis"
    )
    
    include_etfs = st.sidebar.checkbox(
        "Include ETFs in Technical Analysis",
        value=False,
        help="Whether to include ETFs in technical analysis (stocks only by default)"
    )
    
    # File upload
    st.header("📁 Upload Portfolio Data")
    uploaded_file = st.file_uploader(
        "Choose your portfolio CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload your broker's portfolio export file"
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
            with st.spinner("🔄 Processing portfolio data..."):
                sections = find_account_sections(df)
                agg_df = create_aggregated_positions(sections, df)
                
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
                    metrics = calculate_portfolio_metrics(symbol_agg)
                    
                    # Add weights
                    agg_df['Position_Weight'] = (agg_df['Market_Value'] / metrics['total_portfolio_value'] * 100).round(2)
                    symbol_agg['Consolidated_Weight'] = (symbol_agg['Market_Value'] / metrics['total_portfolio_value'] * 100).round(2)
            
            # Display key metrics
            st.header("📊 Portfolio Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Portfolio Value",
                    f"${metrics['total_portfolio_value']:,.0f}",
                    help="Total market value of all positions"
                )
            
            with col2:
                st.metric(
                    "Unique Positions",
                    f"{metrics['position_count']}",
                    help="Number of unique securities (consolidated across accounts)"
                )
            
            with col3:
                st.metric(
                    "Largest Position",
                    f"{metrics['single_largest_weight']:.1f}%",
                    f"{metrics['largest_position_symbol']}",
                    help="Percentage of portfolio in largest single position"
                )
            
            with col4:
                st.metric(
                    "Cash Allocation",
                    f"{metrics['cash_percentage']:.1f}%",
                    help="Percentage of portfolio in cash and cash equivalents"
                )
            
            # Display cash positions if detected
            cash_positions = symbol_agg[symbol_agg['Sector'] == 'Cash & Cash Equivalents']
            if len(cash_positions) > 0:
                with st.expander("💰 Cash Positions Detected"):
                    for _, pos in cash_positions.iterrows():
                        st.write(f"**{pos['Symbol']}**: ${pos['Market_Value']:,.2f} ({pos['Consolidated_Weight']:.1f}%)")
            
            # Top Holdings
            st.header("🏆 Top 10 Holdings")
            top_10 = symbol_agg.head(10)[['Symbol', 'Description', 'Market_Value', 'Consolidated_Weight', 'Sector']]
            top_10.columns = ['Symbol', 'Description', 'Market Value', 'Weight %', 'Sector']
            top_10['Market Value'] = top_10['Market Value'].apply(lambda x: f"${x:,.0f}")
            top_10['Weight %'] = top_10['Weight %'].apply(lambda x: f"{x:.1f}%")
            st.dataframe(top_10, use_container_width=True)
            
            # Technical Analysis
            st.header("📈 Technical Analysis")
            
            # Get symbols for technical analysis
            top_positions = symbol_agg.head(tech_positions)
            stock_symbols = []
            
            for _, pos in top_positions.iterrows():
                symbol = pos['Symbol']
                if include_etfs:
                    if symbol and len(symbol) <= 5 and symbol.isalpha() and pos['Sector'] != 'Cash & Cash Equivalents':
                        stock_symbols.append((symbol, pos['Market_Value'], pos['Consolidated_Weight']))
                else:
                    if symbol and len(symbol) <= 5 and symbol.isalpha() and pos['Sector'] not in ['Cash & Cash Equivalents', 'ETF']:
                        stock_symbols.append((symbol, pos['Market_Value'], pos['Consolidated_Weight']))
            
            if stock_symbols:
                st.info(f"Analyzing {len(stock_symbols)} positions for technical signals...")
                
                # Get technical data
                symbols_only = [item[0] for item in stock_symbols]
                tech_data = get_technical_analysis(symbols_only, tech_period)
                
                if tech_data:
                    # Add position info to technical data
                    for tech_item in tech_data:
                        symbol = tech_item['Symbol']
                        pos_info = next((item for item in stock_symbols if item[0] == symbol), None)
                        if pos_info:
                            tech_item['Position_Value'] = pos_info[1]
                            tech_item['Position_Weight'] = pos_info[2]
                    
                    tech_df = pd.DataFrame(tech_data)
                    tech_df = tech_df.sort_values('Position_Value', ascending=False)
                    
                    # Display technical analysis results
                    st.subheader("📊 Technical Signals")
                    
                    display_cols = ['Symbol', 'Current_Price', 'Position_Weight', 'RSI', 'Price_vs_SMA20', 'Signal']
                    display_df = tech_df[display_cols].copy()
                    display_df.columns = ['Symbol', 'Price', 'Weight %', 'RSI', 'vs SMA20', 'Signal']
                    display_df['Price'] = display_df['Price'].apply(lambda x: f"${x:.2f}")
                    display_df['Weight %'] = display_df['Weight %'].apply(lambda x: f"{x:.1f}%")
                    display_df['RSI'] = display_df['RSI'].apply(lambda x: f"{x:.1f}")
                    
                    st.dataframe(display_df, use_container_width=True)
                else:
                    st.warning("Could not fetch technical data for the selected positions")
                    tech_df = pd.DataFrame()
            else:
                st.info("No individual stocks found for technical analysis in top positions")
                tech_df = pd.DataFrame()
            
            # Options Analysis
            st.header("💼 Options Strategies")
            
            options_data = analyze_options_strategies(symbol_agg, options_min_weight)
            
            if options_data:
                options_df = pd.DataFrame(options_data)
                options_df = options_df.sort_values('Position_Value', ascending=False)
                
                st.subheader("📈 Covered Call Opportunities")
                
                display_options = options_df[['Symbol', 'Position_Weight', 'Suggested_Strike', 'Monthly_Yield_Est', 'Annual_Yield_Est', 'Recommended_Expiration']].copy()
                display_options.columns = ['Symbol', 'Weight %', 'Suggested Strike', 'Monthly Yield %', 'Annual Yield %', 'Expiration']
                display_options['Weight %'] = display_options['Weight %'].apply(lambda x: f"{x:.1f}%")
                display_options['Suggested Strike'] = display_options['Suggested Strike'].apply(lambda x: f"${x:.2f}")
                display_options['Monthly Yield %'] = display_options['Monthly Yield %'].apply(lambda x: f"{x:.2f}%")
                display_options['Annual Yield %'] = display_options['Annual Yield %'].apply(lambda x: f"{x:.1f}%")
                
                st.dataframe(display_options, use_container_width=True)
                
                st.info("💡 **Options Strategy Note**: These are estimated yields based on typical option premiums. Actual premiums vary based on volatility, time to expiration, and market conditions. Consider 30-45 day expirations for covered calls.")
            else:
                st.info(f"No positions above {options_min_weight}% found for options analysis")
                options_df = pd.DataFrame()
            
            # Risk Analysis
            st.header("⚠️ Risk Analysis")
            
            risk_data = create_risk_analysis(symbol_agg, agg_df, metrics)
            risk_df = pd.DataFrame(risk_data)
            
            # Color code risk levels
            def color_risk_level(val):
                if val == 'CRITICAL':
                    return 'background-color: #ff6b6b'
                elif val == 'HIGH':
                    return 'background-color: #ffa500'
                elif val == 'MEDIUM':
                    return 'background-color: #ffeb3b'
                else:
                    return 'background-color: #4caf50'
            
            styled_risk = risk_df.style.applymap(color_risk_level, subset=['Risk_Level'])
            st.dataframe(styled_risk, use_container_width=True)
            
            # Prepare Excel export
            st.header("📥 Download Complete Analysis")
            
            # Prepare all dataframes for Excel
            summary_data = {
                'Metric': [
                    'Total Portfolio Value',
                    'Unique Positions (Consolidated)', 
                    'Individual Holdings (All Accounts)',
                    'Average Position Size (Consolidated)',
                    'Largest Position Symbol',
                    'Largest Position Value',
                    'Largest Position Weight',
                    'Top 5 Concentration',
                    'Top 10 Concentration',
                    'Cash Percentage',
                    'Number of Accounts',
                    'Analysis
