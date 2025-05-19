import blpapi
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

def initialize_session():
    """Initialize Bloomberg API session"""
    session_options = blpapi.SessionOptions()
    session_options.setServerHost("localhost")
    session_options.setServerPort(8194)
    session = blpapi.Session(session_options)
    
    if not session.start():
        print("Failed to start session.")
        return None
    
    if not session.openService("//blp/refdata"):
        print("Failed to open //blp/refdata")
        return None
    
    return session

def get_cot_data(session, commodity_tickers, start_date, end_date):
    """
    Fetch COT data for specified commodities
    
    Parameters:
    - session: Bloomberg API session
    - commodity_tickers: List of Bloomberg tickers for commodities
    - start_date: Start date for historical data
    - end_date: End date for historical data
    
    Returns:
    - DataFrame with COT positions
    """
    refdata_service = session.getService("//blp/refdata")
    request = refdata_service.createRequest("HistoricalDataRequest")
    
    # Set request parameters
    for ticker in commodity_tickers:
        request.append("securities", ticker)
    
    # COT fields for asset managers/pension funds
    request.append("fields", "CFTC_AM_ALL_NET_POSITIONS")  # Net positions
    request.append("fields", "CFTC_AM_ALL_LONG_POSITIONS")  # Long positions
    request.append("fields", "CFTC_AM_ALL_SHORT_POSITIONS")  # Short positions
    request.append("fields", "CFTC_AM_ALL_SPREADING_POSITIONS")  # Spreading positions
    
    request.set("startDate", start_date)
    request.set("endDate", end_date)
    request.set("periodicitySelection", "WEEKLY")
    
    # Send request
    session.sendRequest(request)
    
    # Process response
    data = {}
    
    while True:
        event = session.nextEvent(500)
        
        for msg in event:
            if msg.messageType() == blpapi.Name("HistoricalDataResponse"):
                security_data = msg.getElement("securityData")
                ticker = security_data.getElementAsString("security")
                
                field_data = security_data.getElement("fieldData")
                
                if ticker not in data:
                    data[ticker] = []
                
                for i in range(field_data.numValues()):
                    point = field_data.getValue(i)
                    date = point.getElementAsDatetime("date").date()
                    
                    # Get position data
                    position_data = {
                        "date": date,
                        "ticker": ticker
                    }
                    
                    # Extract all available fields
                    for field in ["CFTC_AM_ALL_NET_POSITIONS", 
                                 "CFTC_AM_ALL_LONG_POSITIONS", 
                                 "CFTC_AM_ALL_SHORT_POSITIONS",
                                 "CFTC_AM_ALL_SPREADING_POSITIONS"]:
                        if point.hasElement(field):
                            position_data[field] = point.getElementAsFloat(field)
                        else:
                            position_data[field] = None
                    
                    data[ticker].append(position_data)
        
        if event.eventType() == blpapi.Event.RESPONSE:
            break
    
    # Convert to DataFrame
    result = []
    for ticker, ticker_data in data.items():
        result.extend(ticker_data)
    
    return pd.DataFrame(result)

def detect_pension_fund_selling(cot_df, window=4):
    """
    Analyze COT data to detect pension fund selling activity
    
    Parameters:
    - cot_df: DataFrame with COT data
    - window: Window size for moving average (default: 4 weeks)
    
    Returns:
    - DataFrame with selling signals
    """
    result = []
    
    for ticker in cot_df['ticker'].unique():
        ticker_data = cot_df[cot_df['ticker'] == ticker].sort_values('date')
        
        # Calculate moving averages
        ticker_data['net_ma'] = ticker_data['CFTC_AM_ALL_NET_POSITIONS'].rolling(window).mean()
        ticker_data['long_ma'] = ticker_data['CFTC_AM_ALL_LONG_POSITIONS'].rolling(window).mean()
        
        # Calculate rate of change
        ticker_data['net_change'] = ticker_data['CFTC_AM_ALL_NET_POSITIONS'].diff()
        ticker_data['long_change'] = ticker_data['CFTC_AM_ALL_LONG_POSITIONS'].diff()
        ticker_data['short_change'] = ticker_data['CFTC_AM_ALL_SHORT_POSITIONS'].diff()
        
        # Detect selling signals
        # 1. Decrease in net positions below moving average
        # 2. Decrease in long positions
        # 3. Increase in short positions
        ticker_data['selling_signal'] = ((ticker_data['CFTC_AM_ALL_NET_POSITIONS'] < ticker_data['net_ma']) & 
                                        (ticker_data['long_change'] < 0) & 
                                        (ticker_data['short_change'] > 0))
        
        # Calculate selling intensity (0-100 scale)
        max_long_decrease = abs(ticker_data['long_change'].min())
        max_short_increase = ticker_data['short_change'].max()
        
        ticker_data['selling_intensity'] = 0
        mask = ticker_data['selling_signal']
        
        ticker_data.loc[mask, 'selling_intensity'] = (
            50 * abs(ticker_data.loc[mask, 'long_change']) / max_long_decrease +
            50 * ticker_data.loc[mask, 'short_change'] / max_short_increase
        )
        
        result.append(ticker_data)
    
    return pd.concat(result, ignore_index=True)

def visualize_pension_fund_activity(cot_df, commodity_ticker):
    """
    Create visualization of pension fund activity for a commodity
    
    Parameters:
    - cot_df: DataFrame with COT data
    - commodity_ticker: Ticker to visualize
    """
    plt.figure(figsize=(14, 10))
    
    # Filter data for the specific ticker
    ticker_data = cot_df[cot_df['ticker'] == commodity_ticker].sort_values('date')
    
    # Plot 1: Net positions
    plt.subplot(3, 1, 1)
    plt.plot(ticker_data['date'], ticker_data['CFTC_AM_ALL_NET_POSITIONS'], 'b-', label='Net Positions')
    plt.plot(ticker_data['date'], ticker_data['net_ma'], 'r--', label='Net Position MA')
    plt.title(f'Pension Fund Positions - {commodity_ticker}')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Long and Short positions
    plt.subplot(3, 1, 2)
    plt.plot(ticker_data['date'], ticker_data['CFTC_AM_ALL_LONG_POSITIONS'], 'g-', label='Long Positions')
    plt.plot(ticker_data['date'], ticker_data['CFTC_AM_ALL_SHORT_POSITIONS'], 'r-', label='Short Positions')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Selling signal and intensity
    plt.subplot(3, 1, 3)
    plt.bar(ticker_data['date'], ticker_data['selling_intensity'], color='purple', alpha=0.7)
    plt.title('Selling Intensity (0-100)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    # Example parameters
    commodity_tickers = [
        "CL1 Comdty",    # Crude Oil
        "GC1 Comdty",    # Gold
        "SI1 Comdty",    # Silver
        "HG1 Comdty",    # Copper
        "C 1 Comdty",    # Corn
        "W 1 Comdty"     # Wheat
    ]
    
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
    
    # Initialize Bloomberg API session
    session = initialize_session()
    if session is None:
        return
    
    try:
        # Get COT data
        print("Fetching COT data...")
        cot_data = get_cot_data(session, commodity_tickers, start_date, end_date)
        
        # Analyze for pension fund selling
        print("Analyzing pension fund activity...")
        analysis_results = detect_pension_fund_selling(cot_data)
        
        # Summary of selling activity
        selling_summary = analysis_results[analysis_results['selling_signal']]
        
        if len(selling_summary) > 0:
            print("\nDetected pension fund selling activity:")
            for ticker in commodity_tickers:
                ticker_selling = selling_summary[selling_summary['ticker'] == ticker]
                if len(ticker_selling) > 0:
                    recent_selling = ticker_selling[ticker_selling['date'] >= (datetime.now() - timedelta(days=60)).date()]
                    if len(recent_selling) > 0:
                        avg_intensity = recent_selling['selling_intensity'].mean()
                        print(f"{ticker}: Selling detected with average intensity of {avg_intensity:.2f}/100")
                        
                        # Visualize this commodity
                        visualize_pension_fund_activity(analysis_results, ticker)
        else:
            print("\nNo significant pension fund selling activity detected in the specified time period.")
            
    finally:
        # Stop the session
        session.stop()

if __name__ == "__main__":
    main()