import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import io
from tqdm import tqdm


from myLogger import myLogger
from utils.load import getStockMarketData
from utils.workspace import auth, generateNewWorkspace, generateNewThread, chatWithThread
from utils.operation import extract_prediction_line, extract_confidence_operation

# ==== 股票预测与结果分析函数 BEGIN ====
def sync_data_by_date_range(df_market, df_operation):
    """
    Get date range from df_operation and sync with df_market data
    If date exists in df_market but not in df_operation, add it to df_operation with 0 values
    Returns filtered dataframes with common date range
    """
    df_market = df_market.sort_index()
    operation_start_date = df_operation.index.min()
    operation_end_date = df_operation.index.max()
    # print(f"Operation date range: {operation_start_date} to {operation_end_date}")
    df_market_filtered = df_market.loc[operation_start_date:operation_end_date]
    # print(f"Market data entries in date range: {len(df_market_filtered)}")
    # print(f"Operation data entries: {len(df_operation)}")
    df_operation_synced = df_operation.reindex(df_market_filtered.index, fill_value=0)
    df_market_synced = df_market_filtered
    # print(f"After synchronization:")
    # print(f"Number of dates: {len(df_market_synced)}")
    # print(f"Date range: {df_market_synced.index.min()} to {df_market_synced.index.max()}")
    return df_market_synced, df_operation_synced

def calculate_returns(df_market_synced, df_operation_synced, initial_capital=1000):
    """
    Calculate returns based on trading operations with high initial capital
    Decision: 1 = buy, -1 = sell, 0 = hold
    Hands: number of shares to trade
    """
    df_combined = df_market_synced.copy()
    df_combined['Decision'] = pd.to_numeric(df_operation_synced['Decision'], errors='coerce').fillna(0).astype(int)
    df_combined['Hands'] = pd.to_numeric(df_operation_synced['Hands'], errors='coerce').fillna(0).astype(int)
    initial_capital = initial_capital * df_combined['close'][0]
    portfolio_values = []
    cash = float(initial_capital)
    shares_held = int(0)
    trade_log = []
    for i, (date, row) in enumerate(df_combined.iterrows()):
        price = float(row['close'])
        decision = int(row['Decision'])
        hands = int(row['Hands'])
        if decision == 1 and hands > 0:
            trade_cost = float(hands) * price
            if cash >= trade_cost:
                cash = cash - trade_cost
                shares_held = shares_held + hands
                trade_log.append({
                    'date': date,
                    'action': 'BUY',
                    'shares': hands,
                    'price': price,
                    'value': trade_cost,
                    'cash_after': cash,
                    'shares_after': shares_held
                })
            else:
                affordable_shares = int(cash / price)
                if affordable_shares > 0:
                    actual_cost = float(affordable_shares) * price
                    cash = cash - actual_cost
                    shares_held = shares_held + affordable_shares
                    trade_log.append({
                        'date': date,
                        'action': 'PARTIAL_BUY',
                        'shares': affordable_shares,
                        'price': price,
                        'value': actual_cost,
                        'cash_after': cash,
                        'shares_after': shares_held
                    })
        elif decision == -1 and hands > 0:
            shares_to_sell = min(hands, shares_held)
            if shares_to_sell > 0:
                trade_value = float(shares_to_sell) * price
                cash = cash + trade_value
                shares_held = shares_held - shares_to_sell
                trade_log.append({
                    'date': date,
                    'action': 'SELL',
                    'shares': shares_to_sell,
                    'price': price,
                    'value': trade_value,
                    'cash_after': cash,
                    'shares_after': shares_held
                })
        current_portfolio_value = cash + (float(shares_held) * price)
        portfolio_values.append(current_portfolio_value)
    df_combined['Portfolio_Value'] = portfolio_values
    df_combined['Cash'] = cash
    df_combined['Shares_Held'] = shares_held
    final_value = portfolio_values[-1]
    total_return = (final_value - initial_capital) / initial_capital
    df_combined['Daily_Return'] = df_combined['Portfolio_Value'].pct_change()
    df_combined['Cumulative_Return'] = (df_combined['Portfolio_Value'] / initial_capital) - 1
    initial_price = float(df_combined['close'].iloc[0])
    final_price = float(df_combined['close'].iloc[-1])
    buy_hold_return = (final_price - initial_price) / initial_price
    performance_metrics = {
        'Initial Capital': initial_capital,
        'Final Portfolio Value': final_value,
        'Total Return': total_return,
        'Total Return (%)': total_return * 100,
        'Buy & Hold Return (%)': buy_hold_return * 100,
        'Excess Return (%)': (total_return - buy_hold_return) * 100,
        'Number of Trades': len(trade_log),
        'Final Cash': cash,
        'Final Shares': shares_held,
        'Final Stock Price': final_price,
        'Annualized Return (%)': (((final_value / initial_capital) ** (252 / len(df_combined))) - 1) * 100,
    }
    if len(df_combined['Daily_Return'].dropna()) > 1:
        daily_returns = df_combined['Daily_Return'].dropna()
        performance_metrics.update({
            'Volatility (%)': daily_returns.std() * np.sqrt(252) * 100,
            'Sharpe Ratio': (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0,
            'Max Drawdown (%)': ((df_combined['Portfolio_Value'] / df_combined['Portfolio_Value'].cummax()) - 1).min() * 100
        })
    return df_combined, pd.DataFrame(trade_log), performance_metrics

def plot_performance_analysis(df_results, metrics):
    """
    Create comprehensive performance visualization
    """
    plt.style.use('default')
    sns.set_palette("husl")
    fig, ((ax1, ax2, ax3, ax4)) = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle('Trading Strategy Performance Analysis', fontsize=16, fontweight='bold')
    ax1_twin = ax1.twinx()
    line1 = ax1.plot(df_results.index, df_results['Portfolio_Value'], 'b-', label='Portfolio Value', linewidth=2.5, alpha=0.8)
    ax1.set_ylabel('Portfolio Value ($)', color='b', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e9:.1f}B' if x >= 1e9 else f'${x/1e6:.1f}M'))
    line2 = ax1_twin.plot(df_results.index, df_results['close'], 'r--', label='Stock Price', linewidth=2, alpha=0.7)
    ax1_twin.set_ylabel('Stock Price ($)', color='r', fontweight='bold')
    ax1_twin.tick_params(axis='y', labelcolor='r')
    ax1.set_title('Portfolio Value vs Stock Price', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    strategy_returns = df_results['Cumulative_Return'] * 100
    buy_hold_returns = (df_results['close'] / df_results['close'].iloc[0] - 1) * 100
    ax2.plot(df_results.index, strategy_returns, 'g-', label='Strategy Return', linewidth=2.5, alpha=0.8)
    ax2.plot(df_results.index, buy_hold_returns, 'orange', label='Buy & Hold Return', linestyle='--', linewidth=2, alpha=0.8)
    ax2.set_ylabel('Cumulative Return (%)', fontweight='bold')
    ax2.set_title('Cumulative Returns Comparison', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    textstr = f'Strategy: {metrics["Total Return (%)"]:.2f}%\nBuy&Hold: {metrics["Buy & Hold Return (%)"]:.2f}%\nExcess: {metrics["Excess Return (%)"]:.2f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax2.text(0.62, 0.98, textstr, transform=ax2.transAxes, fontsize=10, verticalalignment='top', bbox=props)
    ax3.plot(df_results.index, df_results['close'], 'k-', alpha=0.6, label='Stock Price', linewidth=1.5)
    buy_dates = df_results[df_results['Decision'] == 1].index
    sell_dates = df_results[df_results['Decision'] == -1].index
    if len(buy_dates) > 0:
        ax3.scatter(buy_dates, df_results.loc[buy_dates, 'close'], color='green', marker='^', s=100, label='Buy', alpha=0.8, edgecolors='darkgreen')
    if len(sell_dates) > 0:
        ax3.scatter(sell_dates, df_results.loc[sell_dates, 'close'], color='red', marker='v', s=100, label='Sell', alpha=0.8, edgecolors='darkred')
    ax3.set_ylabel('Stock Price ($)', fontweight='bold')
    ax3.set_title('Trading Activity', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    stock_values = df_results['Shares_Held'] * df_results['close']
    cash_values = df_results['Portfolio_Value'] - stock_values
    ax4.fill_between(df_results.index, 0, cash_values, label='Cash', alpha=0.6, color='lightblue')
    ax4.fill_between(df_results.index, cash_values, df_results['Portfolio_Value'], label='Stock Holdings', alpha=0.6, color='lightcoral')
    ax4.set_ylabel('Value ($)', fontweight='bold')
    ax4.set_title('Portfolio Composition', fontweight='bold')
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e9:.1f}B' if x >= 1e9 else f'${x/1e6:.1f}M'))
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    for ax in [ax1, ax2, ax3, ax4]:
        ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()
    # 打印详细指标
    formatted_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, float):
            if 'Capital' in key or 'Value' in key or 'Cash' in key or 'Price' in key:
                if value >= 1e9:
                    formatted_metrics[key] = f"${value/1e9:.2f}B"
                elif value >= 1e6:
                    formatted_metrics[key] = f"${value/1e6:.2f}M"
                else:
                    formatted_metrics[key] = f"${value:,.2f}"
            elif '%' in key or 'Return' in key or 'Ratio' in key:
                formatted_metrics[key] = f"{value:.4f}" if 'Ratio' in key else f"{value:.2f}%"
            else:
                formatted_metrics[key] = f"{value:.2f}"
        else:
            formatted_metrics[key] = str(value)
    metrics_df = pd.DataFrame(list(formatted_metrics.items()), columns=['Metric', 'Value'])
    print(metrics_df.to_string(index=False))
    return fig
# ==== 股票预测与结果分析函数 END ====

quantitive_factor = """
You are an AI investment advisor specializing in small-cap value strategies, providing insights and predictions based on historical data.

## Investment Focus
- Small-cap stocks (market cap < 1B USD)
- Value factors: low P/E, P/B, P/S ratios
- Quality metrics: ROE, ROA, debt levels

## Decision Framework
**BUY Signals:**
- Strong value metrics vs peers/history
- Solid fundamentals despite small size
- Positive alpha potential

**SELL Signals:**
- Valuation reaches fair value
- Deteriorating fundamentals
- Better opportunities available

## Response Format
**Stock**: [Ticker/Name]
**Price**: [Current] | **Market Cap**: [Value]

**Scores** (1-10):
- Size Factor: [Score]
- Value Factor: [Score] 
- Quality Factor: [Score]

**Decision**: [BUY/HOLD/SELL]
**Confidence**: [1-10]
**Target Allocation**: [%]
**Expected Return**: [% over timeframe]

**Rationale**: [1-2 key points]
**Key Risk**: [Main concern]
**Entry/Exit**: [Price levels]

Note: Size factor effectiveness in A-shares has weakened post-2017.
"""

bollinger_bands = """
You are a Bollinger Bands mean reversion trader. Your strategy: buy when price breaks below lower band, sell when price breaks above upper band, exit at middle band.

## Setup
- Period: 20 days
- Standard deviation: 2.0
- Upper Band = MA + (2 x StdDev)
- Lower Band = MA - (2 x StdDev)

## Signals
**BUY**: Price closes below lower band
**SELL**: Price closes above upper band  
**EXIT**: Price returns to middle band or 3% stop-loss

## Decision Format
**Symbol**: [Ticker]
**Signal**: [BUY/SELL/EXIT/HOLD]
**Current Price**: [Price]
**Band Position**: [Above/Below/Within bands]
**Entry/Exit Price**: [Specific level]
**Stop Loss**: [3% from entry]
**Rationale**: [1-2 sentences]

## Risk Rules
- Max 5% portfolio per trade
- No trading in strong trends
- Exit if no reversion within 10 days

Keep responses brief and actionable.
"""

camp_model = """
You are a CAPM model analyst. Your role is to determine when CAPM is appropriate for asset pricing and when multi-factor models are needed.

## CAPM Framework
**Formula**: E(Ri) = Rf + βi(E(Rm) - Rf)
- Ri = Expected return of asset i
- Rf = Risk-free rate
- βi = Beta (systematic risk)
- Rm = Market return

## Decision Criteria

**Use CAPM When:**
- Analyzing broad market portfolios
- Beta explains >70% of return variance
- No significant market anomalies present
- Short-term analysis (< 1 year)

**Use Multi-Factor Models When:**
- Significant alpha detected (α ≠ 0)
- Market anomalies present (size, value effects)
- Individual stock analysis
- Long-term analysis (> 1 year)

## Response Format
**Asset**: [Name/Ticker]
**Beta**: [Value]
**R-squared**: [% variance explained by market]

**Model Recommendation**: [CAPM/Multi-Factor]
**Reasoning**: [1-2 sentences why]

**If Multi-Factor Needed:**
- Additional factors: [Size/Value/Momentum/etc.]
- Expected alpha: [%]

**CAPM Limitations**: [Key assumptions violated]
"""


default_character_presets = {
    "base": "You are an AI investment advisor specializing in small-cap value strategies, providing insights and predictions based on historical data.",
    "quantitive_factor": quantitive_factor,
    "bollinger_bands": bollinger_bands,
    "camp_model": camp_model,
    # "final": "You are a final decision maker, aggregating insights from multiple agents to provide a comprehensive investment strategy."
    }
leader_preset = "You are a leader decision maker, aggregating insights from multiple agents to provide a comprehensive investment strategy."
anything_api = "1DV9A3A-SFFM1XR-QF4TYMR-HZ5X8RY"
entrypoint = "http://10.201.35.124:3001/api/v1/"
message_preset = "@agent Get {} stock info, from {} to {}, predict the later day's price, and give the buy-in or hold or sell-out decision on {}, with confidence."
message_outformat = "Current holding is {} shares, max holding is {} shares. Trade limit per operation is{}, expected return percentage is {}. Answer MUST contain Example style: '{}, buy-in, 0.5, hold, 0.1, sell-out, 0.4, hands-in, 200, hands-off, 100'"

@st.cache_data
def get_authed():
        # auth
    try:
        auth_response = auth()
        print(f"Authentication successful: {auth_response}")
    except Exception as e:
        print(f"Error during authentication: {e}")
        sys.exit(1)

    try:
        workspace_name = "My New Workspace"
        workspace_slug = generateNewWorkspace(workspace_name)
        print(f"Workspace created successfully: {workspace_slug}")
        return True, workspace_slug
    except Exception as e:
        print(f"Error during workspace creation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Title
    st.title('Agent decision making for stock trading')


    ticker = st.text_input('Enter ticker', value='AAPL')
    df_stock = getStockMarketData(ticker)

    start_date = st.date_input('Start date', value=pd.to_datetime('2023-01-01'))
    end_date = st.date_input('End date', value=pd.to_datetime('2023-1-31'))
    lookback_days = st.number_input('Lookback days', min_value=1, value=30, step=1)

    # current_holding = 500
    current_holding = st.number_input('Current holding shares', min_value=0, value=500, step=100)
    max_holding = st.number_input('Max holding shares', min_value=100, value=1000, step=100)
    trade_limit = st.number_input('Trade limit per operation', min_value=100, value=300, step=100)
    return_expectation = st.number_input('Expected return percentage', min_value=0.05, value=0.1, step=0.05)

    daterange = pd.date_range(start=start_date, end=end_date, freq='B') # Business days only

    
    # Filter the dataframe to only include dates in the stepped range
    df_filtered = df_stock[df_stock.index.isin(daterange)]
    daterange = df_filtered.index
    # sort
    daterange = sorted(daterange)

    # Multi-select characters
    character_options = list(default_character_presets.keys())
    selected_characters = st.multiselect(
        'Select agent characters to use',
        options=character_options,
        default=character_options
    )
    character_presets = {char: default_character_presets[char] for char in selected_characters}

    authed, workspace_slug = get_authed()        

    logger = myLogger(name=str(os.getpid()), log_filename=workspace_slug, propagate=False)

    if st.button('Create Agents'):
        if not authed:
            st.error("Please authenticate first by clicking 'Create Agents' button.")
            sys.exit(1)
        st.info(f"Using workspace: {workspace_slug}")
        #%% create threads for each character
        st.session_state.character_slugs = {}
        for character, preset in character_presets.items():
            try:
                logger(f"Creating thread for character: {character}")
                thread_slug = generateNewThread(workspace_slug)
                logger(f"Thread created successfully: {thread_slug}")
                st.session_state.character_slugs[character] = thread_slug
            except Exception as e:
                logger.error_(f"Error during thread creation: {e}")

            # opening
            try:
                message = f"@agent {preset}"
                chat_response = chatWithThread(workspace_slug, st.session_state.character_slugs[character], message)
                logger(f"Chat response: {chat_response}")
            except Exception as e:
                logger.error_(f"Error during chat: {e}")

        #%% create leader thread
        try:
            st.session_state.leader_thread_slug = generateNewThread(workspace_slug)
            logger(f"Leader thread created successfully: {st.session_state.leader_thread_slug}")
            # opening
            message = f"@agent {leader_preset}"
            chat_response = chatWithThread(workspace_slug, st.session_state.leader_thread_slug, message)
            print(f"Chat response: {chat_response}")
            st.success("Agents created and initialized successfully.")
        except Exception as e:
            logger.error_(f"Error during final thread creation: {e}")
        

    if st.button('Start making decisions'):
        date_decision = {}
        results_output = st.info("Decisions will be displayed here.")
        
        pd_results = pd.DataFrame(columns=['timestamp', 'Decision', 'Hands'])
        pbar = tqdm(daterange, desc="Processing dates", unit="date")
        for date in pbar:
            date_str = date.strftime('%Y-%m-%d')
            logger(f"Processing date: {date_str}")
            
            # # Check if this date was already processed (resume capability)
            # existing_results = pd.read_csv(results_file)
            # if date_str in existing_results['timestamp'].values:
            #     logger(f"Date {date_str} already processed, skipping...")
            #     # Load existing decision for current_holding calculation
            #     existing_row = existing_results[existing_results['timestamp'] == date_str].iloc[0]
            #     current_holding = max(current_holding + existing_row['Hands'] * existing_row['Decision'], 0)
            #     continue
            
            lookback_start_date = (date - pd.Timedelta(days=lookback_days)).strftime('%Y-%m-%d')
            logger(f"Lookback start date: {lookback_start_date}")
            
            message = message_preset.format(
                ticker, lookback_start_date, date_str, date_str, 
            ) + message_outformat.format(current_holding, max_holding,
                trade_limit, return_expectation,
                f"{ticker}, {date_str}"
            )
            
            try:
                response_list = []
                for character, thread_slug in st.session_state.character_slugs.items():
                    logger(f"Sending message to {character}: {message}")
                    try:
                        chat_response = chatWithThread(workspace_slug, thread_slug, message)
                        logger(f"Chat response from {character}: {chat_response}")
                        response_list.append(chat_response)
                    except Exception as e:
                        logger.error_(f"Error during chat with {character}: {e}")
                        response_list.append(f"Error from {character}: {str(e)}")
                
                # Aggregate responses
                logger("Aggregating responses from all characters...")
                aggregated_response = "\n".join(response_list)
                
                try:
                    logger("Sending aggregated response to final thread")
                    chat_response = chatWithThread(workspace_slug, st.session_state.leader_thread_slug, 
                                                aggregated_response + message_outformat.format(
                                                    current_holding, max_holding, 
                                                    trade_limit, return_expectation,
                                                    f"{ticker}, {date_str}"
                                                ))
                    logger(f"Chat response: {chat_response}")
                    
                    one_line = extract_prediction_line(chat_response)
                    decision, hands = extract_confidence_operation(one_line)
                    logger(f"Extracted decision on date {date_str}: {decision} from response: {one_line}")
                    
                    current_holding = max(current_holding + hands * decision, 0)
                    
                    # Save result immediately
                    new_result = pd.DataFrame({
                        'timestamp': [date_str],
                        'Decision': [decision],
                        'Hands': [hands]
                    })
                    
                    # Append to existing results
                    pd_results = pd.concat([pd_results, new_result], ignore_index=True)
                    
                    logger(f"Results saved for date {date_str}: Decision={decision}, Hands={hands}")
                    
                except Exception as e:
                    logger.error_(f"Error during aggregation chat for date {date_str}: {e}")
                    # Save error result
                    error_result = pd.DataFrame({
                        'timestamp': [date_str],
                        'Decision': [0],  # Default to no action on error
                        'Hands': [0]
                    })
                    pd_results = pd.concat([pd_results, error_result], ignore_index=True)
                    
            except Exception as e:
                logger.error_(f"Critical error processing date {date_str}: {e}")
                # Save error result
                error_result = pd.DataFrame({
                    'timestamp': [date_str],
                    'Decision': [0],
                    'Hands': [0]
                })
                pd_results = pd.concat([pd_results, error_result], ignore_index=True)

            # show current results in streamlit
            # remove previous output if exists
            if 'results_output' in locals():
                results_output.empty()
            results_output = st.dataframe(pd_results)
