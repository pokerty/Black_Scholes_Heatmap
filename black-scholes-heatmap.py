import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from scipy.stats import norm
from datetime import datetime
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl

def get_spot_price(ticker):
    """Fetch the current spot price for a given ticker."""
    stock = yf.Ticker(ticker)
    return float(stock.history(period='1d')['Close'].iloc[-1])

def get_option_chain(ticker, expiry_date):
    """Fetch and process the option chain data."""
    stock = yf.Ticker(ticker)
    options = stock.option_chain(expiry_date)

    calls = options.calls
    puts = options.puts # Get puts as well

    numeric_columns = ['strike', 'lastPrice', 'bid', 'ask', 'impliedVolatility']
    for col in numeric_columns:
        if col in calls.columns:
            calls[col] = calls[col].astype(float)
        if col in puts.columns: # Process puts too
            puts[col] = puts[col].astype(float)

    return calls, puts # Return both

def calculate_pop(S, K, T, r, sigma, premium):
    """
    Calculate probability of profit for a long call option
    Returns: probability as a percentage
    """
    # Calculate breakeven price
    breakeven = K + premium

    # Calculate parameters for the lognormal distribution
    drift = r - (sigma**2 / 2)

    # Calculate Z-score for breakeven
    z_score = (np.log(breakeven/S) - (drift - sigma**2/2)*T) / (sigma * np.sqrt(T))

    # Calculate probability of being above breakeven
    pop = (1 - norm.cdf(z_score)) * 100

    return pop

def calculate_put_pop(S, K, T, r, sigma, premium):
    """
    Calculate probability of profit for a long put option
    Returns: probability as a percentage
    """
    # Calculate breakeven price for a put
    breakeven = K - premium
    if breakeven <= 0: # Cannot have negative breakeven price
         return 0.0

    # Calculate parameters for the lognormal distribution
    drift = r - (sigma**2 / 2)

    # Calculate Z-score for breakeven (probability of S being BELOW breakeven)
    z_score = (np.log(breakeven/S) - (drift - sigma**2/2)*T) / (sigma * np.sqrt(T))

    # Calculate probability of being below breakeven
    pop = norm.cdf(z_score) * 100

    return pop

class BlackScholes:
    def __init__(self, S, K, T, r, sigma):
        self.S = float(S)  # Spot price
        self.K = float(K)  # Strike price
        self.T = float(T)  # Time to maturity in years
        self.r = float(r)  # Risk-free interest rate
        self.sigma = float(sigma)  # Volatility

    def d1(self):
        if self.T <= 0:
            return 0
        return (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))

    def d2(self):
        if self.T <= 0:
            return 0
        return self.d1() - self.sigma * np.sqrt(self.T)

    def calculate_call_price(self):
        if self.T <= 0:
            return max(0, self.S - self.K)
        d1 = self.d1()
        d2 = self.d2()
        return self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)

    def calculate_put_price(self):
        """Calculate the Black-Scholes price for a European put option."""
        if self.T <= 0:
            # Intrinsic value at expiration
            return max(0, self.K - self.S)
        d1 = self.d1()
        d2 = self.d2()
        # Put price formula
        put_price = self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)
        return put_price

def create_glossy_colormap():
    """Create a custom colormap with red for losses and green for profits"""
    # Define colors for negative values (red gradient)
    red_colors = [
        (0.3, 0.05, 0.05, 0.4),   # Very subtle red
        (0.4, 0.05, 0.05, 0.6),   # Light red
        (0.5, 0.05, 0.05, 0.7),   # Medium red
        (0.6, 0.07, 0.07, 0.8),   # Stronger red
        (0.7, 0.1, 0.1, 0.9)      # Most intense red
    ]
    # Define colors for positive values (green gradient)
    green_colors = [
        (0.05, 0.3, 0.05, 0.4),  # Very subtle green
        (0.05, 0.4, 0.05, 0.6),  # Light green
        (0.05, 0.5, 0.05, 0.7),  # Medium green
        (0.07, 0.6, 0.07, 0.8),  # Stronger green
        (0.1, 0.7, 0.1, 0.9)     # Most intense green
    ]

    # Create color segments with smooth transition
    colors = np.vstack((red_colors[::-1], [(0.2,0.2,0.2,0.3)], green_colors))

    # Create custom colormap with more bins for smoother gradient
    n_bins = 200
    return LinearSegmentedColormap.from_list("custom_diverging", colors, N=n_bins)

def plot_option_values_heatmap(ticker, expiry_date, holding_strike, option_price, risk_free_rate=0.05, option_type='call'):
    """
    Create a dark theme heatmap visualization of option profit/loss scenarios
    Parameters:
    ticker (str): Stock ticker symbol
    expiry_date (str): Option expiry date in 'YYYY-MM-DD' format
    holding_strike (float): Strike price of the option being analyzed
    option_price (float): Price paid for the option
    risk_free_rate (float): Risk-free interest rate (default 0.05)
    option_type (str): 'call' or 'put'
    Returns:
    matplotlib.pyplot: Plot object
    """
    # Set the style to dark background
    plt.style.use('dark_background')

    # Get option chain data
    calls_chain, puts_chain = get_option_chain(ticker, expiry_date) # Get both chains
    current_spot = get_spot_price(ticker)

    # Select the correct chain based on option_type
    if option_type == 'call':
        option_chain = calls_chain
    elif option_type == 'put':
        option_chain = puts_chain
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    # Filter strikes around holding strike (±10 strikes)
    # Ensure the holding strike exists in the selected chain
    if holding_strike not in option_chain['strike'].values:
         st.warning(f"Holding strike ${holding_strike} not found in the {option_type} chain for {expiry_date}. Using nearest available strike.")
         # Find nearest strike (optional, or just error out)
         available_strikes = option_chain['strike'].unique()
         holding_strike = available_strikes[np.abs(available_strikes - holding_strike).argmin()]
         st.info(f"Using nearest strike: ${holding_strike}")


    mask = (option_chain['strike'] >= holding_strike - 10) & (option_chain['strike'] <= holding_strike + 10)
    filtered_chain = option_chain[mask].copy()

    # Sort strikes in descending order
    strike_prices = np.sort(filtered_chain['strike'].unique())[::-1]

    # Calculate days to expiry
    current_date = datetime.now()
    expiry_datetime = datetime.strptime(expiry_date, "%Y-%m-%d")
    total_days = (expiry_datetime - current_date).days
    if total_days < 0:
        st.error("Selected expiry date is in the past.")
        return None # Return None if date is invalid
    dates = np.arange(total_days + 1)

    # Initialize matrix
    value_matrix = np.zeros((len(strike_prices), len(dates)))

    # Get implied volatility for holding strike
    holding_option_row = filtered_chain[filtered_chain['strike'] == holding_strike]
    if holding_option_row.empty:
         st.error(f"Could not find data for holding strike ${holding_strike} in the {option_type} chain.")
         return None
    holding_option = holding_option_row.iloc[0]

    # Check for NaN IV and handle it (e.g., use average IV or skip)
    holding_iv = float(holding_option['impliedVolatility'])
    if np.isnan(holding_iv):
        st.warning(f"Implied Volatility for strike ${holding_strike} is NaN. Using average IV from filtered chain.")
        average_iv = filtered_chain['impliedVolatility'].mean()
        if np.isnan(average_iv):
             st.error("Could not determine a valid Implied Volatility.")
             return None
        holding_iv = average_iv
        st.info(f"Using average IV: {holding_iv:.2%}")


    # Calculate PoP based on option type
    time_to_expiry = total_days / 365.0
    if option_type == 'call':
        pop = calculate_pop(
            S=current_spot, K=holding_strike, T=time_to_expiry,
            r=risk_free_rate, sigma=holding_iv, premium=option_price
        )
    else: # put
         pop = calculate_put_pop(
            S=current_spot, K=holding_strike, T=time_to_expiry,
            r=risk_free_rate, sigma=holding_iv, premium=option_price
        )


    # Calculate profit/loss matrix
    for i, strike in enumerate(strike_prices):
        row_data = filtered_chain[filtered_chain['strike'] == strike].iloc[0]
        implied_vol = float(row_data['impliedVolatility'])
        # Handle NaN IV in the loop as well
        if np.isnan(implied_vol):
            implied_vol = holding_iv # Use holding IV as fallback

        for j, days in enumerate(dates):
            time_to_expiry = (total_days - days) / 365.0
            bs = BlackScholes(
                S=current_spot,
                K=strike,
                T=time_to_expiry,
                r=risk_free_rate,
                sigma=implied_vol # Use the specific IV for each strike
            )
            # Calculate theoretical price based on option type
            if option_type == 'call':
                theoretical_price = bs.calculate_call_price()
            else: # put
                theoretical_price = bs.calculate_put_price()

            # Handle potential division by zero if option_price is 0
            if option_price != 0:
                value_matrix[i, j] = ((theoretical_price - option_price) / option_price) * 100
            else:
                value_matrix[i, j] = np.inf if theoretical_price > 0 else (-np.inf if theoretical_price < 0 else 0) # Or handle as appropriate


    # Create figure with dark background
    fig = plt.figure(figsize=(15, 10))
    ax = plt.gca()

    # Set background color to semi-transparent dark grey
    fig.patch.set_facecolor('#2A2A2A')
    ax.set_facecolor('#3A3A3A80')  # More transparent background

    # Create custom colormap
    custom_cmap = create_glossy_colormap()

    # Plot heatmap
    sns.heatmap(value_matrix[::-1,::],
                xticklabels=[f'{d}d' for d in reversed(dates)],
                yticklabels=[f'${s:.1f}' for s in strike_prices],
                annot=True,
                fmt='.1f',
                cmap=custom_cmap,
                cbar=False,
                annot_kws={'size': 6, 'color': 'white', 'ha': 'center', 'va': 'center'},
                center=0) # Center color map at 0 P/L

    # Style title and labels
    plt.title(f'{option_type.capitalize()} Option Profit/Loss Over Time (%) - {ticker}\n' + # Updated title
              f'Strike: ${holding_strike}, Spot: ${current_spot:.2f}, Price Paid: ${option_price:.2f}, IV: {holding_iv:.1%}\n' +
              f'Probability of Profit: {pop:.1f}%',
              color='white', pad=20)

    plt.xlabel('Days to Expiry', color='white', labelpad=10)
    plt.ylabel('Strike Price', color='white', labelpad=10)

    # Style tick labels
    ax.tick_params(colors='white')

    # Highlight holding strike with a glowing effect
    holding_idx_list = np.where(strike_prices == holding_strike)[0]
    if len(holding_idx_list) > 0:
         holding_idx = holding_idx_list[0]
         plt.axhline(y=holding_idx, color='#00FF00', alpha=0.3, linewidth=2)
         plt.axhline(y=holding_idx, color='#00FF00', alpha=0.1, linewidth=4)
    else:
         st.warning(f"Could not visually highlight strike ${holding_strike} on the heatmap.")


    # Add grid for better readability
    ax.grid(True, color='#404040', linestyle='--', alpha=0.3)

    plt.tight_layout()
    return fig # Return the figure object instead of plt

# Streamlit App UI
st.title('Option Profit/Loss Heatmap')

st.sidebar.header('Input Parameters')
ticker = st.sidebar.text_input('Ticker Symbol', 'SPY')
# TODO: Add validation or selection for expiry dates
expiry_date = st.sidebar.text_input('Expiry Date (YYYY-MM-DD)', '2025-05-30')
holding_strike = st.sidebar.number_input('Holding Strike Price', value=535.0, format="%.2f")
option_price_call = st.sidebar.number_input('Call Option Price Paid', value=10.0, format="%.2f") # Separate input for call price
option_price_put = st.sidebar.number_input('Put Option Price Paid', value=8.0, format="%.2f") # Separate input for put price
risk_free_rate = st.sidebar.number_input('Risk-Free Rate', value=0.05, format="%.2f")

if st.sidebar.button('Generate Heatmaps'): # Changed button text
    try:
        st.subheader(f"Call Options - {ticker} {expiry_date}")
        # Generate and display Call heatmap
        fig_call = plot_option_values_heatmap(
            ticker,
            expiry_date,
            holding_strike,
            option_price_call, # Use call price
            risk_free_rate,
            option_type='call' # Explicitly set type
        )
        if fig_call: # Check if fig was created successfully
             st.pyplot(fig_call)
        else:
             st.warning("Could not generate Call heatmap.")

        st.divider() # Add a visual separator

        st.subheader(f"Put Options - {ticker} {expiry_date}")
         # Generate and display Put heatmap
        fig_put = plot_option_values_heatmap(
            ticker,
            expiry_date,
            holding_strike,
            option_price_put, # Use put price
            risk_free_rate,
            option_type='put' # Explicitly set type
        )
        if fig_put: # Check if fig was created successfully
             st.pyplot(fig_put)
        else:
             st.warning("Could not generate Put heatmap.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.error("Please ensure the ticker and expiry date are valid and data is available.")

# Remove or comment out the original example usage
# if __name__ == "__main__":
#     # Set parameters
#     ticker = 'SPY'
#     expiry_date = '2025-01-31'  # Must be a valid option expiration date
#     holding_strike = 585.0      # Strike price of the option
#     option_price = 10.0         # Price paid for the option
#
#     # Create and display the visualization
#     fig = plot_option_values_heatmap(ticker, expiry_date, holding_strike, option_price)
#     plt.show()