# Black-Scholes Option Profit/Loss Heatmap

This app visualizes the potential profit/loss of European call/put options over time using the Black-Scholes model. It fetches real-time stock data and option chain information using the yfinance library

Run this app online at https://black-scholes-heatmap-hp.streamlit.app/

## Features

*   Fetches current spot price for a given ticker.
*   Retrieves option chain data (calls and puts) for a specified expiry date.
*   Calculates the theoretical option price using the Black-Scholes formula.
*   Calculates the Probability of Profit (PoP) for both call and put options.
*   Generates interactive heatmaps showing the percentage profit/loss for a range of strike prices relative to the holding strike, across the remaining days until expiry.
*   Separate heatmaps for call and put options based on user-provided holding strike and purchase price.
*   Dark theme visualization.
*   Handles potential issues like missing data or past expiry dates.

## Installation

1.  **Clone the repository or download the script.**
2.  **Install the required Python libraries:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the Streamlit application:**
    ```bash
    streamlit run black-scholes-ticker.py
    ```
2.  **Open your web browser** and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).
3.  **Use the sidebar** to input:
    *   Ticker Symbol (e.g., SPY, AAPL)
    *   Option Expiry Date (YYYY-MM-DD format)
    *   Holding Strike Price (the strike you are analyzing)
    *   Call Option Price Paid (the premium paid for the call)
    *   Put Option Price Paid (the premium paid for the put)
    *   Risk-Free Interest Rate (as a decimal, e.g., 0.05 for 5%)
4.  **Click "Generate Heatmaps"** to view the visualizations for both call and put options based on your inputs.

## Dependencies

*   streamlit
*   numpy
*   matplotlib
*   seaborn
*   yfinance
*   scipy
