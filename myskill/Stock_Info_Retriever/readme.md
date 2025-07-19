## Stock Info Retriever

This agent skill retrieves stock information using the Alpha Vantage API.

## Setup

1. Obtain a free API key from [Alpha Vantage](https://www.alphavantage.co/support/#api-key).
2. Set the API key as an environment variable:
    ```bash
    export ALPHAVANTAGE_API_KEY=your_api_key_here
    ```
3. Install any required dependencies as specified in the project.

## Usage

Once installed, use the agent skill to fetch real-time stock data. Example prompt:

```
Get the latest price for AAPL.
```

You can also request historical data or other stock metrics supported by Alpha Vantage. Refer to the documentation for more advanced queries.
