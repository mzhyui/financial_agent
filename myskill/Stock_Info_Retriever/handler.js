/**
 * @typedef {Object} AnythingLLM
 * @property {import('./plugin.json')} config - your plugin's config
 * @property {function(string|Error): void} logger - Logging function
 * @property {function(string): void} introspect - Print a string to the UI while agent skill is running
 */
/** @type {AnythingLLM} */
module.exports.runtime = {
  handler: async function ({symbol, start_date, end_date}) {
    const callerId = `${this.config.name}-v${this.config.version}`;
    // const apiKey = `this.config.entrypoint.params.apiKey || ""`.trim();
    const apiKey = this.config.setup_args.ALPHAVANTAGE_API_KEY.value || "";
    
    try {
      const ticker = symbol || "";
      const startDate = start_date || "";
      const endDate = end_date || "";
      
      if (!ticker) {
        this.introspect("Stock ticker symbol is required.");
        throw new Error("Stock ticker symbol is required.");
      }
      
      // Log start of the search
      this.logger(`Retrieving stock data for ${ticker.toUpperCase()}...`);
      this.introspect(`Retrieving stock data for ${ticker.toUpperCase()}...`);
      
      // Construct API URL
      const baseUrl = "https://www.alphavantage.co/query";
      const params = new URLSearchParams({
        function: "TIME_SERIES_DAILY",
        symbol: ticker.toUpperCase(),
        outputsize: "full",
        apikey: apiKey
      });
      
      const url = `${baseUrl}?${params.toString()}`;
      
      // Make API request
      const response = await fetch(url, { timeout: 30000 });
      
      if (!response.ok) {
        throw new Error(`API request failed with status: ${response.status}`);
      }
      
      const data = await response.json();
      
      // Check for API errors
      if (data["Error Message"]) {
        throw new Error(`API Error: ${data["Error Message"]}`);
      }
      
      if (data["Note"]) {
        throw new Error(`API Limit: ${data["Note"]}`);
      }
      
      const timeSeriesData = data["Time Series (Daily)"];
      
      if (!timeSeriesData) {
        throw new Error("No time series data found in API response");
      }
      
      this.introspect(`Retrieved ${Object.keys(timeSeriesData).length} days of data`);
      
      // Filter data by date range if provided
      let filteredData = timeSeriesData;
      let dateFilterApplied = false;
      
      if (startDate || endDate) {
        filteredData = this.filterByDateRange(timeSeriesData, startDate, endDate);
        dateFilterApplied = true;
        this.introspect(`Filtered data to ${Object.keys(filteredData).length} days within specified range`);
      }
      
      // Convert to array format similar to pandas DataFrame
      const stockData = this.convertToDataFrame(filteredData);
      
      // Generate summary
      const summary = this.generateSummary(stockData, ticker.toUpperCase(), startDate, endDate, dateFilterApplied);
      
      this.logger("Stock data retrieval completed");
      this.introspect("Stock data retrieval completed - processing results...");
      
      // return {
      //   success: true,
      //   symbol: ticker.toUpperCase(),
      //   dataCount: stockData.length,
      //   dateRange: {
      //     start: startDate || (stockData.length > 0 ? stockData[stockData.length - 1].date : null),
      //     end: endDate || (stockData.length > 0 ? stockData[0].date : null)
      //   },
      //   data: stockData,
      //   summary: summary
      // };

      // return in json format
      return JSON.stringify({
        success: true,
        symbol: ticker.toUpperCase(),
        dataCount: stockData.length,
        dateRange: {
          start: startDate || (stockData.length > 0 ? stockData[stockData.length - 1].date : null),
          end: endDate || (stockData.length > 0 ? stockData[0].date : null)
        },
        data: stockData,
        summary: summary
      })
      
    } catch (e) {
      this.logger(e);
      this.introspect(`${callerId} failed to execute. Reason: ${e.message}`);
      return {
        success: false,
        error: e.message,
        message: `Failed to retrieve stock data. Error: ${e.message}`
      };
    }
  },
  
  /**
   * Filter time series data by date range
   * @param {Object} timeSeriesData - Raw time series data from API
   * @param {string} startDate - Start date in YYYY-MM-DD format
   * @param {string} endDate - End date in YYYY-MM-DD format
   * @returns {Object} Filtered time series data
   */
  filterByDateRange: function(timeSeriesData, startDate, endDate) {
    const filtered = {};
    
    for (const date in timeSeriesData) {
      const currentDate = new Date(date);
      let includeDate = true;
      
      if (startDate) {
        const start = new Date(startDate);
        if (currentDate < start) {
          includeDate = false;
        }
      }
      
      if (endDate && includeDate) {
        const end = new Date(endDate);
        if (currentDate > end) {
          includeDate = false;
        }
      }
      
      if (includeDate) {
        filtered[date] = timeSeriesData[date];
      }
    }
    
    return filtered;
  },
  
  /**
   * Convert time series data to DataFrame-like structure
   * @param {Object} timeSeriesData - Filtered time series data
   * @returns {Array} Array of stock data objects
   */
  convertToDataFrame: function(timeSeriesData) {
    const dataFrame = [];
    
    for (const date in timeSeriesData) {
      const dayData = timeSeriesData[date];
      dataFrame.push({
        date: date,
        open: parseFloat(dayData["1. open"]),
        high: parseFloat(dayData["2. high"]),
        low: parseFloat(dayData["3. low"]),
        close: parseFloat(dayData["4. close"]),
        volume: parseInt(dayData["5. volume"])
      });
    }
    
    // Sort by date (most recent first)
    dataFrame.sort((a, b) => new Date(b.date) - new Date(a.date));
    
    return dataFrame;
  },
  
  /**
   * Generate summary of stock data
   * @param {Array} stockData - Array of stock data objects
   * @param {string} ticker - Stock ticker symbol
   * @param {string} startDate - Start date filter
   * @param {string} endDate - End date filter
   * @param {boolean} dateFilterApplied - Whether date filtering was applied
   * @returns {string} Summary string
   */
  generateSummary: function(stockData, ticker, startDate, endDate, dateFilterApplied) {
    if (stockData.length === 0) {
      return `No stock data found for ${ticker}`;
    }
    
    const latest = stockData[0];
    const oldest = stockData[stockData.length - 1];
    
    // Calculate basic statistics
    const prices = stockData.map(d => d.close);
    const maxPrice = Math.max(...prices);
    const minPrice = Math.min(...prices);
    const avgPrice = prices.reduce((sum, price) => sum + price, 0) / prices.length;
    
    // Calculate price change
    const priceChange = latest.close - oldest.close;
    const percentChange = ((priceChange / oldest.close) * 100);
    
    let summary = `Stock Data Summary for ${ticker}:\n\n`;
    
    if (dateFilterApplied) {
      summary += `Date Range: ${startDate || 'earliest'} to ${endDate || 'latest'}\n`;
    }
    
    summary += `Period: ${oldest.date} to ${latest.date}\n`;
    summary += `Total Trading Days: ${stockData.length}\n\n`;
    
    summary += `Latest Price (${latest.date}): $${latest.close.toFixed(2)}\n`;
    summary += `Opening Price (${oldest.date}): $${oldest.close.toFixed(2)}\n`;
    summary += `Price Change: $${priceChange.toFixed(2)} (${percentChange.toFixed(2)}%)\n\n`;
    
    summary += `Statistics for the period:\n`;
    summary += `  Highest Price: $${maxPrice.toFixed(2)}\n`;
    summary += `  Lowest Price: $${minPrice.toFixed(2)}\n`;
    summary += `  Average Price: $${avgPrice.toFixed(2)}\n`;
    summary += `  Price Range: $${(maxPrice - minPrice).toFixed(2)}\n\n`;
    
    summary += `Recent Trading Activity:\n`;
    stockData.slice(0, 5).forEach((day, index) => {
      summary += `  ${day.date}: Open $${day.open.toFixed(2)}, High $${day.high.toFixed(2)}, Low $${day.low.toFixed(2)}, Close $${day.close.toFixed(2)}\n`;
    });
    
    return summary;
  }
};