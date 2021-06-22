import matplotlib.pyplot as plt
import pandas as pd
from pandas_datareader import data
from pandas.plotting import register_matplotlib_converters
from pandas import DataFrame

import investpy

class DataLoader:
    '''Class to download stock data from yahoo finance.
    
    Methods
    -------
    get_data(self):
        Returns Data Frame containing stock data of pre-defined ticker, start and end date.
    get_high(self):
        Returns Data Frame containing daily highest value of stock.
    get_low(self):
        Returns Data Frame containing daily lowest value of stock.
    get_open(self):
        Returns Data Frame containing daily opening value of stock.
    get_close(self):
        Returns Data Frame containing daily closing value of stock.
    get_volume(self):
        Returns Data Frame containing daily volume value of stock.
    get_adjclose(self):
        Returns Data Frame containing daily adjusted closing value of stock.
    calculate_returns_open(self):
        Calculates and returns Data Frame containing return values of daily opening price.
    calculate_returns_close(self):
        Calculates and returns Data Frame containing return values of daily closing price.
    calculate_returns_adj(self):
        Calculates and returns Data Frame containing return values of daily adjusted closing price.
    plotting_grid(self):
        Plots data specified when object is created in grid format 2x3.
    plotting(self):
        Plots data specified when object is created.
    statistics(self):
        Provides basic statistics of data specified when object is created.
    save_frame(self, name: str):
        Takes stock data loaded and saves it as a csv in current directory.
    '''
    def __init__(self, ticker: str, start_date: str, end_date: str) -> object:
        '''
            Parameters:
                ticker (str): Stock exchange ticker symbol of stock.
                start_date (str): Start date of price retrieval, format: yyyy-mm-dd
                end_date (str): End date of price retrieval, format: yyyy-mm-dd
            Returns:
                (Object): Dataloader object containing stock data data frame 
        '''
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date

        self.stockdata = data.DataReader(ticker, 'yahoo', start_date, end_date)

    def __str__(self) -> str:
        '''Returns basic object information on what ticker data, start and end date, and the total count of data points
           are held by the object.
        '''
        return f"Total count of rows: {self.stockdata.shape[0]}\nTicker: {self.ticker}\nStart: {self.start_date}\nEnd: {self.end_date}"
    
    def get_data(self) -> DataFrame:
        '''Returns data frame containing stock data of ticker symbol specified.

            Returns:
                (DataFrame): Data frame containing stock data.
        '''
        return self.stockdata

    def get_high(self) -> DataFrame:
        '''Returns data frame containing daily high valuses of ticker symbol specified.

            Returns:
                (DataFrame): Data frame containing daily high values of ticker symbol.
        '''
        return pd.DataFrame(self.stockdata['High'])

    def get_low(self) -> DataFrame:
        '''Returns data frame containing daily lowest valuses of ticker symbol specified.

            Returns:
                (DataFrame): Data frame containing daily lowest values of ticker symbol.
        '''
        return pd.DataFrame(self.stockdata['Low'])

    def get_open(self) -> DataFrame:
        '''Returns data frame containing daily opening valuses of ticker symbol specified.

            Returns:
                (DataFrame): Data frame containing daily opening values of ticker symbol.
        '''
        return pd.DataFrame(self.stockdata['Open'])

    def get_close(self) -> DataFrame:
        '''Returns data frame containing daily closing valuses of ticker symbol specified.

            Returns:
                (DataFrame): Data frame containing daily closing values of ticker symbol.
        '''
        return pd.DataFrame(self.stockdata['Close'])

    def get_volume(self) -> DataFrame:
        '''Returns data frame containing daily volume valuses of ticker symbol specified.

            Returns:
                (DataFrame): Data frame containing daily volume values of ticker symbol.
        '''
        return pd.DataFrame(self.stockdata['Volume'])

    def get_adjclose(self) -> DataFrame:
        '''Returns data frame containing daily adjusted closing valuses of ticker symbol specified.

            Returns:
                (DataFrame): Data frame containing daily adjusted closing values of ticker symbol.
        '''
        return pd.DataFrame(self.stockdata['Adj Close'])

    def calculate_returns_open(self) -> DataFrame:
        '''Calculates return values of daily opening values. return = ((price today - price yesterday)/price yesterday)

            Returns:
                (DataFrame): Data frame containing daily return values.
        '''
        data = pd.DataFrame(self.stockdata['Open'].diff(1))
        data.rename(columns={'Open':'Return'}, inplace=True)
        return data

    def calculate_returns_close(self) -> DataFrame:
        '''Calculates return values of daily closing values. return = ((price today - price yesterday)/price yesterday)

            Returns:
                (DataFrame): Data frame containing daily return values.
        '''
        data = pd.DataFrame(self.stockdata['Close'].diff(1))
        data.rename(columns={'Close':'Return'}, inplace=True)
        return data

    def calculate_returns_adj(self) -> DataFrame:
        '''Calculates return values of daily adjusted closing values. return = ((price today - price yesterday)/price yesterday)

            Returns:
                (DataFrame): Data frame containing daily return values.
        '''
        data = pd.DataFrame(self.stockdata['Adj Close'].diff(1))
        data.rename(columns={'Adj Close':'Return'}, inplace=True)
        return data

    def plotting_grid(self):
        '''Plots graphs representing data loaded in one grid.
        '''
        plt.figure(figsize = (8,10))
        plt.subplot(3, 2, 1)
        self.stockdata['High'].plot(title = 'Daily highest values')

        plt.subplot(3, 2, 2)
        self.stockdata['Low'].plot(title = 'Daily lowest values')

        plt.subplot(3, 2, 3)
        self.stockdata['Open'].plot(title = 'Daily opening price')

        plt.subplot(3, 2, 4)
        self.stockdata['Close'].plot(title = 'Daily closing price')

        plt.subplot(3, 2, 5)
        self.stockdata['Volume'].plot(title = 'Daily volume')

        plt.subplot(3, 2, 6)
        self.stockdata['Adj Close'].plot(title = 'Daily adjusted closing price')
       
        plt.tight_layout()
        plt.show()

    def plotting(self):
        '''Plots graphs representing data loaded one by one.
        '''
        plot0 = plt.figure(0)
        self.stockdata['High'].plot(title = 'Daily highest values')

        plot1 = plt.figure(1)
        self.stockdata['Low'].plot(title = 'Daily lowest values')

        plot2 = plt.figure(2)
        self.stockdata['Open'].plot(title = 'Daily opening price')

        plot3 = plt.figure(3)
        self.stockdata['Close'].plot(title = 'Daily closing price')

        plot4 = plt.figure(4)
        self.stockdata['Volume'].plot(title = 'Daily volume')

        plot5 = plt.figure(5)
        self.stockdata['Adj Close'].plot(title = 'Daily adjusted closing price')
       
        plt.show()

    def statistics(self) -> DataFrame:
        '''Calculates and returns basic statistics of data loaded. Statistics include: count, mean, std, min, 25%, 50%, 75%, max.
            Returns:
                (DataFrame): Dataframe containing statistics.
        '''
        return self.stockdata.describe()
    
    def save_frame(self, name: str):
        '''Takes data loaded and saves it in the same directory as a csv file.
            
            Parameters:
                name (str): Name of file to be saved as .csv
        '''
        return self.stockdata.to_csv(name + '.csv')

class AlternativeDataLoader:
    '''Alternative class to download financial data from the investpy library.
    
    Methods
    -------
    get_data(self):
        Returns Data Frame containing stock data of pre-defined ticker, start and end date.
    get_high(self):
        Returns Data Frame containing daily highest value of stock.
    get_low(self):
        Returns Data Frame containing daily lowest value of stock.
    get_open(self):
        Returns Data Frame containing daily opening value of stock.
    get_close(self):
        Returns Data Frame containing daily closing value of stock.
    get_volume(self):
        Returns Data Frame containing daily volume value of stock.
    calculate_returns_open(self):
        Calculates and returns Data Frame containing return values of daily opening price.
    calculate_returns_close(self):
        Calculates and returns Data Frame containing return values of daily closing price.
    plotting_grid(self):
        Plots data specified when object is created in grid format 2x3.
    plotting(self):
        Plots data specified when object is created.
    statistics(self):
        Provides basic statistics of data specified when object is created.
    save_frame(self, name: str):
        Takes stock data loaded and saves it as a csv in current directory.
    '''
    def __init__(self, ticker: str, country: str, start_date: str, end_date: str, category: str) -> object:
        '''
            Parameters:
                ticker (str): Stock exchange ticker symbol of stock.
                country (str): Country of financial product.
                start_date (str): Start date of price retrieval, format: dd/mm/yyyy
                end_date (str): End date of price retrieval, format: dd/mm/yyyy
                category (str): Financial product: stock, fund, etf, index
            Returns:
                (Object): Dataloader object containing stock data data frame 
        '''
        self.ticker = ticker
        self.country = country
        self.start_date = start_date
        self.end_date = end_date
        self.category = category

        if self.category == 'stock':
            self.stockdata = investpy.get_stock_historical_data(stock = self.ticker, country = self.country, from_date = self.start_date, to_date = self.end_date)
        elif self.category == 'fund':
            self.stockdata = investpy.get_fund_historical_data(fund = self.ticker, country = self.country, from_date = self.start_date, to_date = self.end_date)
        elif self.category == 'etf':
            self.stockdata = investpy.get_etf_historical_data(etf = self.ticker, country = self.country, from_date = self.start_date, to_date = self.end_date)
        elif self.category == 'index':
            self.stockdata = investpy.get_index_historical_data(index = self.ticker, country = self.country, from_date = self.start_date, to_date = self.end_date)
        else:
            raise 'No data can be found'

    def __str__(self) -> str:
        '''Returns basic object information on what ticker data, start and end date, and the total count of data points
           are held by the object.
        '''
        return f"Total count of rows: {self.stockdata.shape[0]}\nTicker: {self.ticker}\nStart: {self.start_date}\nEnd: {self.end_date}"
    
    def get_data(self) -> DataFrame:
        '''Returns data frame containing stock data of ticker symbol specified.

            Returns:
                (DataFrame): Data frame containing stock data.
        '''
        return self.stockdata

    def get_high(self) -> DataFrame:
        '''Returns data frame containing daily high valuses of ticker symbol specified.

            Returns:
                (DataFrame): Data frame containing daily high values of ticker symbol.
        '''
        return pd.DataFrame(self.stockdata['High'])

    def get_low(self) -> DataFrame:
        '''Returns data frame containing daily lowest valuses of ticker symbol specified.

            Returns:
                (DataFrame): Data frame containing daily lowest values of ticker symbol.
        '''
        return pd.DataFrame(self.stockdata['Low'])

    def get_open(self) -> DataFrame:
        '''Returns data frame containing daily opening valuses of ticker symbol specified.

            Returns:
                (DataFrame): Data frame containing daily opening values of ticker symbol.
        '''
        return pd.DataFrame(self.stockdata['Open'])

    def get_close(self) -> DataFrame:
        '''Returns data frame containing daily closing valuses of ticker symbol specified.

            Returns:
                (DataFrame): Data frame containing daily closing values of ticker symbol.
        '''
        return pd.DataFrame(self.stockdata['Close'])

    def get_volume(self) -> DataFrame:
        '''Returns data frame containing daily volume valuses of ticker symbol specified.

            Returns:
                (DataFrame): Data frame containing daily volume values of ticker symbol.
        '''
        return pd.DataFrame(self.stockdata['Volume'])

    def calculate_returns_open(self) -> DataFrame:
        '''Calculates return values of daily opening values. return = ((price today - price yesterday)/price yesterday)

            Returns:
                (DataFrame): Data frame containing daily return values.
        '''
        data = pd.DataFrame(self.stockdata['Open'].diff(1))
        data.rename(columns={'Open':'Return'}, inplace=True)
        return data

    def calculate_returns_close(self) -> DataFrame:
        '''Calculates return values of daily closing values. return = ((price today - price yesterday)/price yesterday)

            Returns:
                (DataFrame): data frame containing daily return values.
        '''
        data = pd.DataFrame(self.stockdata['Close'].diff(1))
        data.rename(columns={'Close':'Return'}, inplace=True)
        return data

    def plotting_grid(self):
        '''Plots graphs representing data loaded in one grid.
        '''
        plt.subplot(3, 2, 1)
        self.stockdata['High'].plot(title = 'Daily highest values')

        plt.subplot(3, 2, 2)
        self.stockdata['Low'].plot(title = 'Daily lowest values')

        plt.subplot(3, 2, 3)
        self.stockdata['Open'].plot(title = 'Daily opening price')

        plt.subplot(3, 2, 4)
        self.stockdata['Close'].plot(title = 'Daily closing price')

        plt.subplot(3, 2, 5)
        self.stockdata['Volume'].plot(title = 'Daily volume')
       
        plt.tight_layout()
        plt.show()

    def plotting(self):
        '''Plots graphs representing data loaded one by one.
        '''
        plot0 = plt.figure(0)
        self.stockdata['High'].plot(title = 'Daily highest values')

        plot1 = plt.figure(1)
        self.stockdata['Low'].plot(title = 'Daily lowest values')

        plot2 = plt.figure(2)
        self.stockdata['Open'].plot(title = 'Daily opening price')

        plot3 = plt.figure(3)
        self.stockdata['Close'].plot(title = 'Daily closing price')

        plot4 = plt.figure(4)
        self.stockdata['Volume'].plot(title = 'Daily volume')
       
        plt.show()

    def statistics(self) -> DataFrame:
        '''Calculates and returns basic statistics of data loaded. Statistics include: count, mean, std, min, 25%, 50%, 75%, max.
            Returns:
                (DataFrame): Data frame containing statistics.
        '''
        return self.stockdata.describe()
    
    def save_frame(self, name: str):
        '''Takes data loaded and saves it in the same directory as a csv file.
            
            Parameters:
                name (str): Name of file to be saved as .csv
        '''
        return self.stockdata.to_csv(name + '.csv')
