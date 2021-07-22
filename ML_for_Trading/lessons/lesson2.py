import datetime

import matplotlib
import numpy as np
import pandas as pd
import scipy.optimize as spo
import yfinance as yf
from pandas_datareader import data as pdr

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta



def get_max_close():
    symbol = 'AAPL'
    df = get_data_from_yahoo('AAPL')
    max_close = df['Close'].max()
    print('Max Close')
    print(symbol, max_close)
    mean_volume = df['Volume'].mean()
    print 'Mean Volume (in millions)'
    print symbol, mean_volume/1e6

def get_data_from_yahoo(ticker):
    yf.pdr_override()
    stocks = [ticker]
    end = datetime.date.today()
    start = datetime.date.today() + relativedelta(months=-200)

    f = pdr.get_data_yahoo(stocks, start=start, end=end)
    return f

def plot_data(df, y_label, title="Stock prices"):
    ax = df.plot(title=title)
    ax.set_xlabel("Date")
    ax.set_ylabel(y_label)
    ax.legend(loc='upper left')
    plt.show()

def plot_boll_bands(df, rm, boll_bands, d_returns, title="Stock prices"):
    ax = df.plot(title=title)

    ax2 = d_returns.plot(title='Daily returns')

    rm.plot(label='Rolling mean', ax=ax)
    boll_bands.plot(label='Bollinger bands', ax=ax)

    d_returns.plot(label='Daily returns', ax=ax2)


    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc='upper left')
    plt.show()



def plot_selected(df, columns, start_index, end_index):
    """Plot the desired columns over index values in the given range."""
    # TODO: Your code here
    # Note: DO NOT modify anything else!
    df1 = df.ix[start_index:end_index, columns]
    df1.plot()
    plt.show()

def normalize_data(df):
    return df / df.ix[0,:]

def get_df_with_relevant_ticks(symbols):
    start_date = datetime.date.today() + relativedelta(months=-24)
    end_date = datetime.date.today()
    dates = pd.date_range(start_date, end_date)
    df1 = pd.DataFrame(index=dates)
    for symbol in symbols:
        df_temp = pd.DataFrame(get_data_from_yahoo(symbol)['Adj Close'])
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        if symbol == 'SPY':
            df_temp = df_temp.dropna(subset=["SPY"])

        df1 = df1.join(df_temp, how='inner')

    # TODO fix the fact that, for example, COIN started recently, but cannot be filled back to the SPY story
    df1.fillna(method='ffill', inplace=True)
    df1.fillna(method='bfill', inplace=True)

    return df1


def plot_bollinger_bands(df1):
    df1 = normalize_data(df1)

    rm = df1['SPY'].rolling(20).mean()
    rstd = df1['SPY'].rolling(20).std()
    upper_band, lower_band = get_bollinger_bands(rm, rstd)
    upper_band.name = 'Bollinger up'
    lower_band.name = 'Bollinger down'
    daily_returns = get_daily_returns(df1)

    boll_bands = pd.concat([upper_band, lower_band], axis=1)
    plot_boll_bands(df1, rm, boll_bands, daily_returns)

def get_daily_returns(df):
    """Compute and return the daily return values."""
    # TODO: Your code here
    # Note: Returned DataFrame must have the same number of rows
    return df.pct_change().fillna(0)
    # we can also calculate it as follows
    # df = df.copy()
    # d_returns = (df / df.shit(1)) - 1
    # or
    # d_returns[1:] = (df[1:] / df[:-1].values) - 1
    # if not using values (numpy array), pandas will synchronize the rows and screw up or shift


def get_bollinger_bands(rm, rstd):
    upper_band = rm + 2 * rstd
    lower_band = rm - 2 * rstd
    return upper_band, lower_band

def plot_histogram(df):
    daily_returns = get_daily_returns(df)
    mean = daily_returns['SPY'].mean()
    std = daily_returns['SPY'].std()


    daily_returns['SPY'].hist(bins=20, label='SPY')
    daily_returns['XOM'].hist(bins=20, label='XOM')
    #print daily_returns.kurtosis()
#
#
    #plt.axvline(mean, color='r', linestyle='dashed', linewidth=2)
    #plt.axvline(std, color='y', linestyle='dashed', linewidth=2)
    #plt.axvline(-std, color='y', linestyle='dashed', linewidth=2)
    plt.legend(loc='upper right')
    plt.show()

    #plot_data(daily_returns, 'Daily returns', title='Daily returns')

def plot_scatter_plot(df,tick_to_compare_with='SPY'):
    daily_returns = get_daily_returns(df)
    columns = [column for column in df.columns if column!=tick_to_compare_with]

    for column in columns:
        daily_returns.plot(kind='scatter', x=tick_to_compare_with, y=column)

        # regression
        beta, alpha = np.polyfit(daily_returns[tick_to_compare_with], daily_returns[column], 1) # arguments as follow: x variable, y variable, pol degree
        line_equation = beta*daily_returns[tick_to_compare_with] + alpha

        print 'beta_%s (vs. %s)=' % (column, tick_to_compare_with), beta
        print 'alpha_%s (vs. %s)=' % (column, tick_to_compare_with), alpha
        plt.plot(daily_returns[tick_to_compare_with], line_equation, '-', color='r')

    # calculate correlation coefficient
    print daily_returns.corr(method='pearson')
    plt.show()

def get_portfolio_values(df, alloc=[0.25, 0.25, 0.25, 0.25], start_val=100000):
    prices = df
    normed = normalize_data(prices)
    alloced = normed * alloc
    pos_vals = alloced * start_val
    port_val = pos_vals.sum(axis=1)
    return port_val


def plot_portfolio_values_and_statistics(df, alloc=[0.25, 0.25, 0.25, 0.25], start_val=20000):
    port_val = get_portfolio_values(df, alloc, start_val=start_val)

    # portfolio statistics
    portf_daily_returns = get_daily_returns(port_val)[1:]
    cum_returns = get_cum_returns(port_val)
    avg_daily_retuns = portf_daily_returns.mean()
    std_daily_ret = portf_daily_returns.std()
    sharpe_ratio = np.sqrt(252)*(portf_daily_returns.rolling(20).mean())/portf_daily_returns.rolling(20).std()

    print 'Cum returns = %s' % cum_returns
    print 'Avg Daily returns = %s' % avg_daily_retuns
    print 'Std daily returns = %s' %std_daily_ret
    plot_data(port_val, 'portfolio value', title='Port values')
    plot_data(portf_daily_returns, 'portfolio daily ret value', title='Port daily ret')
    plot_data(sharpe_ratio, 'Sharpe ratio', title='Sharpe ratio')





def get_cum_returns(df):
    return (df[-1]/df[0]) - 1

def f(X):
    Y = (X-1.5)**2 + 0.5
    print "X = {}, Y={}".format(X,Y)
    return Y

def function_optimizer_parabola(func_name):
    x_guess= 2.0
    min_result = spo.minimize(func_name, x_guess, method='SLSQP', options={'disp': True})
    print "Minima found at:"
    print "X = {}, Y={}".format(min_result.x,min_result.fun)

def function_optimizer_line(func_name):
    # define original line
    l_orig= np.float32([4,2]) # to test if the minimizer can get this line back
    print "Original line C0 = {}, C1={}".format(l_orig[0], l_orig[1])
    Xorig = np.linspace(0,10,21)
    Yorig = l_orig[0]*Xorig + l_orig[1]
    plt.plot(Xorig, Yorig, 'b--', linewidth=2.0, label="Original Line")

    # Generate noisy data points
    noise_sigma = 3.0
    noise = np.random.normal(0, noise_sigma, Yorig.shape)
    data = np.asarray([Xorig, Yorig + noise]).T
    plt.plot(data[:,0], data[:,1], 'go', label='data points')

    # Try to fit line
    l_fit = fit_line(data, error)
    print "Fitted line C0 = {}, C1={}".format(l_fit[0], l_fit[1])
    plt.plot(data[:, 0], l_fit[0]*data[:,0]+l_fit[1], 'r--', linewidth=2.0, label='fitted line')
    plt.legend(loc='upper left')

    plt.show()

def function_optimizer_poly(degree=3):
    # initial guess
    coefficients_original = np.poly1d([300.0,-300.0,400.0,2000.0])

    # plot initial guess
    x = np.linspace(-5, 5, 21)
    poly_X = x
    poly_Y = np.polyval(coefficients_original, x)
    plt.plot(poly_X, poly_Y, 'b--', linewidth=2.0, label='Original poly')



    # Generate noisy data points
    noise_sigma = 1000.0
    noise = np.random.normal(0, noise_sigma, poly_Y.shape)
    data = np.asarray([poly_X, poly_Y + noise]).T
    plt.plot(data[:,0], data[:,1], 'go', label='data points')

    # Try to fit line
    poly = fit_poly(data, error_ploy, degree=degree)
    plt.plot(data[:, 0], poly(data[:,0]), 'r--', linewidth=2.0, label='fitted poly')
    plt.legend(loc='upper left')

    plt.show()

def fit_line(data, error_func):
    # initial guess
    l = np.float32([0,np.mean(data[:,1])])

    #plot
    x_ends = np.float32([-5,5])
    plt.plot(x_ends, l[0]*x_ends+l[1], 'm--', linewidth=2.0, label='Initial guess')

    # minimization
    min_result = spo.minimize(error_func, l, args=(data,) ,method='SLSQP', options={'disp': True}) # args is passing the data to our error function
    return min_result.x

def fit_poly(data, error_func, degree=3):
    # initial guess
    coefficients_guess = np.poly1d(np.ones(degree + 1, dtype=np.float32))

    #plot initial guess
    x = np.linspace(-5,5,21)
    plt.plot(x, np.polyval(coefficients_guess, x), 'm--', linewidth=2.0, label='Initial guess')

    # minimization
    min_result = spo.minimize(error_func, coefficients_guess, args=(data,) ,method='SLSQP', options={'disp': True}) # args is passing the data to our error function
    return np.poly1d(min_result.x) # convert optimal result into a poly1d object

def get_fitted_portfolio_allocations(df, negative_yearly_sharpe_ratio, initial_allocations=[0.25,0.25,0.25,0.25]):
    # initial guess
    allocation_guess = initial_allocations

    # minimization
    bnds = ((0.0, None),(0.0, None),(0.0, None),(0.0, None))
    cons = ({'type': 'eq', 'fun': constraint})
    min_result = spo.minimize(negative_yearly_sharpe_ratio, allocation_guess, args=(df,), method='SLSQP', bounds=bnds, constraints=cons, options={'disp': True}) # args is passing the data to our error function
    return min_result.x # convert optimal result into a poly1d object

def constraint(x):
    return 1.0 - x[0]-x[1]-x[2]-x[3]

def negative_yearly_sharpe_ratio(allocations, df):
    portf_daily_returns = get_portfolio_values(df, allocations)
    portf_daily_returns = get_daily_returns(portf_daily_returns)
    sharpe_ratio = (252**0.5) * (
                (portf_daily_returns.mean()) / portf_daily_returns.std())
    print allocations, -sharpe_ratio
    return -sharpe_ratio

def error(line, data):
    """
    line: tuple/list/array of two coefficients -- C0 and C_1, the line equation parameters
    """
    err = np.sum((data[:, 1] - (line[0]*data[:,0] + line[1]))**2)
    return err

def error_ploy(coefficients, data):
    err = np.sum((data[:,1]-np.polyval(coefficients, data[:,0]))**2)
    return err

def optimize_allocation_using_sharpe_ratio(df):
    # initial guess
    fig, ax = plt.subplots(2)
    df_for_opt = df.copy()

    # obtaining data
    data = df.reset_index().values
    df = normalize_data(df)
    for column in df.columns:
        ax[0].plot(data[:, 0], df[column], linewidth=1.0, label=column)

    original_allocations = [0.25,0.25,0.25,0.25]
    original_port_values_data = normalize_data(get_portfolio_values(df, alloc=original_allocations)).reset_index().values
    ax[1].plot(original_port_values_data[:,0], original_port_values_data[:,1],linewidth=1.0, label='Alloc:%s'%original_allocations)

    # Try to fit line
    fitted_allocations = list(get_fitted_portfolio_allocations(df_for_opt, negative_yearly_sharpe_ratio, original_allocations)) #course [0.0,0.4, 0.6, 0.0]
    fitted_port_values_data = normalize_data(get_portfolio_values(df,alloc=fitted_allocations)).reset_index().values
    fitted_allocations = ['%.2f' % elem for elem in fitted_allocations]
    ax[1].plot(fitted_port_values_data[:, 0], fitted_port_values_data[:, 1], linewidth=1.0,
               label='Alloc:%s'%fitted_allocations)
    ax[0].legend(loc='upper left')
    ax[1].legend(loc='upper left')

    plt.show()

def add_attributes(df):
    column = 'SPY'
    df['Future Price'] = ((df[column].shift(-7) - df[column]) / (df[column])) * 100
    df['ROC'] = ((df[column] - df[column].shift(9)) / (df[column].shift(9)))
    df['ROC200'] = ((df[column] - df[column].shift(200)) / (df[column].shift(200))) * 100
    rm = df[column].rolling(20).mean()
    df['Avg'] = rm

    rstd = df[column].rolling(20).std()
    df['Bollinger Up'], df['Bollinger Down'] = get_bollinger_bands(rm, rstd)
    return df

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

def price_predictor(df):

    # read https://algotrading101.com/learn/train-test-split/
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error

    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

    df = normalize_data(df)
    df = add_attributes(df)

    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)




    y = df["Future Price"]
    X = df.drop(columns=["Future Price"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
    random_forest = RandomForestRegressor(max_depth=50, n_estimators=100, random_state=1)


    #root_mean_squared_error = np.sqrt(mean_squared_error(y_train, random_forest.predict(X_train)))

    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(random_forest, X_train_scaled, y_train,
                             scoring="neg_mean_squared_error", cv=10)
    tree_rmse_scores = np.sqrt(-scores)
    tree_rmse_scores = tree_rmse_scores/np.abs(y_train).mean()

    display_scores(tree_rmse_scores)

    y_unseen = y_test
    X_unseen = X_test

    random_forest.fit(X_train, y_train)

    root_mean_squared_error = np.sqrt(mean_squared_error(y_unseen, random_forest.predict(X_unseen)))
    print root_mean_squared_error/np.abs(y_unseen).mean()



if __name__=='__main__':

    symbols = ['SPY']
    df = get_df_with_relevant_ticks(symbols)
    price_predictor(df)

    #plot_histogram(df)
    #plot_scatter_plot(df) # this one does a linear regressions as well

    # plot_bollinger_bands(df)

    # portfolio values
    # get_portfolio_values(df, start_val=19000, alloc=[1])

    # optimizers
    # function_optimizer_parabola(f)
    # function_optimizer_line(f)
    # function_optimizer_poly()

    # optimize portfolio allocation
    #optimize_allocation_using_sharpe_ratio(df)  ----> main project part 1

    # more advanced optimization, mean variable optimization of portfolios
    # see https://www.kaggle.com/vijipai/lesson-5-mean-variance-optimization-of-portfolios

