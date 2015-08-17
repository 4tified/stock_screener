__author__ = 'Nigel'

import os, errno
import urllib2
import time
import datetime
import collections
from progressbar import ProgressBar
import ystockquote as ysq
import finsymbols as fin
import pandas
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from matplotlib.finance import candlestick
import matplotlib
import pylab
from Tkinter import Tk
from tkFileDialog import askopenfilename


from matplotlib.dates import datestr2num, WeekdayLocator, DateFormatter
from matplotlib.dates import MO, TU, WE, TH, FR, SA, SU
from matplotlib.ticker import FuncFormatter, MaxNLocator



matplotlib.rcParams.update({'font.size': 9})

evenBetter = ['AWH', 'AFAM', 'ACAS', 'ASI', 'NLY', 'ANH', 'AIZ', 'AXS', 'BHLB', 'COF', 'CMO', 'DYN', 'FDEF', 'HMN', 'IM', 'IRDM', 'KMPR', 'LNC', 'LMIA', 'MANT', 'MCGC', 'MRH', 'MVC', 'NAVG', 'OVTI', 'PRE', 'PMC', 'PJC', 'PTP', 'BPOP', 'PL', 'RF', 'SKYW', 'STI', 'SPN', 'SYA', 'TECD', 'TSYS', 'TITN', 'TPC', 'UNM', 'VOXX', 'XL']


def rsiFunc(prices, n=14):
    deltas = np.diff(prices)
    seed = deltas[:n+1]
    up = seed[seed>=0].sum()/n
    down = -seed[seed<0].sum()/n
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100./(1.+rs)

    for i in range(n, len(prices)):
        delta = deltas[i-1] # the diff is 1 shorter

        if delta>0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(n-1) + upval)/n
        down = (down*(n-1) + downval)/n

        rs = up/down
        rsi[i] = 100. - 100./(1.+rs)

    return rsi

def movingaverage(values,window):
    weigths = np.repeat(1.0, window)/window
    smas = np.convolve(values, weigths, 'valid')
    return smas # as a numpy array

########EMA CALC ADDED############
def ExpMovingAverage(values, window):
    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum()
    a =  np.convolve(values, weights, mode='full')[:len(values)]
    a[:window] = a[window]
    return a


def computeMACD(x, slow=26, fast=12):
    """
    compute the MACD (Moving Average Convergence/Divergence) using a fast and slow exponential moving avg'
    return value is emaslow, emafast, macd which are len(x) arrays
    """
    emaslow = ExpMovingAverage(x, slow)
    emafast = ExpMovingAverage(x, fast)
    return emaslow, emafast, emafast - emaslow

###############################
def graphData(stock,MA1,MA2):
    #######################################
    #######################################
    '''
        Use this to dynamically pull a stock:
    '''
    try:
        print 'Currently Pulling',stock


        netIncomeAr = []
        revAr = []
        ROCAr = []


        endLink = 'sort_order=asc&auth_token=a3fpXxHfsiN7AF4gjakQ'
        try:
            netIncome = urllib2.urlopen('http://www.quandl.com/api/v1/datasets/OFDP/DMDRN_'+stock.upper()+'_NET_INC.csv?&'+endLink).read()
            revenue = urllib2.urlopen('http://www.quandl.com/api/v1/datasets/OFDP/DMDRN_'+stock.upper()+'_REV_LAST.csv?&'+endLink).read()
            ROC = urllib2.urlopen('http://www.quandl.com/api/v1/datasets/OFDP/DMDRN_'+stock.upper()+'_ROC.csv?&'+endLink).read()

            splitNI = netIncome.split('\n')
            print 'Net Income:'
            for eachNI in splitNI[1:-1]:
                print eachNI
                netIncomeAr.append(eachNI)

            print '_________'
            splitRev = revenue.split('\n')
            print 'Revenue:'
            for eachRev in splitRev[1:-1]:
                print eachRev
                revAr.append(eachRev)


            print '_________'
            splitROC = ROC.split('\n')
            print 'Return on Capital:'
            for eachROC in splitROC[1:-1]:
                print eachROC
                ROCAr.append(eachROC)


            incomeDate, income = np.loadtxt(netIncomeAr, delimiter=',',unpack=True,
                                            converters={ 0: mdates.strpdate2num('%Y-%m-%d')})

            revDate, revenue = np.loadtxt(revAr, delimiter=',',unpack=True,
                                            converters={ 0: mdates.strpdate2num('%Y-%m-%d')})

            rocDate, ROC = np.loadtxt(ROCAr, delimiter=',',unpack=True,
                                            converters={ 0: mdates.strpdate2num('%Y-%m-%d')})

        except Exception, e:
            print 'failed in the quandl grab'
            print str(e)
            time.sleep(555)






        print str(datetime.datetime.fromtimestamp(int(time.time())).strftime('%Y-%m-%d %H:%M:%S'))
        #Keep in mind this is close high low open, lol.
        urlToVisit = 'http://chartapi.finance.yahoo.com/instrument/1.0/'+stock+'/chartdata;type=quote;range=10y/csv'
        stockFile =[]
        try:
            sourceCode = urllib2.urlopen(urlToVisit).read()
            splitSource = sourceCode.split('\n')
            for eachLine in splitSource:
                splitLine = eachLine.split(',')
                if len(splitLine)==6:
                    if 'values' not in eachLine:
                        stockFile.append(eachLine)
        except Exception, e:
            print str(e), 'failed to organize pulled data.'
    except Exception,e:
        print str(e), 'failed to pull pricing data'
    #######################################
    #######################################
    try:
        date, closep, highp, lowp, openp, volume = np.loadtxt(stockFile,delimiter=',', unpack=True,
                                                              converters={ 0: mdates.strpdate2num('%Y%m%d')})
        x = 0
        y = len(date)
        newAr = []
        while x < y:
            appendLine = date[x],openp[x],closep[x],highp[x],lowp[x],volume[x]
            newAr.append(appendLine)
            x+=1

        Av1 = movingaverage(closep, MA1)
        Av2 = movingaverage(closep, MA2)

        SP = len(date[MA2-1:])

        fig = plt.figure(facecolor='#07000d')

        ax1 = plt.subplot2grid((9,4), (1,0), rowspan=4, colspan=4, axisbg='#07000d')
        candlestick(ax1, newAr[-SP:], width=.6, colorup='#53c156', colordown='#ff1717')

        Label1 = str(MA1)+' SMA'
        Label2 = str(MA2)+' SMA'

        ax1.plot(date[-SP:],Av1[-SP:],'#e1edf9',label=Label1, linewidth=1.5)
        ax1.plot(date[-SP:],Av2[-SP:],'#4ee6fd',label=Label2, linewidth=1.5)

        ax1.grid(True, color='w')
        ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.yaxis.label.set_color("w")
        ax1.spines['bottom'].set_color("#5998ff")
        ax1.spines['top'].set_color("#5998ff")
        ax1.spines['left'].set_color("#5998ff")
        ax1.spines['right'].set_color("#5998ff")
        ax1.tick_params(axis='y', colors='w')
        plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))
        ax1.tick_params(axis='x', colors='w')
        plt.ylabel('Stock price and Volume')

        maLeg = plt.legend(loc=9, ncol=2, prop={'size':7},
                   fancybox=True, borderaxespad=0.)
        maLeg.get_frame().set_alpha(0.4)
        textEd = pylab.gca().get_legend().get_texts()
        pylab.setp(textEd[0:5], color = 'w')

        volumeMin = 0

        ax0 = plt.subplot2grid((9,4), (0,0), sharex=ax1, rowspan=1, colspan=4, axisbg='#07000d')
        rsi = rsiFunc(closep)
        rsiCol = '#c1f9f7'
        posCol = '#386d13'
        negCol = '#8f2020'

        ax0.plot(date[-SP:], rsi[-SP:], rsiCol, linewidth=1.5)
        ax0.axhline(70, color=negCol)
        ax0.axhline(30, color=posCol)
        ax0.fill_between(date[-SP:], rsi[-SP:], 70, where=(rsi[-SP:]>=70), facecolor=negCol, edgecolor=negCol, alpha=0.5)
        ax0.fill_between(date[-SP:], rsi[-SP:], 30, where=(rsi[-SP:]<=30), facecolor=posCol, edgecolor=posCol, alpha=0.5)
        ax0.set_yticks([30,70])
        ax0.yaxis.label.set_color("w")
        ax0.spines['bottom'].set_color("#5998ff")
        ax0.spines['top'].set_color("#5998ff")
        ax0.spines['left'].set_color("#5998ff")
        ax0.spines['right'].set_color("#5998ff")
        ax0.tick_params(axis='y', colors='w')
        ax0.tick_params(axis='x', colors='w')
        plt.ylabel('RSI')

        ax1v = ax1.twinx()
        ax1v.fill_between(date[-SP:],volumeMin, volume[-SP:], facecolor='#00ffe8', alpha=.4)
        ax1v.axes.yaxis.set_ticklabels([])
        ax1v.grid(False)
        ###Edit this to 3, so it's a bit larger
        ax1v.set_ylim(0, 3*volume.max())
        ax1v.spines['bottom'].set_color("#5998ff")
        ax1v.spines['top'].set_color("#5998ff")
        ax1v.spines['left'].set_color("#5998ff")
        ax1v.spines['right'].set_color("#5998ff")
        ax1v.tick_params(axis='x', colors='w')
        ax1v.tick_params(axis='y', colors='w')
        ax2 = plt.subplot2grid((9,4), (5,0), sharex=ax1, rowspan=1, colspan=4, axisbg='#07000d')
        fillcolor = '#00ffe8'
        nslow = 26
        nfast = 12
        nema = 9
        emaslow, emafast, macd = computeMACD(closep)
        ema9 = ExpMovingAverage(macd, nema)
        ax2.plot(date[-SP:], macd[-SP:], color='#4ee6fd', lw=2)
        ax2.plot(date[-SP:], ema9[-SP:], color='#e1edf9', lw=1)
        ax2.fill_between(date[-SP:], macd[-SP:]-ema9[-SP:], 0, alpha=0.5, facecolor=fillcolor, edgecolor=fillcolor)

        plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))
        ax2.spines['bottom'].set_color("#5998ff")
        ax2.spines['top'].set_color("#5998ff")
        ax2.spines['left'].set_color("#5998ff")
        ax2.spines['right'].set_color("#5998ff")
        ax2.tick_params(axis='x', colors='w')
        ax2.tick_params(axis='y', colors='w')
        plt.ylabel('MACD', color='w')
        ax2.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, prune='upper'))

        ######################################
        ######################################
        ax3 = plt.subplot2grid((9,4), (6,0), sharex=ax1, rowspan=1, colspan=4, axisbg='#07000d')
        ax3.plot(incomeDate,income,'#4ee6fd')
        ax3.spines['bottom'].set_color("#5998ff")
        ax3.spines['top'].set_color("#5998ff")
        ax3.spines['left'].set_color("#5998ff")
        ax3.spines['right'].set_color("#5998ff")
        ax3.tick_params(axis='x', colors='w')
        ax3.tick_params(axis='y', colors='w')
        ax3.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, prune='upper'))
        plt.ylabel('NI', color='w')

        ax4 = plt.subplot2grid((9,4), (7,0),sharex=ax1, rowspan=1, colspan=4, axisbg='#07000d')
        ax4.plot(revDate, revenue,'#4ee6fd')
        ax4.spines['bottom'].set_color("#5998ff")
        ax4.spines['top'].set_color("#5998ff")
        ax4.spines['left'].set_color("#5998ff")
        ax4.spines['right'].set_color("#5998ff")
        ax4.tick_params(axis='x', colors='w')
        ax4.tick_params(axis='y', colors='w')
        ax4.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, prune='upper'))
        plt.ylabel('Rev', color='w')

        ax5 = plt.subplot2grid((9,4), (8,0), rowspan=1, sharex=ax1, colspan=4, axisbg='#07000d')
        ax5.plot(rocDate, ROC,'#4ee6fd')
        ax5.spines['bottom'].set_color("#5998ff")
        ax5.spines['top'].set_color("#5998ff")
        ax5.spines['left'].set_color("#5998ff")
        ax5.spines['right'].set_color("#5998ff")
        ax5.tick_params(axis='x', colors='w')
        ax5.tick_params(axis='y', colors='w')
        ax5.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, prune='upper'))
        plt.ylabel('ROC', color='w')




        for label in ax5.xaxis.get_ticklabels():
            label.set_rotation(45)

        plt.suptitle(stock,color='w')

        plt.setp(ax0.get_xticklabels(), visible=False)
        ### add this ####
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.setp(ax3.get_xticklabels(), visible=False)
        plt.setp(ax4.get_xticklabels(), visible=False)



        plt.subplots_adjust(left=.09, bottom=.14, right=.94, top=.95, wspace=.20, hspace=0)
        plt.show()
        fig.savefig('example.png',facecolor=fig.get_facecolor())

    except Exception,e:
        print 'main loop',str(e)









def screener(stock):
    try:
        #print 'doing',stock
        sourceCode = urllib2.urlopen('http://finance.yahoo.com/q/ks?s='+stock).read()
        pbr = sourceCode.split('Price/Book (mrq):</td><td class="yfnc_tabledata1">')[1].split('</td>')[0]
        #print 'price to book ratio:',stock,pbr

        if float(pbr) < 1:
            #print 'price to book ratio:',stock,pbr

            PEG5 = sourceCode.split('PEG Ratio (5 yr expected)<font size="-1"><sup>1</sup></font>:</td><td class="yfnc_tabledata1">')[1].split('</td>')[0]
            if 0 < float(PEG5) < 2:
                #print 'PEG forward 5 years',PEG5


                DE = sourceCode.split('Total Debt/Equity (mrq):</td><td class="yfnc_tabledata1">')[1].split('</td>')[0]
                #print 'Debt to Equity:',DE
                #if 0 < float(DE) < 2:
                PE12 = sourceCode.split('Trailing P/E (ttm, intraday):</td><td class="yfnc_tabledata1">')[1].split('</td>')[0]
                #print 'Trailing PE (12mo):',PE12
                if float(PE12) < 15:
                    # Your own SCREENED array....
                    #evenBetter.append(stock)

                    print '______________________________________'
                    print ''
                    print stock,'meets requirements'
                    print 'price to book:',pbr
                    print 'PEG forward 5 years',PEG5
                    print 'Trailing PE (12mo):',PE12
                    print 'Debt to Equity:',DE
                    print '______________________________________'

                    if showCharts.lower() == 'y':

                        try:
                            graphData(stock,25,50)

                        except Exception, e:
                            print 'failed the main quandl loop for reason of',str(e)



    except Exception,e:
        #print 'failed in the main loop',str(e)
        pass


class Figure(object):

    #---------------------------------------------------------------------------
    #                                 __init__
    #---------------------------------------------------------------------------
    def __init__(self, figSize=(5.0, 3.0), dpi=300):

        # Initialize class variables
        self.numPlots = 0
        self._plotDescriptions = []

        # Create a figure
        self._figure = pylab.Figure(figsize=figSize, dpi=dpi)

        # Perform some figure setup
        pylab.hold(True)
        pylab.grid(True)

        # Add a formatter for the y-axis
        self._formatter = FuncFormatter(self._formatDollars)

        # Create a SetTitle method
        self.SetTitle = pylab.title

        return


    #---------------------------------------------------------------------------
    #                                addPlot
    #---------------------------------------------------------------------------
    def addPlot(self, dates, prices, style='b-', desc='', format='%m/%d/%Y'):
        '''
        This method allows different plots to be added to a figure. This method
        is basically a wrapper for the pylab.plot_date method, but it includes
        some other special formatting to make things look pretty.

        Arguments:
            * dates (list)  - A list of datetime objects
            * prices (list) - A list of float values (ints are also acceptible)
            * style (str)   - This is the line style for the plot. See the
                              MatPlotLib documentation for the "plot" method for
                              more information.
            * desc (str)    - A description of the plot. This is used to create
                              a legend if more than one plot is added to the
                              figure.
            * format (str)  - This follows a strftime format. This is the
                              format of the x-axis labels.
        '''

        # Convert the dates to a pylab-friendly format
        dates = datestr2num(dates)

        # Create a date-based plot
        pylab.plot_date(dates, prices, style)

        # Get the current axis
        ax = pylab.gca()

        # Rotate the x-axis labels by 90 degrees
        pylab.xticks(rotation=90)

        # Make room for the labels
        pylab.subplots_adjust(bottom=0.20)

        # Format the date labels on the x-axis
        #ax.xaxis.set_major_locator( WeekdayLocator(byweekday=FR) )
        #ax.xaxis.set_major_formatter( DateFormatter('%a - %m/%d/%Y') )
        ax.xaxis.set_major_formatter( DateFormatter(format) )

        # Find out how many x-tick marks to use
        numTicks = self._getXTicks(len(dates))

        # Set the number of x-axis tick marks
        ax.xaxis.set_major_locator(MaxNLocator(numTicks))

        # Limit the x-axis
        pylab.xlim([dates[0], dates[-1]])

        # Format the y-axis for money
        ax.yaxis.set_major_formatter(self._formatter)

        # Set the number of y-axis tick marks
        ax.yaxis.set_major_locator(MaxNLocator(10))

        # Add the description to the list of plot descriptions
        self._plotDescriptions.append(desc)

        # Increment the plot counter
        self.numPlots += 1

        # See if we should add a legend
        if self.numPlots > 1:
            pylab.legend(self._plotDescriptions, loc='best')

        return


    #---------------------------------------------------------------------------
    #                                _getXTicks
    #---------------------------------------------------------------------------
    def _getXTicks(self, numDates):
        ''' Helper method to get the number of X-tick marks. '''
        N = 0
        if numDates <= 15:
            N = numDates
        elif numDates <= 100:
            N = 15
        else:
            N = max(int(numDates / 10.), 15)
        return N


    #---------------------------------------------------------------------------
    #                                addDatePlot
    #---------------------------------------------------------------------------
    def addDatePlot(self, dates, prices, style='b-', description=''):
        '''
        This is a simple wrapper for the "addPlot" method. This method just
        makes it easier to create date-based plots, where the x-axis is
        labeled as a date, not a time.
        '''
        self.addPlot(dates, prices, style, description, format='%m/%d/%Y')
        return


    #---------------------------------------------------------------------------
    #                                addTimePlot
    #---------------------------------------------------------------------------
    def addTimePlot(self, times, prices, style='b-', description=''):
        '''
        This is a simple wrapper for the "addPlot" method. This method just
        makes it easier to create time-based plots, where the x-axis is
        labeled as a time, not a date.
        '''
        self.addPlot(times, prices, style, description, format='%I:%M:%S %p')
        return


    #---------------------------------------------------------------------------
    #                              _formatDollars
    #---------------------------------------------------------------------------
    def _formatDollars(self, value, tickPos):
        ''' Helper method to format the y-axis labels. '''
        return '$%.2f' % value


    #---------------------------------------------------------------------------
    #                                  show
    #---------------------------------------------------------------------------
    def show(self):
        ''' This is a blocking method, until the window is closed. '''
        # Show the figure
        pylab.show()
        return


    #---------------------------------------------------------------------------
    #                                  save
    #---------------------------------------------------------------------------
    def save(self, filePath='stockPlot.pdf', dpi=100):
        ''' This method saves the stock plot to a file. '''
        pylab.savefig(filePath, dpi=dpi)
        return


class OwnedStocks:
    def __init__(self):
        self.stock_list = []
        self.owned_submenu()

    def owned_submenu(self):

        prompt = raw_input('What to do? (add, remove, view, main): ')
        if 'add' in prompt:
            self.add()
        if 'remove' in prompt:
            self.remove()
        if 'view' in prompt:
            self.view()
        if 'main' in prompt:
            return
        self.owned_submenu()

    def read_file(self, file_name):
        self.stock_list = []
        print file_name
        #if os.path.isfile("saved/" + file_name):

        target = open("saved/" + file_name, 'a')
        pull_list = target.read()
        print "test"
        print pull_list
        target.close()
        pull_list = pull_list.split('@')
        #del pull_list[(len(pull_list) - 1)]
        for entry in pull_list:
            entry = entry.split('|')
            self.stock_list.append([entry[0], entry[1], entry[2], entry[3], entry[4], entry[5]])
        #else:
        #    print "File not found!"
    def write_file(self, name_file, save_list):
        target = open("saved/" + name_file + '.stk', "w")
        for entry in save_list:
            target.write(entry[0] + '|' + entry[1] + '|' + entry[2] + '|' + entry[3] + '|' + entry[4] + '|' + entry[5] + '@')

        target.close

    def add_entry(self, name_file, copy_list):
        ticker_list = []
        for copy_entry in copy_list:
            ticker_list.append(copy_entry[0])
        print "Owned Stocks: "
        print ticker_list
        entry_prompt = raw_input('What to do? (new, quit): ')
        if 'new' in entry_prompt:
            stock_ticker = raw_input('Stock ticker: ')
            stock_amount = raw_input('Number of shares: ')
            stock_price = raw_input('Price bought at: ')
            stock_date = raw_input('Date purchase (yyyy-mm-dd): ')
            stock_notes = raw_input('Notes: ')
            stock_sell = raw_input('Intended sell off price per stock: ')
            print "Stock Entered"
            copy_list.append([stock_ticker, stock_amount, stock_price, stock_date, stock_notes, stock_sell])
            self.add_entry(name_file, copy_list)
        if 'quit' in entry_prompt:
            save_prompt = raw_input('Save? (y/n): ')
            if 'y' in save_prompt:
                self.write_file(name_file, copy_list)
                print "File Saved"
            if 'n' in save_prompt:
                self.write_file(name_file, self.stock_list)
                print "No Changes Made"
            return
    def add(self):
        add_prompt = raw_input('What to do? (new, saved): ')
        if 'new' in add_prompt:
            name_file = raw_input('Save File Name As: ')
            copy_list = self.stock_list
        if 'saved' in add_prompt:

            Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
            filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
            name_file = filename.split('.stk')[0]
            self.read_file(filename)
            copy_list = self.stock_list
        self.add_entry(name_file, copy_list)

class ScreenTickers:
    def __init__(self, ticker_list, stock_ex):
        self.ticker_list = ticker_list
        self.exchange = stock_ex
        self.rising = []
        self.dipping = []
        self.rising_three = []
        self.dipping_three = []
        self.min_rising = []
        self.max_dipping = []
        self.just_rising = []
        self.error_list = []
        self.start_screen()



    def start_screen(self):
        to_do = raw_input('What to do? (pull, import, view, filter, owned, watchlist): ')
        if 'pull' in to_do:
            self.percent_change()
        if 'import' in to_do:
            self.import_list()
        if 'view' in to_do:
            self.view_data()
        if 'owned' in to_do:
            owned_sub = OwnedStocks()
        if 'watchlist' in to_do:
            self.watchlist()
        self.start_screen()





    def get_price_history(self, ticker, start_date, end_date):
        """
        example usage:
        get_price_history('appl','2014-11-01','2014-11-30')
        returns a pandas dataframe

        """
        data = ysq.get_historical_prices(ticker, start_date, end_date)
        df = pandas.DataFrame(collections.OrderedDict(sorted(data.items()))).T
        df = df.convert_objects(convert_numeric=True)
        return df

    def local_maxima(self, xval, yval):
        xval = np.asarray(xval)
        yval = np.asarray(yval)

        sort_idx = np.argsort(xval)
        yval = yval[sort_idx]
        gradient = np.diff(yval)
        #print gradient
        maxima = np.diff((gradient > 0).view(np.int8))
        #print maxima
        return np.concatenate((([0],) if gradient[0] < 0 else ()) +
                              (np.where(maxima == -1)[0] + 1,) +
                              (([len(yval)-1],) if gradient[-1] > 0 else ()))

    def write_cache(self):
        rising_write = ""
        up_three_write = ""
        dipping_write = ""
        down_three_write = ""
        date_today = datetime.date.today().strftime("%m.%d.%Y")
        target = open("cache/" + date_today + "_" + self.exchange + ".dat", "w")
        target.write("rising\n")
        for rise in self.rising:
            rising_write = rising_write + rise[0] + '|' + rise[1] + '\n'
        target.write(rising_write + '\n' + '?')
        target.write("upthree\n")
        for up_three in self.rising_three:
            up_three_write = up_three_write + up_three[0] + '|' + up_three[1] + '\n'
        target.write(up_three_write + '\n' + '?')
        target.write("dipping\n")
        for dip in self.dipping:
            dipping_write = dipping_write + dip[0] + '|' + dip[1] + '\n'
        target.write(dipping_write + '\n' + '?')
        target.write("downthree\n")
        for down_three in self.dipping_three:
            down_three_write = down_three_write + down_three[0] + '|' + down_three[1] + '\n'
        target.write(dipping_write + '\n' + '?')
        target.close()

    def write_just_rising(self):
        just_rising_write = ""
        date_today = datetime.date.today().strftime("%m.%d.%Y")
        target = open("cache/" + date_today + "_" + self.exchange + ".dat", "a")
        target.write("just_rising\n")
        for just_rise in self.just_rising:
            just_rising_write = just_rising_write + str(just_rise[0]) + '|' + str(just_rise[1]) + '\n'
        target.write(just_rising_write + '\n' + '?')
        target.close()
    def import_list(self):
        date_today = datetime.date.today().strftime("%m.%d.%Y")
        if os.path.isfile("cache/" + date_today + "_" + self.exchange + ".dat"):

            target = open("cache/" + date_today + "_" + self.exchange + ".dat", 'r')
            pull_list = target.read()
            just_rising_list = pull_list.split('just_rising\n')[1].split('\n?')[0]
            just_rising_temp = just_rising_list.split(';')
            del just_rising_temp[(len(just_rising_temp) - 1)]
            for just_temp_unit in just_rising_temp:
                just_split = just_temp_unit.split('|')
                self.just_rising.append([just_split[0],just_split[1]])
            print "List imported"
        else:
            print "No list to import!"
        return

    def get_cache(self):
        date_today = datetime.date.today().strftime("%m.%d.%Y")
        if os.path.isfile("cache/" + date_today + "_" + self.exchange + ".dat"):

            target = open("cache/" + date_today + "_" + self.exchange + ".dat", 'r')
            pull_list = target.read()
            target.close()
            rise_list = pull_list.split("rising\n")[1].split('\n?')[0]
            up_three_list = pull_list.split('upthree\n')[1].split('\n?')[0]
            dip_list = pull_list.split('dipping\n')[1].split('\n?')[0]
            down_three_list = pull_list.split('downthree\n')[1].split('\n?')[0]
            rising_temp = rise_list.split('\n')
            del rising_temp[(len(rising_temp) - 1)]
            for rising_temp_unit in rising_temp:
                rising_split = rising_temp_unit.split('|')
                self.rising.append([rising_split[0],rising_split[1]])
            rising_three_temp = up_three_list.split('\n')
            del rising_three_temp[(len(rising_three_temp) - 1)]
            for rising_three_unit in rising_three_temp:
                up_three_split = rising_three_unit.split('|')
                self.rising_three.append([up_three_split[0],up_three_split[1]])
            dipping_temp = dip_list.split('\n')
            del dipping_temp[(len(dipping_temp) - 1)]
            for dipping_temp_unit in dipping_temp:
                dipping_split = dipping_temp_unit.split('|')
                self.dipping.append([dipping_split[0],dipping_split[1]])
            dipping_three_temp = down_three_list.split('\n')
            del dipping_three_temp[(len(dipping_three_temp) - 1)]
            for dipping_three_unit in dipping_three_temp:
                down_three_split = dipping_three_unit.split('|')
                self.dipping_three.append([down_three_split[0],down_three_split[1]])


        return



    def local_minima(self, xval, yval):
        xval = np.asarray(xval)
        yval = np.asarray(yval)

        sort_idx = np.argsort(xval)
        yval = yval[sort_idx]
        gradient = np.diff(yval)
        #print gradient
        minima= np.diff((gradient > 0).view(np.int8))
        #print minima
        return np.concatenate((([0],) if gradient[0] > 0 else ()) +
                              (np.where(minima == 1)[0] + 1,) +
                              (([len(yval)-1],) if gradient[-1] < 0 else ()))



    def percent_change(self):
        tick_data = []
        repull_all = raw_input('Would you like to pull ticker data? (Y/N): ')
        ask_print = raw_input('Show after pull? (Y/N): ')
        if 'y' in repull_all:
            pbar = ProgressBar()
            for ticker in pbar(self.ticker_list):
                #print str(ticker_list.index(ticker) + 1) + ' / ' + str(len(ticker_list)) + '  ' + str((float(ticker_list.index(ticker) + 1) / len(ticker_list))) + '%'

                try:
                    tick_data = ysq.get_all(ticker)
                    percentage = (float(tick_data['change']) / (float(tick_data['price']) - float(tick_data['change']))) * 100

                    if percentage > 0:
                        self.rising.append([ticker,tick_data['price']])
                    if percentage > 3:
                        self.rising_three.append([ticker,tick_data['price']])
                    if percentage < 0:
                        self.dipping.append([ticker,tick_data['price']])
                    if percentage < -3:
                        self.dipping_three.append([ticker,tick_data['price']])

                except Exception, e:
                    #print "Data Pull Error For: " + str(ticker)
                    self.error_list.append(ticker)
                    pass
            self.write_cache()

        if 'n' in repull_all:
            self.get_cache()

        if 'y' in ask_print:
            print "Stocks up: "
            print str(len(self.rising)) + '/' + str(len(self.rising) + len(self.dipping)) + ' - ' + str((float(len(self.rising)) / (len(self.rising) + len(self.dipping))) * 100) + '%'
            print "Stocks down: "
            print str(len(self.dipping)) + '/' + str(len(self.rising) + len(self.dipping)) + ' - ' + str((float(len(self.dipping)) / (len(self.rising) + len(self.dipping))) * 100) + '%'
            print "Rising Three Percent:"
            print str(len(self.rising_three)) + '/' + str(len(self.rising) + len(self.dipping)) + ' - ' + str((float(len(self.rising_three)) / (len(self.rising) + len(self.dipping))) * 100) + '%'
            print "Dipping Three Percent"
            print str(len(self.dipping_three)) + '/' + str(len(self.rising) + len(self.dipping)) + ' - ' + str((float(len(self.dipping_three)) / (len(self.rising) + len(self.dipping))) * 100) + '%'

        err_ticks = []
        period_length = raw_input('Time Period (in days): ')
        pbar = ProgressBar()
        for jr_split in pbar(self.rising):
            pos = jr_split[0]
            today = datetime.date.today().strftime("%m/%d/%Y")
            dates = pandas.bdate_range(end=today, periods=int(period_length) + 1)
            yesterday = dates[len(dates) - 1].strftime("%Y/%m/%d")
            first_date = dates[0].strftime("%Y/%m/%d")
            try:
                hist_data = self.get_price_history(pos,first_date,yesterday)['Adj Close']

                now_price = ysq.get_price(pos)
                points_list = []
                index_num = 1
                for val in hist_data:
                    points_list.append([index_num, val])
                    index_num += 1
                #print points_list
                points_list = np.array(points_list)

                x = list(range(1, len(hist_data) + 1))
                y = hist_data

                max_data = self.local_maxima(x, y)
                min_data = self.local_minima(x, y)

                min_list = []
                for x_val in min_data:
                    min_list.append(hist_data[x_val])
                #print min_list
                minimum_price = min(min_list)
                #print pos
                #print now_price
                #print minimum_price
                percent_rise = ((float(now_price) - float(minimum_price)) / float(minimum_price)) * 100
                if 1 < percent_rise < 5:
                    self.just_rising.append([pos,now_price])

                # calculate polynomial


            except Exception, e:
                #print "Error Pulling Historical Data For: " + str(pos)
                err_ticks.append(pos)
                #raw_input('Hold: ')
                pass

        # print 'Just Rising:'
        # print self.just_rising
        # print 'Errors:'
        # print self.error_list
        # print 'Errors For Rising'
        # print err_ticks
        self.write_just_rising()

    def view_data(self):
        print "Tickers to view:"
        print str(len(self.just_rising))
        print self.just_rising
        confirm_view = raw_input('Show all tickers?:(y/n)')
        if 'y' in confirm_view:
            date_range = raw_input('Date range to show (in days)')
            for just_rising in self.just_rising:
                today = datetime.date.today().strftime("%m/%d/%Y")
                today_frame = datetime.date.today().strftime("%Y-%m-%d")
                today_print = datetime.date.today().strftime("%m.%d.%Y")
                dates = pandas.bdate_range(end=today, periods=int(date_range) + 1)
                yesterday = dates[len(dates) - 1].strftime("%Y/%m/%d")
                first_date = dates[0].strftime("%Y/%m/%d")
                try:
                    hist_data = self.get_price_history(just_rising[0],first_date,yesterday)

                    now_price = ysq.get_price(just_rising[0])
                    points_list = []
                    index_num = 1
                    for val in hist_data:
                        points_list.append([index_num, val])
                        index_num += 1
                    #print points_list
                    points_list = np.array(points_list)

                    x = list(range(1, len(hist_data) + 1))
                    y = hist_data


                    df2 = pandas.DataFrame([[now_price]], index=[today_frame])
                    df2 = df2.convert_objects(convert_numeric=True)
                    #df2.name(str(today_frame))
                    hist_data_prices = hist_data['Adj Close'].append(df2)
                    print hist_data_prices
                    # Break the data into GOOG and AAPL
                    #googData = histData['GOOG']
                    #aaplData = histData['AAPL']

                    # Create a vectors with date/time stamps and stock prices (Google)
                    googDTS = []
                    googPrices = []
                    googHigh = []
                    googLow = []
                    googOpen = []
                    #for quote in hist_data:
                    googHigh = hist_data['High']
                    googLow = hist_data['Low']
                    googDTS = list(hist_data_prices.index)
                    googPrices = hist_data_prices
                    googOpen = hist_data['Open']

                    # Create a vectors with date/time stamps and stock prices (Apple)
                    #aaplDTS = []
                    #aaplPrices = []
                    #for quote in aaplData:
                    #    aaplDTS.append(quote.date)
                    #    aaplPrices.append(quote.adjClose)


                    #---------------------------------------------------------------------------
                    #  Test the plotting functionality...
                    #---------------------------------------------------------------------------

                    # Create a figure
                    stockFig = Figure()

                    # Add the plots
                    stockFig.addDatePlot(googDTS, googPrices, 'b-o', just_rising[0])
                    #stockFig.addDatePlot(googDTS, googPrices, 'b-o', ticker)
                    #stockFig.addDatePlot(googDTS, googOpen, 'g-o', 'Opening Price')

                    #stockFig.addDatePlot(googDTS, googHigh, 'g--o', 'High')
                    #stockFig.addDatePlot(googDTS, googLow, 'r--o', 'Low')

                    #stockFig.addDatePlot(aaplDTS, aaplPrices, 'r-', 'AAPL')

                    # Set a title for the stock plot
                    stockFig.SetTitle(just_rising[0])

                    # Save the figure
                    #stockFig.save('GOOG_vs_AAPL.png')

                    # Show the figure
                    stockFig.show()

                    #max_data = self.local_maxima(x, y)
                    #min_data = self.local_minima(x, y)

                    #min_list = []
                    #for x_val in min_data:
                    #    min_list.append(hist_data[x_val])
                    #print min_list
                    #minimum_price = min(min_list)
                    #print pos
                    #print now_price
                    #print minimum_price
                    #percent_rise = ((float(now_price) - float(minimum_price)) / float(minimum_price)) * 100
                    #if 1 < percent_rise < 5:
                    #    self.just_rising.append(pos)

                    # calculate polynomial

                    # z = np.polyfit(x, y, 3)
                    # f = np.poly1d(z)
                    # #print f
                    # # calculate new x's and y's
                    # x_new = np.linspace(x[0], x[-1], 50)
                    # y_new = f(x_new)
                    # plt.suptitle(just_rising[0],color='w')
                    # plt.plot(x,y,'o', x_new, y_new)
                    # plt.xlim([x[0]-1, x[-1] + 1 ])
                    # plt.show()










                except Exception, e:
                    print "Error Pulling Historical Data For: " + str(just_rising[0])
                    pass

                #raw_input('Press enter for next stock')





        if 'n' in confirm_view:
            return
        return



showCharts = raw_input('Would you like to show the financial data (Quandl) charts? (Y/N): ')

if showCharts.lower()=='y':
    print 'okay, charts will be shown'

elif showCharts.lower()=='n':
    print 'okay, charts will NOT be shown.'

else:
    print 'invalid input, charts will NOT be shown.'


try:
    os.makedirs("cache/")
except OSError, e:
    if e.errno != errno.EEXIST:
        raise
try:
    os.makedirs("saved/")
except OSError, e:
    if e.errno != errno.EEXIST:
        raise
stock_ex = raw_input('Which stock market would you like to screen: ')
repull_data = raw_input('Pull stock tickers names(y/n): ')

if 'y' in repull_data:
    if 'nyse' in stock_ex:
        stock_list = fin.get_nyse_symbols()
    if 'sp500' in stock_ex:
        stock_list = fin.get_sp500_symbols()
    if 'nasdaq' in stock_ex:
        stock_list = fin.get_nasdaq_symbols()
    if 'amex' in stock_ex:
        stock_list = fin.get_amex_symbols()
    ticker_list = []
    for stock_n in stock_list:
        ticker_list.append(str(stock_n['symbol']))
    target = open(stock_ex + '_list.txt', 'w')
    for ticker in ticker_list:
        target.write(ticker + '\n')
    target.close()
if 'n' in repull_data:
    ticker_list = []
    target = open(stock_ex + '_list.txt', 'r')
    pull_list = target.readlines()
    target.close()
    for pulled in pull_list:
        pulled = pulled.split('\n')
        ticker_list.append(pulled[0])


screen = ScreenTickers(ticker_list, stock_ex)

