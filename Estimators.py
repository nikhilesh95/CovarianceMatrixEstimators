import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.covariance import LedoitWolf, EmpiricalCovariance
from collections import defaultdict

''' 
    Class DateRange
    - Used to define start and end dates of in-sample and out-of-sample data ranges
'''
class DateRange:
    def __init__(self, inStartDate, inEndDate, outStartDate, outEndDate):
        self.inStartDate = inStartDate      
        self.inEndDate = inEndDate 
        self.outStartDate = outStartDate
        self.outEndDate = outEndDate

        
'''
    Class Estimator
    - Process input files and estimate co-variance matrix from in-sample data based on the chosen estimator.
    - Currently supports 3 different estimators, listed in self.fn_list
'''
class Estimator:
    def __init__(self, filename, dateRange, estimationFn, frequency):
        self.filename = filename            #Pandas Dataframe
        self.dateRange = dateRange
        self.estimationFn = estimationFn    #Which estimator to use
        self.frequency = frequency
        
        self.rawData = None
        self.data = None
        self.cov = None                     #the output covariance matrix
        self.expectedCovShape = None

        self.fn_list = {
            'sample'                : self.estimatorSample,
            'ledoit-wolf'           : self.estimatorLedoitWolf,
            'aggregate'             : self.estimatorAggregate
        }

        self.processInput()

    def processInput(self):
        self.rawData = pd.read_csv(self.filename, parse_dates=['Date'])


        #change frequency to monthly if needed
        if (self.frequency=='M'):
            # print("Changing to monthly data:")
            self.rawData = self.rawData.set_index('Date')
            self.rawData = self.rawData.resample('M').sum()
            self.rawData = self.rawData.reset_index()

        #trim data to in-sample range
        mask = (self.rawData['Date'] > self.dateRange.inStartDate) & (self.rawData['Date'] <= self.dateRange.inEndDate)
        self.data = self.rawData.loc[mask]

        #Calculate number of rows and columns of data, discounting labels
        numRows = self.data.shape[0] - 1
        numColumns = self.data.shape[1] - 1

        #Check that output matrix has the right shape
        self.expectedCovShape = (numColumns,numColumns)

    #Calls the appropriated estimator function
    def estimate(self):
        #return result of called function back to main
        return ( self.fn_list[self.estimationFn]() )    

    #Sample co-variance
    def estimatorSample(self):
        #remove Date column for this function
        trimmedData = self.data.drop('Date', axis=1)
        cov = EmpiricalCovariance().fit(trimmedData).covariance_ #centers the data
        assert cov.shape == self.expectedCovShape
        self.cov = cov
        return self.cov

    #Ledoit-Wolf shrinkage co-variance estimator
    def estimatorLedoitWolf(self):
        #remove Date column for this function
        trimmedData = self.data.drop('Date', axis=1)
        cov = LedoitWolf().fit(trimmedData).covariance_ #centers the data
        assert cov.shape == self.expectedCovShape
        self.cov = cov
        return self.cov

    #Equally weighted aggregate of previous estimators
    def estimatorAggregate(self):
        n = 0
        cov = np.zeros(self.expectedCovShape)
        for fn in self.fn_list:
            if(fn == "aggregate"):
                continue
            n = n+1
            cov = np.add(cov, ( self.fn_list[fn]() ))
        cov = cov / n
        assert cov.shape == self.expectedCovShape
        self.cov = cov
        return cov


''' 
    Class: Portfolio
    - Generates minimum variance portfolio from Estimator object.
    - Calculates its total value, standard deviation, and the
      standard deviation of each asset at the end of the out-of-sample period.
'''
class Portfolio:
    def __init__(self, estimator):
        self.estimator = estimator
        self.minVarWeights = np.zeros(estimator.cov.shape[1])
        self.portfolioValue = 0
        self.portfolioRisk = 0
        self.assetRisks = None

    #Calculate the minimum variance portfolio weights from the estimator's co-variance matrix
    def generateMinVarPortfolio(self):
        cov = self.estimator.cov
        inv_cov = np.linalg.inv(cov)
        ones = np.ones(cov.shape[1])
        numerator = np.matmul(ones.T, inv_cov)
        denominator = np.matmul(ones, (np.matmul(inv_cov, ones)))

        #weight for each asset in the minimum variance portfolio
        self.minVarWeights = numerator/denominator

    #Calculate total portfolio value for the out-of-sample period
    def calculatePortfolioValue(self):
        data = self.estimator.rawData
        mask = (data['Date'] > self.estimator.dateRange.outStartDate) & (data['Date'] <= self.estimator.dateRange.outEndDate)
        outSampleData = data.loc[mask]
        #drop the date column to perform matrix multiplication
        outSampleData = outSampleData.drop('Date', axis=1)
        assetReturns = (outSampleData.sum(axis=0)) * self.minVarWeights
        self.portfolioValue = assetReturns.sum()
        return self.portfolioValue

    #Measure standard deviation of each investment
    def calculateAssetRisk(self):
        #the diagonal of the co-variance matrix is the variance of each asset
        diag = np.diagonal(self.estimator.cov)
        self.assetRisks = np.sqrt(diag)
        return self.assetRisks

    #Measure standard deviation of portfolio
    def calculatePortfolioRisk(self):
        #variance of portfolio = wT * cov_matrix * w
        portfolioVariance = (np.dot(self.minVarWeights.T, np.dot(self.estimator.cov, self.minVarWeights)))
        self.portfolioRisk = np.sqrt(portfolioVariance)
        return self.portfolioRisk   #std dev of portfolio as proxy for volatility/risk

'''
    Class Benchmark
    - Compares performance of portfolios generated by different estimators
    - Takes in the data needed to build a set estimators with same input data, date ranges and data frequency
      and then compares the performance of the minimum variance portfolios generated by them.
    - The main driver class that creates Estimator and Portfolio objects and executes their functions
'''
class Benchmark:
    def __init__(self, filename, estimatorFns, dateRange, frequency):
        self.filename = filename
        self.estimatorFns = estimatorFns
        self.estimators = []
        self.portfolios = []
        self.dateRange = dateRange
        self.frequency = frequency

        self.totalValues = {}
        self.portfolioRisks = {}
        self.assetRisks = {}

    #Create the Estimator objects
    def generateEstimators(self):
        for fn in self.estimatorFns:
            e = Estimator(filename= self.filename, dateRange = self.dateRange, estimationFn=fn, frequency=self.frequency)
            e.estimate()
            self.estimators.append(e)

    #Create Minimum Variance Portfolios from the Estimator objects
    def generatePortfolios(self):
        for e in self.estimators:
            portfolio = Portfolio(e)

            portfolio.generateMinVarPortfolio()
            portfolio.calculatePortfolioValue()
            portfolio.calculatePortfolioRisk()
            portfolio.calculateAssetRisk()

            self.portfolios.append(portfolio)

    #Calculate each portfolio's key values to be compared
    def runBenchmark(self):
        for portfolio in self.portfolios:
            self.totalValues[portfolio.estimator.estimationFn] = portfolio.portfolioValue
            self.portfolioRisks[portfolio.estimator.estimationFn] = portfolio.portfolioRisk
            self.assetRisks[portfolio.estimator.estimationFn] = portfolio.assetRisks
        

# Sample test function #1
def test1():

    in_start_date = pd.to_datetime('1/3/2007')
    in_end_date = pd.to_datetime('12/31/2015')
    out_start_date = pd.to_datetime('12/31/2015')
    out_end_date = pd.to_datetime('12/30/2016')
    dates = DateRange(in_start_date, in_end_date, out_start_date, out_end_date)

    estimatorFnList = ['sample', 'ledoit-wolf', 'aggregate']
    benchmark1 = Benchmark(filename="Returns.csv", estimatorFns=estimatorFnList, dateRange=dates, frequency='M')

    benchmark1.generateEstimators()
    benchmark1.generatePortfolios()
    benchmark1.runBenchmark()

    print("Total value of portfolio for each estimator:")
    print(benchmark1.totalValues)
    print("Portfolio Volatility for each estimator:")
    print(benchmark1.portfolioRisks)

# Sample test function #2
# NOTE: Due to time-constraints, this function isn't as flexible as I would like. 
def test2():

    totalReturns = defaultdict(list)
    portfolioRisks = defaultdict(list)
    benchmarks = []
    numiter = 3 #hardcoded

    #first iteration
    inStart1 = pd.to_datetime('1/3/2007')
    inEnd1 = pd.to_datetime('12/31/2013')
    outStart1 = pd.to_datetime('12/31/2013')
    outEnd1 = pd.to_datetime('12/31/2014')

    #second iteration
    inStart2 = pd.to_datetime('1/2/2008')
    inEnd2 = pd.to_datetime('12/31/2014')
    outStart2 = pd.to_datetime('12/31/2014')
    outEnd2 = pd.to_datetime('12/31/2015')

    #third iteration
    inStart3 = pd.to_datetime('1/2/2009')
    inEnd3 = pd.to_datetime('12/31/2015')
    outStart3 = pd.to_datetime('12/31/2015')
    outEnd3 = pd.to_datetime('12/30/2016')

    #Set date ranges for each iteration
    dateRange1 = DateRange(inStart1, inEnd1, outStart1, outEnd1)
    dateRange2 = DateRange(inStart2, inEnd2, outStart2, outEnd2)
    dateRange3 = DateRange(inStart3, inEnd3, outStart3, outEnd3)

    estimatorFnList = ['sample', 'ledoit-wolf', 'aggregate']

    #Create benchmarks for each iteration
    benchmark1 = Benchmark(filename="Returns.csv", estimatorFns=estimatorFnList, dateRange=dateRange1, frequency='D')
    benchmark1.generateEstimators()
    benchmark1.generatePortfolios()
    benchmark1.runBenchmark()
    benchmark2 = Benchmark(filename="Returns.csv", estimatorFns=estimatorFnList, dateRange=dateRange2, frequency='D')
    benchmark2.generateEstimators()
    benchmark2.generatePortfolios()
    benchmark2.runBenchmark()
    benchmark3 = Benchmark(filename="Returns.csv", estimatorFns=estimatorFnList, dateRange=dateRange3, frequency='D')
    benchmark3.generateEstimators()
    benchmark3.generatePortfolios()
    benchmark3.runBenchmark()
    benchmarks.append(benchmark1)
    benchmarks.append(benchmark2)
    benchmarks.append(benchmark3)

    for benchmark in benchmarks:
        for fn in estimatorFnList:
            totalReturns[fn].append(benchmark.totalValues[fn])
            portfolioRisks[fn].append(benchmark.portfolioRisks[fn])
    

    #Plot Total Returns over iterations
    x = list(range(1,numiter+1))
    y1 = sorted(totalReturns['sample'])
    y2 = sorted(totalReturns['ledoit-wolf'])
    y3 = sorted(totalReturns['aggregate'])
    plt.title("Total Returns of Portfolio over iterations")
    plt.scatter(x, y1, label = "sample")
    plt.scatter(x, y2, label = "ledoit wolf")
    plt.scatter(x, y3, label = "aggregate")
    plt.legend()
    plt.ylabel("Returns")
    plt.xlabel('Iterations')
    plt.show()

    #Plot Portfolio Risk over iterations
    y1 = sorted(np.array(portfolioRisks['sample'])*100)
    y2 = sorted(np.array(portfolioRisks['ledoit-wolf'])*100) 
    y3 = sorted(np.array(portfolioRisks['aggregate'])*100) 
    plt.title("Portfolio Risk over iterations")
    plt.scatter(x, y1, label = "sample")
    plt.scatter(x, y2, label = "ledoit wolf")
    plt.scatter(x, y3, label = "aggregate")
    plt.legend()
    plt.ylabel('Std Dev x 100')
    plt.xlabel('Iterations')
    plt.show()


#Main tester program
def main():

    print("Running test #1")
    test1()
    print("Running test #2")
    test2()
    

main()