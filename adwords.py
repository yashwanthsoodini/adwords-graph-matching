import pandas as pd
import numpy as np
import sys

def greedy(bids, queries):
    revenue = 0
    for query in queries:
        match_bids = bids.loc[(bids['Keyword']==query)&(bids['Budget']>=bids['Bid Value'])]
        if match_bids.empty:
            continue
        ix = match_bids['Bid Value'].argmax()
        bid_val = match_bids['Bid Value'].iloc[ix]
        adv = match_bids['Advertiser'].iloc[ix]
        bids.loc[bids['Advertiser']==adv,'Budget'] -= bid_val
        revenue += bid_val
    return revenue

def msvv(bids, queries):
    def psi(x):
        return 1-np.exp(x-1)
    
    def best_adv(bids):
        bids = bids.assign(Fraction_Spent = (bids['Initial_Budget']-bids['Budget'])/bids['Initial_Budget'])
        bids = bids.assign(Psi = psi(bids['Fraction_Spent']))
        bids = bids.assign(Prod = bids['Bid Value']*bids['Psi'])
        return bids.Advertiser.iloc[bids.Prod.argmax()]
    
    revenue = 0
    bids = bids.assign(Initial_Budget=bids['Budget'])
    for query in queries:
        match_bids = bids[(bids['Keyword']==query)&(bids['Budget']>=bids['Bid Value'])]
        if match_bids.empty:
            continue
        adv = best_adv(match_bids)
        bid_val = float(match_bids.loc[match_bids['Advertiser']==adv,'Bid Value'])
        bids.loc[bids['Advertiser']==adv,'Budget'] -= bid_val
        revenue += bid_val
    return revenue

def balance(bids, queries):
    revenue = 0
    bids = bids.assign(Initial_Budget=bids['Budget'])
    for query in queries:
        match_bids = bids[(bids['Keyword']==query)&(bids['Budget']>=bids['Bid Value'])]
        if match_bids.empty:
            continue
        adv = match_bids.Advertiser.iloc[match_bids.Budget.argmax()]
        bid_val = float(match_bids.loc[match_bids['Advertiser']==adv,'Bid Value'])
        bids.loc[bids['Advertiser']==adv,'Budget'] -= bid_val
        revenue += bid_val
    return revenue

if __name__ == "__main__":
    bids = pd.read_csv("bidder_dataset.csv")
    bids = bids.fillna(method='ffill')
    sum_budgets = bids.Budget.sum()
    queries = np.loadtxt("queries.txt",dtype="str", delimiter="\n")
    np.random.seed(0)
    if sys.argv[1]=='greedy':
        algo = greedy
    elif sys.argv[1]=='mssv':
        algo = mssv
    elif sys.argv[1]=='balanced':
        algo = balanced
    else:
        print("Invalid Algorithm!")
    print(algo(bids,queries))
    for i in range(100):
        bids = pd.read_csv("bidder_dataset.csv")
        bids = bids.fillna(method='ffill')
        np.random.shuffle(queries)
        rev = algo(bids,queries)
        comp_ratio = rev/sum_budgets
        print(comp_ratio)