import pandas as pd
from apyori import apriori # apriori algorithm

# read dataset
dataset = pd.read_csv('G:\Dataset\Groceries_dataset.csv', header = None)
print(dataset.head)

transactions = []

for i in range(0,38766):
    count = dataset.values[i,2].count('/')  # Counting the '/' sign in the datset
    # print(count)

    if count == 1 :   # if there are only one '/' character, then there are 2 data
        data1, data2 = dataset.values[i,2].split('/')  # spliting the data when there are '/' character
        transactions.append([str(data1), str(data2)]) # input data as list into transactions list

    elif count == 2:  # if there are only two '/' character, then there are 3 data
        data1, data2,data3 = dataset.values[i,2].split('/')
        transactions.append([str(data1), str(data2) , str(data3)])
        # transactions.append([str(data1)])
        # transactions.append([str(data2)])
        # transactions.append([str(data3)])
    elif count == 3: 
        data1, data2,data3, data4 = dataset.values[i,2].split('/')
        transactions.append([str(data1) , str(data2) , str(data3) , str(data4)])
        
    else:
        transactions.append([str(dataset.values[i,2])])
# print(transactions)

# apriori function is used 
rules = apriori(transactions, min_support=.04, min_confidence=.02, min_lift=2)
# print(rules)

# Converting rules into list
results = list(rules)
print(results)
