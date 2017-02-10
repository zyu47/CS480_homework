# Delete all variables defined so far (in notebook)
for name in dir():
    if not callable(globals()[name]) and not name.startswith('_'):
        del globals()[name]

import numpy as np
import os

## Comment THIS out FOR students' version
# import A2mysolution as mine
# import imp
# imp.reload(mine)
# train = mine.train
# use = mine.use
# rmse = mine.rmse
# partitionKFolds = mine.partitionKFolds
# multipleLambdas = mine.multipleLambdas

def within(correct, attempt, diff):
    return np.abs(correct-attempt)  < diff

g = 0

for func in ['train','use','rmse','partitionKFolds', 'multipleLambdas']:
    if func not in dir() or not callable(globals()[func]):
        print('CRITICAL ERROR: Function named \'{}\' is not defined.'.format(func))

print(''' Testing:
X = np.arange(20).reshape((-1,1))
T = np.abs(X -10) + X
for Xtrain,Ttrain,Xval,Tval,Xtest,Ttest,_ in partitionKFolds(X,T,5,shuffle=False,nPartitions=3)''')

X = np.arange(20).reshape((-1,1))
T = np.abs(X -10) + X

count = 0
for Xtrain,Ttrain,Xval,Tval,Xtest,Ttest,_ in partitionKFolds(X,T,5,shuffle=False,nPartitions=3):
    count += 1
    parts = (Xtrain,Ttrain,Xval,Tval,Xtest,Ttest)

if count != 20:
    print(' 0/20 points. partitionKFolds produced {:d} partitions, but it should have produced 20 partitions.'.format(count))
else:
    g += 20
    print('20/20 points. partitionKFolds produced 20 partitions. Correct.')

if parts[0].shape[0] != 12:
    print(' 0/10 points. Final training set contains {:d} samples, but it should contain 12 samples.'.format(parts[0].shape[0]))
else:
    g += 10
    print('10/10 points. Final training set contains 12 samples. Correct.')
    
if parts[2].shape[0] != 4:
    print(' 0/10 points. Final validation set contains {:d} samples, but it should contain 4 samples.'.format(parts[2].shape[0]))
else:
    g += 10
    print('10/10 points. Final validation set contains 4 samples. Correct.')
    
    

print('''Testing:
X = np.linspace(0,100,1000).reshape((-1,1))
T = X * 0.1
results = multipleLambdas(X,T,4,range(0,10))''')


X = np.linspace(0,100,1000).reshape((-1,1))
T = X * 0.1

results = multipleLambdas(X,T,4,range(0,10))


if np.all(results[:,1] == 0):
    g += 20
    print('20/20 points. All best lambdas are 0.  Correct.')
else:
    print(' 0/20 points. Not all best lambdas are 0.  Incorrect.')
    

if within(1.e-10, np.mean(results[:,2:]), 1.e-3):
    g += 20
    print('20/20 points. Mean of all train, validation and test errors for best lambda are correct.')
else:
    print(' 0/20 points. Mean of all train, validation and test errors for best lambda are incorrect.')

    

print('\n{} Grade is {}/100'.format(os.getcwd().split('/')[-1], g))
print('Up to 20 more points will be given based on the qualty of your descriptions of the method and the results.')
