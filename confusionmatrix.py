import pickle
import numpy as np
import pandas as pd

unified_data = pickle.load(open('unifieddata.p','rb'))

ypreds = []
for i, each in enumerate(unified_data['filename']):
    if unified_data.at[i, 'truenpreds'] > unified_data.at[i, 'npreds']:
        if unified_data.at[i, 'npreds'] != 0:
            for j in range(unified_data.at[i, 'npreds']):
                ypreds.append(unified_data.at[i, 'class'][j])
        for j in range(unified_data.at[i, 'truenpreds']-unified_data.at[i, 'npreds']):
            ypreds.append('None')
    else:
        for j in range(unified_data.at[i, 'truenpreds']):
            ypreds.append(unified_data.at[i, 'class'][j])

ytrue = []
for i, each in enumerate(unified_data['trueclass']):
    if len(each) > 1:
        for j in range(len(each)):
            ytrue.append(each[j])
    else:
        ytrue.append(each[0])

ytrue = pd.Series(ytrue, name = 'Actual')
ypreds = pd.Series(ypreds, name = 'Predicted')
df_confusion = pd.crosstab(ytrue, ypreds, rownames=['Actual'], colnames=['Predicted'], margins=True)

pickle.dump(df_confusion, open('df_confusion.p', 'wb'))
#inspect on a jupyter notebook
