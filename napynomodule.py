import pandas as pd
import numpy asnp
from typing import List


def df_unique(df_list: List[pd.DataFrame], columns=None):
    if columns is not None:
        df_list = list(map(lambda x: x.loc[:, columns], df_list))

    lendf = [0] + list(map(lambda x: x.shape[0], df_list))
    cumlendf = np.cumsum(lendf)
    newdf = pd.concat(df_list, axis=0)
    add = newdf.apply(lambda x: '-'.join(map(str, x)), axis=1)
    uni, ia, ic = np.unique(add, return_index=True, return_inverse=True)
    iclist = [[ic[j] for j in range(cumlendf[i], cumlendf[i + 1])] for i in range(len(cumlendf) - 1)]
    return uni, newdf.iloc[ia, :], iclist

