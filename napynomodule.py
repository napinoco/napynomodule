import pandas as pd
import numpy as np
from typing import List
import os
import sys
import chardet
import scipy.sparse as sp
import scipy.sparse.csgraph as csg


importer_path = ''


def set_importer_path(path: str):
    global importer_path
    importer_path = path


def df_filter(df: pd.DataFrame, columns: List, values: List[List]):
    m = len(columns)
    m2 = len(values)
    if m != m2:
        raise Exception('Lengths must agree.')

    flag = pd.Series([True for i in range(df.shape[0])])
    for i in range(m):
        flag &= df[columns[i]].map(lambda x: x in values[i])

    return df.loc[flag, :]


def df_unique(df_list: List[pd.DataFrame], columns=None):
    if columns is not None:
        df_list = list(map(lambda x: x.loc[:, columns], df_list))

    lendf = [0] + list(map(lambda x: x.shape[0], df_list))
    cumlendf = np.cumsum(lendf)
    newdf = pd.concat(df_list, axis=0)
    add = newdf.apply(lambda x: '-'.join(map(str, x)), axis=1)
    uni, ia, ic = np.unique(add, return_index=True, return_inverse=True)
    iclist = [[ic[j] for j in range(cumlendf[i], cumlendf[i + 1])] for i in range(len(cumlendf) - 1)]
    retdf = newdf.iloc[ia, :]
    retdf.assign(id=uni)
    return uni, retdf, ia, iclist


def split_df_col(df: pd.DataFrame, delim: str, split_col: str, new_cols: List[str]=None):
    df2 = df[split_col].str.split(delim, expand=True)
    if new_cols is not None:
        df2.columns = new_cols
    return pd.concat([df, df2], axis=1)


def get_cwd():
    if getattr(sys, 'frozen', False):
        # we are running in a bundle
        cwd = os.path.dirname(sys.executable)
    else:
        # we are running in a normal Python environment
        # try:
        #     cwd = os.path.dirname(os.path.abspath(__file__))
        # except NameError:
        #     cwd = os.getcwd()
        cwd = importer_path
    return cwd


def get_abs_path(file_name: str):
    return os.path.join(get_cwd(), file_name)


def check_encoding(file_path: str):
    with open(file_path, mode='rb') as f:
        return chardet.detect(f.readline())['encoding']


def block_diagonalize_permutation(A: sp.spmatrix):
    A = A.tocsc()
    ATA = A.transpose().dot(A)
    n, cmp = csg.connected_components(ATA)
    col_list = [list(np.where(cmp == j)[0]) for j in range(n)]
    row_list = [list(np.unique(A[:, col_list[j]].nonzero()[0])) for j in range(n)]
    return row_list, col_list

