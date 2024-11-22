import pandas as pd
from compute_icc.compute_icc_with_statsmodel import compute_icc_with_statsmodel
import numpy as np


def compute_ICC_map(subjects, data_test, data_retest, bICC_1 = True, bICC_A1 = False, bICC_C1=False, bICC_k = False, bICC_Ak = False, bICC_Ck = False):
    """
    :param subjects:['sujet1', 'sujet2', ..., 'sujetN']
    :param data_test: [[list measure sujet 1],[list measure sujet 2],...,[list measure sujet N]]
    :param data_retest: [[list measure sujet 1],[list measure sujet 2],...,[list measure sujet N]]
    :param ICC_1 : ICC random one way anova based. Set the boolean to compute it or not
    :param ICC_1 : ICC random or mixed model two way anova based, consistency between raters. Set the boolean to compute it or not
    :param ICC_1 : ICC random or mixed model two way anova based, absolute agreement beween raters. Set the boolean to compute it or not

    :return:
    """
    #load retest and set the appropriate format
    df_retest = pd.DataFrame(data=data_retest, index=subjects)
    nb_vertex = len(df_retest.columns)
    #print('the number of vertex is : ', nb_vertex)
    vertexs = ['v_' + str(i) for i in range(nb_vertex)]
    df_retest.columns = vertexs
    #load test and set the appropriate format
    df_test = pd.DataFrame(data=data_test, index=subjects)
    df_test.columns = vertexs
    # here we get for df_test and df_retest:
    #       |v_0|v_1|...|v_N
    # suj1  | . | . | . | .
    # sujN  | . | . | . | .

    data = pd.concat([df_test.assign(Rater='MR1'), df_retest.assign(Rater='MR2')],axis=0)
    #here we get for data:
    #       |v_0|v_1|...|v_N|Rater
    # suj1  | . | . | . | . | MR1
    # sujN  | . | . | . | . | MR1
    # suj1  | . | . | . | . | MR2
    # sujN  | . | . | . | . | MR2
    data.reset_index(inplace=True)
    data = data.rename(columns={'index': 'Subject'})
    #here we get for data (we add  'subject' as column name):
    # Subject|v_0|v_1|...|v_N|Rater
    # suj1   | . | . | . | . | MR1
    # sujN   | . | . | . | . | MR1
    # suj1   | . | . | . | . | MR2
    # sujN   | . | . | . | . | MR2
    #print(data)

    ICC_1_map = [np.NAN]*nb_vertex
    ICC_A1_map = [np.NAN]*nb_vertex
    ICC_C1_map = [np.NAN]*nb_vertex
    low_ICC_1_map = [np.NAN]*nb_vertex
    low_ICC_A1_map = [np.NAN]*nb_vertex
    low_ICC_C1_map = [np.NAN]*nb_vertex
    high_ICC_1_map = [np.NAN]*nb_vertex
    high_ICC_A1_map = [np.NAN]*nb_vertex
    high_ICC_C1_map = [np.NAN]*nb_vertex


    for idx_vert, vert in enumerate(vertexs):
        print('vertex : ', idx_vert , '/', nb_vertex)
        # we extract only the column with the vertex of interest
        datav = data[['Subject', vert, 'Rater']]
        # we rename the column according the needed inputs of the compute_icc_with_stats_model function
        datav = datav.rename(columns={vert: 'Value'})
        ICC_data  = compute_icc_with_statsmodel(datav, bICC_1 = bICC_1, bICC_A1=bICC_A1, bICC_C1=bICC_C1)


        if bICC_1:
            ICC_1_map[idx_vert] = ICC_data['ICC'].loc['ICC_1']['ICC_score']
            low_ICC_1_map[idx_vert] = ICC_data['ICC'].loc['ICC_1']['low_ICC']
            high_ICC_1_map[idx_vert] = ICC_data['ICC'].loc['ICC_1']['high_ICC']


        if bICC_A1:
            ICC_A1_map[idx_vert] = ICC_data['ICC'].loc['ICC_A1']['ICC_score']
            low_ICC_A1_map[idx_vert] = ICC_data['ICC'].loc['ICC_A1']['low_ICC']
            high_ICC_A1_map[idx_vert] = ICC_data['ICC'].loc['ICC_A1']['high_ICC']


        if bICC_C1:
            ICC_C1_map[idx_vert] = ICC_data['ICC'].loc['ICC_C1']['ICC_score']
            low_ICC_C1_map[idx_vert] = ICC_data['ICC'].loc['ICC_C1']['low_ICC']
            high_ICC_C1_map[idx_vert] = ICC_data['ICC'].loc['ICC_C1']['high_ICC']


    dict_ICC = { 'ICC_1' : ICC_1_map,
             'ICC_A1': ICC_A1_map,
             'ICC_C1': ICC_C1_map}

    dict_low_ICC = {'low_ICC_1': low_ICC_1_map,
                'low_ICC_A1': low_ICC_A1_map,
                'low_ICC_C1': low_ICC_C1_map}

    dict_high_ICC = {'high_ICC_1': high_ICC_1_map,
                'high_ICC_A1': high_ICC_A1_map,
                'high_ICC_C1': high_ICC_C1_map}

    ICC_map = pd.DataFrame(dict_ICC)
    low_ICC_map = pd.DataFrame(dict_low_ICC)
    high_ICC_map = pd.DataFrame(dict_high_ICC)
    return ICC_map, low_ICC_map, high_ICC_map


if __name__ == '__main__':
    subjects = ['S1', 'S2', 'S3']
    T1 = [[1,2,3,4],[4,5,6,7],[20,50,100,200]]
    T2 = [[2, 2, 3, 4.5], [5, 5, 6.2, 7.1], [28, 42, 98, 200.5]]
    ICC_map, low_ICC_map, high_ICC_map = compute_ICC_map(subjects, T1, T2, bICC_A1=True)

    print(ICC_map)
    print(low_ICC_map)
    print(high_ICC_map)