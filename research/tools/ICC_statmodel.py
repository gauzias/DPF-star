import statsmodels.api as sm
from statsmodels.formula.api import ols
import numpy as np
import pandas as pd
from scipy.stats import f as fisher_f


def compute_icc_with_statsmodel(data, subject, rater, value, bICC_1=True, bICC_A1=True, bICC_C1=True, bICC_k=True, bICC_Ak=True,
                                bICC_Ck=True):
    """
    :param data: a dataframe with 3 columns : Subject - Rater - Value
    :return: ICC_score with confidence interval and the 2way anova table
    """
    data = data.rename(columns={subject : 'Subject', rater: 'Rater',value : 'Value' })

    # ANOVA stats model
    # compute_icc STATSMODEL : calcul of ANOVA
    data_lm = ols('Value ~ C(Rater) + C(Subject)', data=data).fit()
    table = sm.stats.anova_lm(data_lm, typa=2)  # Type 2 ANOVA DataFrame
    # print(table)

    # compute_icc STATSMODEL : calcul of compute_icc
    K = len(np.unique(data['Rater']))
    # print('nombre de rater : ', K)
    nobs = len(data)
    # print('nombre observations', nobs)
    N = len(np.unique(data['Subject']))
    # print('nombre de sujet : ', J)

    dfr = table.loc['C(Subject)']['df']
    dfc = table.loc['C(Rater)']['df']
    dfe = table.loc['Residual']['df']

    MSBS = table.loc['C(Subject)']['mean_sq']
    MSBM = table.loc['C(Rater)']['mean_sq']
    MSE = table.loc['Residual']['mean_sq']
    MSWS = (table.loc['C(Rater)']['sum_sq'] + table.loc['Residual']['sum_sq']) / (dfc + dfe)

    ICC_A1 = np.NAN
    ICC_C1 = np.NAN
    ICC_1 = np.NAN
    ICC_k = np.NAN
    ICC_Ak = np.NAN
    ICC_Ck = np.NAN
    low_ICC_1 = np.NAN
    low_ICC_A1 = np.NAN
    low_ICC_C1 = np.NAN
    low_ICC_k = np.NAN
    low_ICC_Ak = np.NAN
    low_ICC_Ck = np.NAN
    high_ICC_1 = np.NAN
    high_ICC_A1 = np.NAN
    high_ICC_C1 = np.NAN
    high_ICC_k = np.NAN
    high_ICC_Ak = np.NAN
    high_ICC_Ck = np.NAN

    if bICC_1 :
        ICC_1 = (MSBS - MSWS) / (MSBS + (K - 1) * MSWS)
        FU_1 = (MSBS / MSWS) * fisher_f.ppf(0.95, N * (K - 1), N - 1)
        FL_1 = (MSBS / MSWS) / fisher_f.ppf(0.95, (N - 1), N * (K - 1))
        low_ICC_1 = (FL_1 - 1) / (FL_1 + (K - 1))
        high_ICC_1 = (FU_1 - 1) / (FU_1 + (K - 1))


    if bICC_A1 :
        ICC_A1 = (MSBS - MSE) / (MSBS + (K - 1) * MSE + K / N * (MSBM - MSE))
        nu_num = (K - 1) * (N - 1) * (K * ICC_A1 * (MSBM / MSE) + N * (1 + (K - 1) * ICC_A1) - K * ICC_A1) ** 2
        nu_den = (N - 1) * ((K * ICC_A1 * (MSBM / MSE)) ** 2) + (N * (1 + (K - 1) * ICC_A1) - K * ICC_A1) ** 2
        nu = nu_num / nu_den
        FU_A1 = fisher_f.ppf(0.95, N - 1, nu)
        FL_A1 = fisher_f.ppf(0.95, nu, N - 1)
        low_ICC_A1 = (N * (MSBS - FU_A1 * MSE)) / (FU_A1 * (K * MSBM + (K * N - K - N) * MSE) + N * MSBS)
        high_ICC_A1 = (N * (FL_A1 * MSBS - MSE)) / (K * MSBM + (K * N - K - N) * MSE + N * FL_A1 * MSBS)



    if bICC_C1 :
        ICC_C1 = (MSBS - MSE) / (MSBS + (K - 1) * MSE)
        FL_C1 = (MSBS / MSE) / fisher_f.ppf(0.95, N - 1, (N - 1) * (K - 1))
        FU_C1 = (MSBS / MSE) * fisher_f.ppf(0.95, (N - 1) * (K - 1), N - 1, )
        low_ICC_C1 = (FL_C1 - 1) / (FL_C1 + (K - 1))
        high_ICC_C1 = (FU_C1 - 1) / (FU_C1 + (K - 1))


    if bICC_k:
        ICC_k = (MSBS - MSWS) / MSBS
        FU_1 = (MSBS / MSWS) * fisher_f.ppf(0.95, N * (K - 1), N - 1)
        FL_1 = (MSBS / MSWS) / fisher_f.ppf(0.95, (N - 1), N * (K - 1))
        low_ICC_k = 1 - 1 / FL_1
        high_ICC_k = 1 - 1 / FU_1

    if bICC_Ak :
        ICC_Ak = (MSBS - MSE) / (MSBS + ((MSBM - MSE) / N))
        low_ICC_Ak = (K * low_ICC_A1) / (1 + (K - 1) * low_ICC_A1)
        high_ICC_Ak = (K * high_ICC_A1) / (1 + (K - 1) * high_ICC_A1)

    if bICC_Ck :
        ICC_Ck = (MSBS - MSE) / MSBS
        FL_C1 = (MSBS / MSE) / fisher_f.ppf(0.95, N - 1, (N - 1) * (K - 1))
        FU_C1 = (MSBS / MSE) * fisher_f.ppf(0.95, (N - 1) * (K - 1), N - 1, )
        low_ICC_Ck = 1 - 1 / FL_C1
        high_ICC_Ck = 1 - 1 / FU_C1

    df_ICC = pd.DataFrame(dict(ICC_type=['ICC_1', 'ICC_A1', 'ICC_C1', 'ICC_k', 'ICC_Ak', 'ICC_Ck'],
                               ICC_score=[ICC_1, ICC_A1, ICC_C1, ICC_k, ICC_Ak, ICC_Ck],
                               low_ICC=[low_ICC_1, low_ICC_A1, low_ICC_C1, low_ICC_k, low_ICC_Ak, low_ICC_Ck],
                               high_ICC=[high_ICC_1, high_ICC_A1, high_ICC_C1, high_ICC_k, high_ICC_Ak, high_ICC_Ck]))
    df_ICC = df_ICC.set_index(['ICC_type'])
    ICC_data = {'ICC': df_ICC,
                'anova_table': table}

    return ICC_data

if __name__ == '__main__':
    data = pd.DataFrame(dict(Subject = ['S1','S2','S3','S1','S2','S3'],
                             Rater = ['T1','T2','T1','T2','T1','T2'],
                             Value = [12, 13, 14, 12.3, 13.5, 14.8]))
    ICC_data = compute_icc_with_statsmodel(data)
    print(ICC_data['ICC'])
