import numpy as np

def intersection_histogram(dist1, dist2, method, binning=None):
    """
    Give informations about the intersection of two histograms of two lists of values.
    Give the length of the intersection interval (X axis of the histograms), the total occurence in the intersection
    histogram (Y axis of the histograms) and give the relative proportion of values in dist1 and dist2
    in the intersection interval.
    :param dist1: first list of values
    :param dist2: second list of value
    :param method: name of the method you want apply to compute intersection of histogram.
            'minmax' : interval = [MAX(min(dist1), min(dist2)) , MIN(max(dist1), max(dist2))]
            'minmax_strict_positif' : interval = Union_i{bin_i | Card(dist1_bin_i) not O and Card(dist_2_bin_i) not O}
    :param binning: list of values that define bin for dist1 and dist2. only needed for 'minmax_strict_positif' method
    :return: length_normalised_interval : length of the intersection interval defined by the method, normalised by the
    range [ MIN(min(dist1),min(dist2)) , MAX(max(dist1), max(dist2)) ]
    :return: normalised_occ_dist1 : (occ for occurence) portion of values of dist1 in the intersection interval relativ
    to the total number of values Card(dist1) + Card(dist2)
    :return: normalised_occ_dist2 : portion of values of dist2 in the intersection interval relativ
    to the total number of values Card(dist1) + Card(dist2)
    :return: normalised_occ_total: normalised_occurence_dist1 + normalised_occurence_dist2
    """
    nb_dist1 = len(dist1)
    nb_dist2 = len(dist2)
    nb_total = nb_dist1 + nb_dist2
    range_total = np.max(np.max(dist1), np.max(dist2)) - np.min(np.min(dist1), np.min(dist2))
    if method == "minmax":
        min_interval = np.max(np.min(dist1), np.min(dist2))
        max_interval = np.min(np.max(dist1), np.max(dist2))
        length_intersection_interval = (max_interval - min_interval) / range_total
        dist1_mask = (dist1 >= min_interval) & (dist1 <= max_interval)
        occ_dist1 = np.sum(dist1_mask.astype(int))
        dist2_mask = (dist2 >= min_interval) & (dist2 <= max_interval)
        occ_dist2 = np.sum(dist2_mask.astype(int))
        normalised_occ_dist1 = occ_dist1/nb_total
        normalised_occ_dist2 = occ_dist2/nb_total
        normalised_occ_total = normalised_occ_dist1 + normalised_occ_dist2
    if method == "minmax_strict_positif":
        hist_dist1 = np.histogram(dist1, bins=binning, density=False)
        hist_dist2 = np.histogram(dist2, bins=binning, density=False)
    else :
        print("incorect method. try 'minmax' or ' minmax_strict_positif'")
    return length_intersection_interval, normalised_occ_dist1, normalised_occ_dist2, normalised_occ_total



def difference_medianes_distribution1_distribution2(distribution1, distribution2):
    """
    :param distribution1:
    :param distribution2:
    :return:
    """
    median_1 = np.median(distribution1)
    median_2 = np.median(distribution2)
    diff_median = np.abs(median_2 - median_1)
    return diff_median

def normalised_spread_distribution(distribution):
    """
    :param distribution:
    :return:
    """
    var_dist = 0
    range_dist = 0
    percentil_5_95 = 0
    return var_dist, range_dist, percentil_5_95

def angle_vec1_vec2(vec1, vec2):
    """
    :param vec1:
    :param vec2:
    :return:
    """
    angle = 0
    return angle