###################################################################################################
#
# signatures.py
#
# a python script for delineating petroleum hydrocarbon data collected from impacted soil
#
###################################################################################################

from numpy import *
from pandas import *
from datetime import *
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn import svm
from sklearn.metrics.pairwise import euclidean_distances

def ReadSoilData(bins, min_filter):
    # read site soil data and return normalized fractions
    measured_df = read_csv('soil_samples.txt', sep='\t')
    mask = measured_df['Analyte'].isin(bins)                                # only include analytes listed under 'bins' array (e.g., TPH)
    soil_df = measured_df[mask]
    soil_df.reset_index(inplace=True)
    soil_df['Date'] = to_datetime(soil_df['Date'])
    nd_match = array(soil_df['Result'][0]=='<')                             # address non-detects by assuming = 1/2 stated detection limit
    soil_df['Result'] = [s.strip('<') for s in soil_df['Result']]
    soil_df['Result'] = soil_df['Result'].astype(float)
    soil_df['Conc'] = nd_match*0.5*soil_df['Result'] + (1-nd_match)*soil_df['Result']
    pivot_df = pivot_table(soil_df, values='Conc', index=['Location ID', 'Depth', 'Date'], columns=['Analyte'])
    pivot_df = pivot_df[pivot_df.isnull().sum(axis=1)==0]                   # remove samples with null values (i.e., non-reported)
    pivot_df['Sum'] = pivot_df.sum(axis=1)                                  # normalize fraction concentrations
    pivot_df = pivot_df[pivot_df['Sum']>min_filter]
    pivot_df['LogSum'] = log10(pivot_df['Sum'])
    for species in bins: pivot_df[species] = pivot_df[species]/pivot_df['Sum']
    pivot_df.reset_index(inplace=True)
    return pivot_df

def CreateTraining(fuels, tph_bins, N, fuel_refs_df):
    # create training sets associated with different fuels (e.g., gasoline, bunker C)
    for i, fuel_type in enumerate(fuels):
        fuels_avg = array(fuel_refs_df[fuel_refs_df['category']==fuel_type].mean())
        fuels_stdev = array(fuel_refs_df[fuel_refs_df['category']==fuel_type].std())
        syn_matrix = fuels_avg + (fuels_stdev * random.randn(N, len(tph_bins)))     # generate N-sized population, randomly distributed about means
        if i:
            # add to full training set matrix
            training_set = concatenate((training_set, syn_matrix), axis=0)
        else:
            training_set = syn_matrix
    training_set_df = DataFrame(training_set, columns = tph_bins)
    training_set_df[training_set_df < 0] = 0.                                       # remove negative fractions
    training_set_df['Sum'] = training_set_df.sum(axis=1)                            # normalize TPH fraction concentrations
    for tph in tph_bins: training_set_df[tph] = training_set_df[tph]/training_set_df['Sum']
    # add tags to synthetic fuels dataframe
    for i, fuel_type in enumerate(fuels):
        tag = full(N, fuel_type, dtype='|S20')
        if i:
            tags = concatenate((tags, tag))
        else:
            tags = tag
    training_set_df['tag'] = tags
    training_set_df.to_csv('training_set.csv')
    return training_set_df

def DistDistrib(points, report):
    # return point-to-point distance percentiles for all points in points
    d_matrix = euclidean_distances(points, points)
    d1 = triu(d_matrix)
    d2 = reshape(d1, -1)
    d_array = d2[d2 > 0.]
    percents = percentile(d_array, report)
    return percents


### main script ###


def Signatures(): 

    # a few definitions ...
    tph_bins = array(['TPH_C06', 'TPH_C07', 'TPH_C08', 'TPH_C09-C10', 'TPH_C11-C12', 'TPH_C13-C14', 'TPH_C15-C16', 'TPH_C17-C18', 'TPH_C19-C20', 'TPH_C21-C22','TPH_C23-C24', 'TPH_C25-C28', 'TPH_C29-C32', 'TPH_C33-C36'])
    fuels = array(['gasoline', 'diesel', 'kerosene', 'bunker C', 'heavy fuel oil', 'crude oil'])
    stretch = 5.                                                                    # vertical exag. factor, used to calculate distance matrices
    report = linspace(start=10., stop=90., num=9, endpoint=True)                    # percentile classes used to process distance matrices
    N = 100                                                                         # number of samples to include in each synthetic reference fuel population
    num_K_clusters = 6                                                              # number of K-Means clusters to assign
    eps=0.1                                                                         # difference tolerance, dbscan cluster analysis
    min_samples=10                                                                  # minimum number of samples for dbscan clusters 

    # read data sets ...

    soil_TPH_df = ReadSoilData(tph_bins, 0.)                                        # consider all reported samples in data sets, including those with values of 0.
    print 'Read and processed all site soil data.'

    locations_df =  read_csv('survey.txt',sep='\t')
    print 'Read boring locations.'

    soil_TPH_df = merge(locations_df, soil_TPH_df, on='Location ID', how='inner')
    elev = array(soil_TPH_df['Surface Elevation(ft-msl)'] - soil_TPH_df['Depth'])
    soil_TPH_df.insert(5, 'elev', elev)
    print 'Merged soil sample survey data with TPH data set.'

    # conduct cluster analyses to find general patterns in TPH data

    X = soil_TPH_df[tph_bins].values                                                # define feature subset
    k_means = KMeans(init='k-means++', n_clusters=num_K_clusters, n_init=25)        # K-means cluster analysis
    z = k_means.fit_predict(X)
    soil_TPH_df['kmeans_group'] = z                                                 # append group indices to soil_TPH data frame
    centroids_df = DataFrame(k_means.cluster_centers_, columns=tph_bins)            # note cluster centroids and write to output file
    centroids_df.to_csv('centroids.csv')
    z = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)                     # DBSCAN cluster analysis
    soil_TPH_df['dbscan_group'] = z
    num_d_clusters = z.max() + 2
    print 'Conducted cluster analyses.'

    # tag data points using SVM algorithm on TPH data
    
    fuel_refs_df = read_csv('fuel_ref.txt',sep='\t')
    print 'Read fuel reference compositions.'
    training_set_df = CreateTraining(fuels, tph_bins, N, fuel_refs_df)              # generate synthetic reference fuel populations (for training sets)
    X = training_set_df[tph_bins].values                                            # define feature subset of training set
    y = training_set_df['tag'].values                                               # define targets of training set        
    C = 1.0                                                                         # fit model (C = SVM regularization parameter)
    lin_svc = svm.LinearSVC(C=C).fit(X, y)
    Z = soil_TPH_df[tph_bins].values                                                # use model to classify the test set
    z = lin_svc.predict(Z)
    soil_TPH_df['svm_predict'] = z                                                  # append fuel 'tags' to soil_TPH data frame
    print 'Conducted support-vector-machine classification analysis.'

    # write output files (for both clustering and SVM) ...
      
    soil_TPH_df.to_csv('soil_TPH.csv')                                                                                      # write fully processed soil hydrocarbon datasets to output files               
    for fuel_type in fuels: soil_TPH_df[soil_TPH_df['svm_predict'] == fuel_type].to_csv(fuel_type + '.csv')                 # write output files by tagged signature
    for i in xrange(num_K_clusters): soil_TPH_df[soil_TPH_df['kmeans_group'] == i].to_csv('kgroup_' + str(i) + '.csv')      # write output files by K-means group index
    for i in xrange(num_d_clusters): soil_TPH_df[soil_TPH_df['dbscan_group'] == i-1].to_csv('dgroup_' + str(i) + '.csv')    # write output files by K-means group index

    # compare distibution of point-to-point distances, within classes and between classes, as a measured of randomness of scatter

    for i, fuel_type in enumerate(fuels):                                           # distance arrays, by tag (i.e., svm-designation)
        points = soil_TPH_df[soil_TPH_df['svm_predict'] == fuel_type][['Easting', 'Northing', 'elev']]
        points['elev'] *= stretch
        percents = DistDistrib(points, report)
        if i:
            dist_matrix = dstack((dist_matrix, percents))
        else:
            dist_matrix = percents
    tag_df = DataFrame(transpose(dist_matrix[0]))
    tag_df.columns = report.astype(str) 
    tag_df.insert(0, 'category', fuels)
    
    for i in xrange(num_K_clusters):                                                  # distance arrays, by cluster: Kmeans-designation
        points = soil_TPH_df[soil_TPH_df['kmeans_group'] == i][['Easting', 'Northing', 'elev']]
        points['elev'] *= stretch
        percents = DistDistrib(points, report)        
        if i:
            dist_matrix = dstack((dist_matrix, percents))
        else:
            dist_matrix = percents
    kcluster_df = DataFrame(transpose(dist_matrix[0]))
    kcluster_df.columns = report.astype(str)

    for i in xrange(num_d_clusters):                                                  # distance arrays, by cluster: dbscan-designation
        points = soil_TPH_df[soil_TPH_df['dbscan_group'] == i-1][['Easting', 'Northing', 'elev']]
        points['elev'] *= stretch
        percents = DistDistrib(points, report)        
        if i:
            dist_matrix = dstack((dist_matrix, percents))
        else:
            dist_matrix = percents
    dcluster_df = DataFrame(transpose(dist_matrix[0]))
    dcluster_df.columns = report.astype(str)

    # distance array for all soil samples
    points = soil_TPH_df[['Easting', 'Northing', 'elev']]
    points['elev'] *= stretch
    percents = DistDistrib(points, report)
    all_df = DataFrame(percents.reshape((1, -1)), columns = report.astype(str))
    all_df.columns = report.astype(str)

    # summarize distances by cluster
    kcluster_all_df = kcluster_df.append(all_df, ignore_index=True)
    kcluster_all_df.to_csv('report_k_cluster.csv')
    dcluster_all_df = dcluster_df.append(all_df, ignore_index=True)
    dcluster_all_df.to_csv('report_d_cluster.csv')

    # summarize distances by svm tags
    all_df.insert(0, 'category', 'ALL')    
    tag_all_df = tag_df.append(all_df, ignore_index=True)
    tag_all_df.to_csv('report_tag.csv')

    print 'Analyzed sample-to-sample distances among sets.'

    print 'Done.'


### run script ###


Signatures()



