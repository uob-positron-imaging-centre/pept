#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#    pept is a Python library that unifies Positron Emission Particle
#    Tracking (PEPT) research, including tracking, simulation, data analysis
#    and visualisation tools
#
#    Copyright (C) 2019 Andrei Leonard Nicusan
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.


# File   : peptml.py
# License: License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 28.08.2019


'''The *peptml* module implements a hierarchical density-based clustering
algorithm for general Positron Emission Particle Tracking (PEPT)

The module aims to provide general classes which can
then be used in a script file as the user sees fit. For example scripts,
look at the base of the pept library.

The peptml subpackage accepts any instace of the LineData base class
and can create matplotlib- or plotly-based figures.

PEPTanalysis requires the following packages:

* **numpy**
* **joblib** for multithreaded operations (such as midpoints-finding)
* **tqdm** for showing progress bars
* **plotly.subplots** and **plotly.graph_objects** for plotly-based plotting
* **hdbscan** for clustering midpoints and centres
* **time** for verbose timing of operations

It was successfuly used at the University of Birmingham to analyse real
Fluorine-18 tracers in air.

If you use this package, you should cite
the following paper: [TODO: paper signature].

'''


import  time
import  sys

import  numpy                                   as          np
from    scipy.spatial                           import      cKDTree

from    joblib                                  import      Parallel,       delayed
from    tqdm                                    import      tqdm
from    plotly.subplots                         import      make_subplots
import  plotly.graph_objects                    as          go

# Fix a deprecation warning inside the sklearn library
try:
    sys.modules['sklearn.externals.six'] = __import__('six')
    sys.modules['sklearn.externals.joblib'] = __import__('joblib')
    import hdbscan
except ImportError:
    import hdbscan

import  pept
from    .extensions.find_cutpoints_api          import      find_cutpoints_api




class Cutpoints(pept.PointData):

    def __init__(self):

        # Call pept.PointData constructor with dummy data
        super().__init__([[0., 0., 0., 0.]],
                         sample_size = 0,
                         overlap = 0,
                         verbose = False)


    @staticmethod
    def get_cutoffs(sample):

        # Check sample has shape (N, 7)
        if sample.ndim != 2 or sample.shape[1] != 7:
            raise ValueError('\n[ERROR]: sample should have dimensions (N, 7). Received {}\n'.format(sample.shape))

        # Compute cutoffs for cutpoints as the (min, max) values of the lines
        # Minimum value of the two points that define a line
        min_x = min(sample[:, 1].min(),
                    sample[:, 4].min())
        # Maximum value of the two points that define a line
        max_x = max(sample[:, 1].max(),
                    sample[:, 4].max())

        # Minimum value of the two points that define a line
        min_y = min(sample[:, 2].min(),
                    sample[:, 5].min())
        # Maximum value of the two points that define a line
        max_y = max(sample[:, 2].max(),
                    sample[:, 5].max())

        # Minimum value of the two points that define a line
        min_z = min(sample[:, 3].min(),
                    sample[:, 6].min())
        # Maximum value of the two points that define a line
        max_z = max(sample[:, 3].max(),
                    sample[:, 6].max())

        cutoffs = np.array([min_x, max_x, min_y, max_y, min_z, max_z])
        return cutoffs


    @staticmethod
    def find_cutpoints_sample(sample, max_distance, cutoffs = None):

        # Check sample has shape (N, 7)
        if sample.ndim != 2 or sample.shape[1] != 7:
            raise ValueError('\n[ERROR]: sample should have dimensions (N, 7). Received {}\n'.format(sample.shape))

        if cutoffs is None:
            cutoffs = Cutpoints.get_cutoffs(sample)
        else:
            cutoffs = np.asarray(cutoffs, order = 'C', dtype = float)
            if cutoffs.ndim != 1 or len(cutoffs) != 6:
                raise ValueError('\n[ERROR]: cutoffs should be a one-dimensional array with values [min_x, max_x, min_y, max_y, min_z, max_z]\n')

        sample_cutpoints = find_cutpoints_api(sample, max_distance, cutoffs)
        return sample_cutpoints


    def find_cutpoints(self,
                       line_data,
                       max_distance,
                       cutoffs = None,
                       verbose = True):

        if verbose:
            start = time.time()

        # Check line_data is an instance (or a subclass!) of pept.LineData
        if not isinstance(line_data, pept.LineData):
            raise Exception('[ERROR]: line_data should be an instance of pept.LineData')

        # If cutoffs were not supplied, compute them
        if cutoffs is None:
            cutoffs = self.get_cutoffs(line_data.line_data)
        # Otherwise make sure they are a C-contiguous numpy array
        else:
            cutoffs = np.asarray(cutoffs, order = 'C', dtype = float)
            if cutoffs.ndim != 1 or len(cutoffs) != 6:
                raise ValueError('\n[ERROR]: cutoffs should be a one-dimensional array with values [min_x, max_x, min_y, max_y, min_z, max_z]\n')

        # Using joblib, collect the cutpoints from every sample in a list
        # of arrays
        cutpoints = Parallel(n_jobs = -1, prefer = 'threads')(delayed(self.find_cutpoints_sample)(sample, max_distance, cutoffs) for sample in tqdm(line_data))

        # cutpoints shape: (n, m, 4), where n is the number of samples, and
        # m is the number of cutpoints in the sample
        cutpoints = np.array(cutpoints)

        number_of_samples = len(cutpoints)
        cutpoints = np.vstack(np.array(cutpoints))
        number_of_cutpoints = len(cutpoints)

        # Average number of cutpoints per sample
        cutpoints_per_sample = int(number_of_cutpoints / number_of_samples)

        super().__init__(cutpoints,
                         sample_size = cutpoints_per_sample,
                         overlap = 0,
                         verbose = False)

        if verbose:
            end = time.time()
            print("\n\n\nFinding the cutpoints took {} seconds".format(end - start))

        return self




def findMeanError(truePositions, foundPositions):

    tree = cKDTree(truePositions)

    meanError = 0
    meanErrorX = 0
    meanErrorY = 0
    meanErrorZ = 0
    n = 0
    for centre in foundPositions:
        d, index = tree.query(centre, k = 1,  n_jobs = -1)
        meanError += np.linalg.norm(centre - truePositions[index])

        meanErrorX += np.abs(centre[0] - truePositions[index][0])
        meanErrorY += np.abs(centre[1] - truePositions[index][1])
        meanErrorZ += np.abs(centre[2] - truePositions[index][2])

        n += 1

    meanError /= n

    meanErrorX /= n
    meanErrorY /= n
    meanErrorZ /= n

    return [meanError, meanErrorX, meanErrorY, meanErrorZ]




class ClustererBase:
    '''
    Base class that provides common functionality between any clustering algorithms.
    Any clustering algorithm should have at least the following attributes:

        sample:     the points that will be clustered
                    sample row: [time, X, Y, Z]

        labels:     a vector of size len(sample) that saves the labels of each
                    datapoint in the sample

        maxLabel:   the largest value in the labels attribute
    '''


    def getLabels(self):
        return self.labels


    def getSampleLabels(self):
        # Return all points as arrays of points with same label
        sampleLabels = []

        # First noise
        sampleLabels.append(self.sample[self.labels == -1])

        # Then actual labels
        for i in range(0, self.maxLabel + 1):
            sampleLabels.append(self.sample[self.labels == i])

        return np.array(sampleLabels)


    def getCentres(self):
        # the centre of a cluster is the average of the time, x, y, z columns
        # and the number of points of that cluster
        # centres row: [time, x, y, z, clusterSize]
        centres = []
        for i in range(0, self.maxLabel + 1):
            # Average time, x, y, z of cluster of label i
            centresRow = np.mean(self.sample[self.labels == i], axis = 0)
            # Append the number of points of label i
            centresRow = np.append(centresRow, (self.labels == i).sum())
            centres.append(centresRow)

        return np.array(centres)


    def plotSampleLabels(self, ax):

        for i in range(0, self.maxLabel + 1):
            dataPointsLabel = self.sample[self.labels == i]
            ax.scatter(dataPointsLabel[:, 1], dataPointsLabel[:, 2], dataPointsLabel[:, 3], alpha = 0.6, marker = '1', s = 1)

        # Plot noise
        dataPointsLabel = self.sample[self.labels == -1]
        ax.scatter(dataPointsLabel[:, 1], dataPointsLabel[:, 2], dataPointsLabel[:, 3], c = 'k', alpha = 0.1, marker = '.', s = 1)


    def plotSampleLabelsAltAxes(self, ax):

        for i in range(0, self.maxLabel + 1):
            dataPointsLabel = self.sample[self.labels == i]
            ax.scatter(dataPointsLabel[:, 3], dataPointsLabel[:, 1], dataPointsLabel[:, 2], alpha = 0.6, marker = '1', s = 1)

        # Plot noise
        dataPointsLabel = self.sample[self.labels == -1]
        ax.scatter(dataPointsLabel[:, 3], dataPointsLabel[:, 1], dataPointsLabel[:, 2], c = 'k', alpha = 0.01, marker = '.', s = 1)


    def plotCentres(self, ax):

        centres = self.getCentres()
        if len(centres) > 0:
            ax.scatter(centres[:, 1], centres[:, 2], centres[:, 3], c = 'r', marker = 'D', s = 1)


    def plotCentresAltAxes(self, ax):

        centres = self.getCentres()
        if len(centres) > 0:
            ax.scatter(centres[:, 3], centres[:, 1], centres[:, 2], c = 'r', marker = 'D', s = 5)


    def getSampleLabelsTraces(self, noise = True):
        traces = []
        for i in range(0, self.maxLabel + 1):
            dataPointsLabel = self.sample[self.labels == i][0::10]
            traces.append(go.Scatter3d(
                x = dataPointsLabel[:, 1],
                y = dataPointsLabel[:, 2],
                z = dataPointsLabel[:, 3],
                mode = 'markers',
                marker = dict(
                    size = 2,
                    opacity = 0.8,
                    colorscale = 'Cividis'
                )
            ))

        if noise == True:
            # Noise points
            dataPointsLabel = self.sample[self.labels == -1][0::10]
            traces.append(go.Scatter3d(
                x = dataPointsLabel[:, 1],
                y = dataPointsLabel[:, 2],
                z = dataPointsLabel[:, 3],
                mode = 'markers',
                marker = dict(
                    size = 1,
                    opacity = 0.4,
                    color = 'black'
                )
            ))

        return traces


    def getCentresTrace(self):
        centres = self.getCentres()
        color = centres[:, -1]
        color = color[[0, len(color) // 2, -1]]
        trace = go.Scatter3d(
            x=centres[:, 1],
            y=centres[:, 2],
            z=centres[:, 3],
            mode='markers',
            marker=dict(
                size=2,
                color=color,   # set color to sample size
                colorscale='Cividis',   # choose a colorscale
                colorbar=dict(
                    title="Number of clustered points"
                ),
                opacity=0.5
            )
        )

        return trace




class HDBSCANClusterer:

    def __init__(self,
                 min_cluster_size = 5,
                 min_samples = None,
                 allow_single_cluster = False):

        if 0 < min_cluster_size < 2:
            print("\n[WARNING]: min_cluster_size was set to 2, as it was {} < 2\n".format(min_cluster_size))
            min_cluster_size = 2

        self.clusterer = hdbscan.HDBSCAN(min_cluster_size = min_cluster_size,
                                         min_samples = min_samples,
                                         core_dist_n_jobs = -1,
                                         allow_single_cluster = allow_single_cluster)

        '''
        # Call pept.PointData constructor with dummy data
        super().__init__([[0., 0., 0., 0.]],
                         sample_size = 0,
                         overlap = 0,
                         verbose = False)
        '''


    @property
    def min_cluster_size(self):
        return self.clusterer.min_cluster_size


    @min_cluster_size.setter
    def min_cluster_size(self, new_min_cluster_size):
        self.clusterer.min_cluster_size = new_min_cluster_size


    @property
    def min_samples(self):
        return self.clusterer.min_cluster_size


    @min_samples.setter
    def min_samples(self, new_min_samples):
        self.clusterer.min_samples = new_min_samples


    @property
    def allow_single_cluster(self):
        return self.clusterer.allow_single_cluster


    @allow_single_cluster.setter
    def allow_single_cluster(self, option):
        self.clusterer.allow_single_cluster = option


    def fit_sample(self,
                   sample,
                   store_labels = False,
                   noise = False,
                   as_array = False,
                   verbose = False):

        if verbose:
            start = time.time()

        # sample row: [time, x, y, z]
        if sample.ndim != 2 or sample.shape[1] < 4:
            raise ValueError('\n[ERROR]: sample should have two dimensions (M, N), where N >= 4. Received {}\n'.format(sample.shape))

        # Only cluster based on [x, y, z]
        labels = self.clusterer.fit_predict(sample[:, 1:4])
        max_label = labels.max()

        centres = []
        clustered_cutpoints = []

        # the centre of a cluster is the average of the time, x, y, z columns
        # and the number of points of that cluster
        # centres row: [time, x, y, z, ..etc.., cluster_size]
        centres = []
        for i in range(0, max_label + 1):
            # Average time, x, y, z of cluster of label i
            centres_row = np.mean(sample[labels == i], axis = 0)
            # Append the number of points of label i => cluster_size
            centres_row = np.append(centres_row, (labels == i).sum())
            centres.append(centres_row)

        centres = np.array(centres)

        if not as_array:
            centres = pept.PointData(centres,
                                     sample_size = 0,
                                     overlap = 0,
                                     verbose = False)

        # Return all cutpoints as a list of numpy arrays for every label
        # where the last column of an array is the label
        if store_labels:
            # Create a list of numpy arrays with rows: [t, x, y, z, ..etc.., label]
            if noise:
                cutpoints = sample[labels == -1]
                cutpoints = np.insert(cutpoints, cutpoints.shape[1], -1, axis = 1)
                clustered_cutpoints.append(cutpoints)

            for i in range(0, max_label + 1):
                cutpoints = sample[labels == i]
                cutpoints = np.insert(cutpoints, cutpoints.shape[1], i, axis = 1)
                clustered_cutpoints.append(cutpoints)

            clustered_cutpoints = np.vstack(np.array(clustered_cutpoints))

            if not as_array:
                clustered_cutpoints = pept.PointData(clustered_cutpoints,
                                                     sample_size = 0,
                                                     overlap = 0,
                                                     verbose = False)

        if verbose:
            end = time.time()
            print("Fitting one sample took {} seconds".format(end - start))

        return [centres, clustered_cutpoints]


    def fit_cutpoints(self,
                      cutpoints,
                      store_labels = False,
                      noise = False,
                      verbose = True):

        if verbose:
            start = time.time()

        if not isinstance(cutpoints, pept.PointData):
            raise Exception('[ERROR]: cutpoints should be an instance of pept.PointData (or any class inheriting from it)')

        # Fit all samples in `cutpoints` in parallel using joblib
        # Collect all outputs as a list
        data_list = Parallel(n_jobs = -1)(delayed(self.fit_sample)(sample,
                                                store_labels = store_labels,
                                                noise = noise,
                                                as_array = True) for sample in tqdm(cutpoints))

        # Access joblib.Parallel output as list comprehensions
        centres = np.array([row[0] for row in data_list if len(row[0]) != 0])
        if len(centres) != 0:
            centres = pept.PointData(np.vstack(centres),
                                     sample_size = 0,
                                     overlap = 0,
                                     verbose = False)

        if store_labels:
            clustered_cutpoints = np.array([row[1] for row in data_list if len(row[1]) != 0])
            clustered_cutpoints = pept.PointData(np.vstack(np.array(clustered_cutpoints)),
                                                 sample_size = 0,
                                                 overlap = 0,
                                                 verbose = False)

        if verbose:
            end = time.time()
            print("Fitting cutpoints took {} seconds".format(end - start))

        if store_labels:
            return [centres, clustered_cutpoints]
        else:
            return [centres, []]




    def fitSampleParallel(self, sample, saveLabels = False, saveNoise = False):
        # Function that will be parallelised
        # Needs to return a list of the needed outputs
        self.fitSample(sample)
        centres = self.getCentres()

        if saveLabels:
            sampleLabelsTraces = self.getSampleLabelsTraces(noise = saveNoise)
        else:
            sampleLabelsTraces = []

        return [centres, sampleLabelsTraces]


    def clusterIterable(self,
                        samples,
                        saveLabels = False,
                        saveNoise = False):

        # Call joblib Parallel subroutine for every sample needed
        # Collects returned data as a list of outputs
        dataList = Parallel(n_jobs = -1)(delayed(self.fitSampleParallel)(sample, saveLabels, saveNoise) for sample in tqdm(samples))

        # Access the output from the parallelised function as list comprehensions
        # centres row: [time, x, y, z, meanMidpointsClusterSize]
        self.centres = np.array([dataRow[0] for dataRow in dataList if len(dataRow[0]) != 0])
        if len(self.centres) != 0:
            self.centres = np.vstack(self.centres)

        # Collect all the lists of clustered midpoints traces and flatten them
        # Plot the midpoints only if plotMidpoints == True
        if saveLabels:
            self.labelsTraces = [dataRow[1] for dataRow in dataList]
            self.labelsTraces = [elem for sublist in self.labelsTraces for elem in sublist]

            return [self.centres, self.labelsTraces]
        else:
            return [self.centres, []]




class HDBSCANclustererAuto(ClustererBase):
    '''
    Automatically vary the harshness of an HDBSCAN clusterer until
    a particle has been found
    '''

    def __init__(self, sampleSize, k = [0.001, 0.8], nIter = 5):
        # sampleSize is the average number of points per particle

        # k is a correction factor ranging from 0 to 1, having the
        # physical meaning of the minimum ratio of points that
        # need to be part of the cluster.
        #   eg. k = 1 means all points need to be close together => harsher
        #       k = 0.2 means 20% of points need to be close => more lenient
        self.sampleSize = sampleSize
        self.k = k
        self.nIter = nIter

        if self.sampleSize <= 0:
            raise Exception('[ERROR]: sampleSize needs to be a positive integer')

        # Create a list of clusterers with min_cluster_size ranging from min to max, with
        # nIter maximum iterations to find a centre

        self.autoClusterer = []

        # lenient because smaller minimum cluster size => more points into cluster
        min_min_cluster_size = k[0] * self.sampleSize #self.sampleSize / 30#10

        # harsh
        max_min_cluster_size = k[1] * self.sampleSize #self.sampleSize / 20#5

        # from harsh to lenient
        sizes = np.linspace(max_min_cluster_size, min_min_cluster_size, self.nIter)

        for min_cluster_size in sizes:
            min_cluster_size = int(min_cluster_size)
            if min_cluster_size < 2:
                min_cluster_size = 2
            self.autoClusterer.append(hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                                      core_dist_n_jobs = -1))


    def changeParameters(self, sampleSize = 0, nIter = 5):
        if sampleSize != 0:
            if self.sampleSize <= 0 or type(sampleSize) != int:
                raise Exception('[ERROR]: sampleSize needs to be a positive integer')
            else:
                self.sampleSize = sampleSize

        self.nIter = nIter

        if self.sampleSize != 0:
            # Create a list of clusterers with min_cluster_size ranging from min to max, with
            # nIter maximum iterations to find a centre

            self.autoClusterer = []

            # lenient because smaller minimum cluster size => more points into cluster
            min_min_cluster_size = self.sampleSize / 10

            # harsh
            max_min_cluster_size = self.sampleSize / 5

            # from harsh to lenient
            sizes = np.linspace(max_min_cluster_size, min_min_cluster_size, self.nIter)

            for min_cluster_size in sizes:
                self.autoClusterer.append(hdbscan.HDBSCAN(min_cluster_size=int(min_cluster_size)))


    def fitSample(self, sample):
        #start = time.time()

        # sample row: [time, x, y, z]
        self.sample = sample

        for clusterer in self.autoClusterer:
            # Only cluster based on [x, y, z]
            clusterer.fit(self.sample[:, 1:4])
            self.labels = clusterer.labels_
            self.maxLabel = self.labels.max()

            # Stop when a particle was found
            if self.maxLabel >= 0:
                break

        #end = time.time()
        #print("Fitting one sample took {} seconds".format(end - start))



class ClusterCentres:
    '''
    Helper/wrapper class that receives cluster centres and can
    yield samples of adaptive size for a second pass of clustering
    '''

    def __init__(self, centres, sampleSize=50, overlap=0):
        self.centres = centres
        self.numberOfCentres = len(centres)
        self.sampleSize = sampleSize
        self.overlap = overlap

        self._index = 0


    def getNumberOfSamples(self):
        if self.numberOfCentres >= self.sampleSize:
            return (self.numberOfCentres - self.sampleSize) // (self.sampleSize - self.overlap) + 1
        else:
            return 0


    def getCentresSampleN(self, sampleN):
        if (sampleN > self.getNumberOfSamples()) or sampleN <= 0:
            raise Exception("\n\n[ERROR]: Trying to access a non-existent sample: asked for sample number {}, when there are {} samples\n".format(sampleN, self.getNumberOfSamples()))

        startIndex = (sampleN - 1) * (self.sampleSize - self.overlap)
        return self.centres[startIndex:(startIndex + self.sampleSize)]


    def __len__(self):
        return self.getNumberOfSamples()


    def __iter__(self):
        return self


    def __next__(self):
        # sampleSize > 0 => return slices
        if self._index != 0:
            self._index = self._index + self.sampleSize - self.overlap
        else:
            self._index = self._index + self.sampleSize


        if self._index > self.numberOfCentres:
            self._index = 0
            raise StopIteration

        return self.centres[(self._index - self.sampleSize):self._index]




class HDBSCANtwoPassClusterer:
    '''
    Helper class which implements a second pass of clustering
    over the centres found by 'clusterer'

    Two-pass clustering helps with spatial resolution and
    determining the trajectory of a moving particle

        clusterer: a clusterer which has the method fitSample(sample) that will
            fit samples of midpoints (first-pass clustering)

        clusterer2: a clusterer which has the method fitSample(sample) that will
            fit samples of centres (second-pass clustering)

        samplesMidpoints: an array of samples of midpoints.
            samplesMidpoints.shape: (numberOfSamples, numberOfMidpointsPerSample, 4)
            (can alternatively send a Midpoints instance which can be iterated over)

        centresSampleSize: number of centres to send per sample  to the second reclustering
    '''

    def __init__(self, clusterer, clusterer2, samplesMidpoints, centresSampleSize, centresOverlap=0):
        # clusterer is an instance which has the method .fitSample(sample)
        # samplesMidpoints is an array of samples of midpoints
        # samplesMidpoints.shape: (numberOfSamples, numberOfMidpointsPerSample, 4)
        # (can alternatively send a Midpoints instance which can be iterated over)

        self.clusterer = clusterer
        self.clusterer2 = clusterer2
        self.samplesMidpoints = samplesMidpoints
        self.centresSampleSize = centresSampleSize
        self.centresOverlap = centresOverlap
        self.centres = []
        self.centres2 = []


    def fit(self):

        start = time.time()
        # First pass of clustering for midpoints
        for sample in self.samplesMidpoints:
            self.clusterer.fitSample(sample)
            newCentres = self.clusterer.getCentres()
            self.centres.extend(newCentres)

        end = time.time()
        print('First pass of clustering took {} s'.format(end - start))

        # centres row: [time, x, y, z, clusterSize]
        self.centres = np.array(self.centres)

        self.samplesCentres = ClusterCentres(self.centres, self.centresSampleSize, self.centresOverlap)

        print('Total number of samples of centres: {}'.format(self.samplesCentres.getNumberOfSamples()))

        start = time.time()
        # Second pass of clustering for centres
        for sample in self.samplesCentres:
            self.clusterer2.fitSample(sample)
            newCentres = self.clusterer2.getCentres()
            self.centres2.extend(newCentres)

        end = time.time()
        print('Second pass of clustering took {} s'.format(end - start))

        # centres2 row: [time, x, y, z, meanCentresClusterSize, clusterSize]
        self.centres2 = np.array(self.centres2)


    def getCentres(self):
        return self.centres


    def getCentres2(self):
        return self.centres2


    def plotCentres(self, ax):

        if len(self.centres) > 0:
            ax.scatter(self.centres[:, 1], self.centres[:, 2], self.centres[:, 3],
                    c = 'r', marker = 'D', s = 5)


    def plotCentresAltAxes(self, ax):

        if len(self.centres) > 0:
            ax.scatter(self.centres[:, 3], self.centres[:, 1], self.centres[:, 2],
                    c = 'r', marker = 'D', s = 5)


    def plotCentres2(self, ax):

        if len(self.centres2) > 0:
            ax.scatter(self.centres2[:, 1], self.centres2[:, 2], self.centres2[:, 3],
                    c = 'b', marker = 'D', s = 5)


    def plotCentres2AltAxes(self, ax):

        if len(self.centres2) > 0:
            ax.scatter(self.centres2[:, 3], self.centres2[:, 1], self.centres2[:, 2],
                    c = 'b', marker = 'D', s = 5)




class TrajectorySeparation:

    def __init__(self, centres, pointsToCheck = 25, maxDistance = 20, maxClusterDiff = 500):
        # centres row: [time, x, y, z, clusterSize]
        # Make sure the trajectory is memory-contiguous for efficient
        # KDTree partitioning
        self.centres = np.ascontiguousarray(centres)
        self.pointsToCheck = pointsToCheck
        self.maxDistance = maxDistance
        self.maxClusterDiff = maxClusterDiff

        # For every point in centres, save a set of the trajectory
        # indices of the trajectories that they are part of
        #   eg. centres[2] is part of trajectories 0 and 1 =>
        #   trajectoryIndices[2] = {0, 1}
        # Initialise a vector of empty sets of size len(centres)
        self.trajectoryIndices = np.array([ set() for i in range(len(self.centres)) ])

        # For every trajectory found, save a list of the indices of
        # the centres that are part of that trajectory
        #   eg. trajectory 1 is comprised of centres 3, 5 and 8 =>
        #   centresIndices[1] = [3, 5, 8]
        self.centresIndices = [[]]

        # Maximum trajectory index
        self.maxIndex = 0


    def findTrajectories(self):

        for i, currentPoint in enumerate(self.centres):

            if i == 0:
                # Add the first point to trajectory 0
                self.trajectoryIndices[0].add(self.maxIndex)
                self.centresIndices[self.maxIndex].append(0)
                self.maxIndex += 1
                continue

            # Search for the closest previous pointsToCheck points
            # within a given maxDistance
            startIndex = i - self.pointsToCheck
            endIndex = i

            if startIndex < 0:
                startIndex = 0

            # Construct a KDTree from the x, y, z (1:4) of the
            # selected points. Get the indices for all the points within
            # maxDistance of the currentPoint
            tree = cKDTree(self.centres[startIndex:endIndex, 1:4])
            closestIndices = tree.query_ball_point(currentPoint[1:4], self.maxDistance, n_jobs=-1)
            closestIndices = np.array(closestIndices) + startIndex

            # If no point was found, it is a new trajectory. Continue
            if len(closestIndices) == 0:
                self.trajectoryIndices[i].add(self.maxIndex)
                self.centresIndices.append([i])
                self.maxIndex += 1
                continue

            # For every close point found, search for all the trajectory indices
            #   - If all trajectory indices sets are equal and of a single value
            #   then currentPoint is part of the same trajectory
            #   - If all trajectory indices sets are equal, but of more values,
            #   then currentPoint diverged from an intersection of trajectories
            #   and is part of a single trajectory => separate it
            #
            #   - If every pair of trajectory indices sets is not disjoint, then
            #   currentPoint is only one of them
            #   - If there exists a pair of trajectory indices sets that is
            #   disjoint, then currentPoint is part of all of them

            # Select the trajectories of all the points that were found
            # to be the closest
            closestTrajectories = self.trajectoryIndices[closestIndices]
            #print("closestTrajectories:")
            #print(closestTrajectories)

            # If all the closest points are part of the same trajectory
            # (just one!), then the currentPoint is part of it too
            if (np.all(closestTrajectories == closestTrajectories[0]) and
                len(closestTrajectories[0]) == 1):

                self.trajectoryIndices[i] = closestTrajectories[0]
                self.centresIndices[ next(iter(closestTrajectories[0])) ].append(i)
                continue

            # Otherwise, check the points based on their cluster size
            else:
                # Create a list of all the trajectories that were found to
                # intersect
                #print('\nIntersection:')
                closestTrajIndices = list( set().union(*closestTrajectories) )

                #print("ClosestTrajIndices:")
                #print(closestTrajIndices)

                # For each close trajectory, calculate the mean cluster size
                # of the last lastPoints points
                lastPoints = 50

                # Keep track of the mean cluster size that is the closest to
                # the currentPoint's clusterSize
                currentClusterSize = currentPoint[4]
                #print("currentClusterSize = {}".format(currentClusterSize))
                closestTrajIndex = -1
                clusterSizeDiff = self.maxClusterDiff

                for trajIndex in closestTrajIndices:
                    #print("trajIndex = {}".format(trajIndex))

                    trajCentres = self.centres[ self.centresIndices[trajIndex] ]
                    #print("trajCentres:")
                    #print(trajCentres)
                    meanClusterSize = trajCentres[-lastPoints:][:, 4].mean()
                    #print("meanClusterSize = {}".format(meanClusterSize))
                    #print("clusterSizeDiff = {}".format(clusterSizeDiff))
                    #print("abs diff = {}".format(np.abs( currentClusterSize - meanClusterSize )))
                    if np.abs( currentClusterSize - meanClusterSize ) < clusterSizeDiff:
                        closestTrajIndex = trajIndex
                        clusterSizeDiff = np.abs( currentClusterSize - meanClusterSize )

                if closestTrajIndex == -1:
                    #self.trajectoryIndices[i] = set(closestTrajIndices)
                    #for trajIndex in closestTrajIndices:
                    #    self.centresIndices[trajIndex].append(i)

                    print("\n**** -1 ****\n")
                    break
                else:
                    #print("ClosestTrajIndex found = {}".format(closestTrajIndex))
                    self.trajectoryIndices[i] = set([closestTrajIndex])
                    self.centresIndices[closestTrajIndex].append(i)




            '''
            # If the current point is not part of any trajectory, assign it
            # the maxIndex and increment it
            if len(self.trajectoryIndices[i]) == 0:
                self.trajectoryIndices[i].append(self.maxIndex)
                self.maxIndex += 1

            print(self.trajectoryIndices[i])
            print(self.maxIndex)

            # Construct a KDTree from the numberOfPoints in front of
            # the current point
            tree = cKDTree(self.trajectory[(i + 1):(i + self.numberOfPoints + 2)][1:4])

            # For every trajectory that the current point is part of,
            # find the closest points in front of it
            numberOfIntersections = len(self.trajectoryIndices[i])
            dist, nextPointsIndices = tree.query(currentPoint, k=numberOfIntersections, distance_upper_bound=self.maxDistance, n_jobs=-1)

            print(nextPointsIndices)

            # If the current point is part of more trajectories,
            # an intersection happened. Call subroutine to part
            # the trajectories
            if numberOfIntersections > 1:
                for j in range(0, len(self.trajectoryIndices[i])):
                    trajIndex = self.trajectoryIndices[i][j]
                    self.trajectoryIndices[i + 1 + nextPointsIndices[j]].append(trajIndex)

            else:
                self.trajectoryIndices[i + 1 + nextPointsIndices].append(self.trajectoryIndices[i][0])

            print(self.trajectoryIndices)
            '''


    def getTrajectories(self):

        self.individualTrajectories = []
        for trajCentres in self.centresIndices:
            self.individualTrajectories.append(self.centres[trajCentres])

        self.individualTrajectories = np.array(self.individualTrajectories)
        return self.individualTrajectories

        '''
        self.individualTrajectories = [ [] for i in range(0, self.maxIndex + 1) ]
        for i in range(0, len(self.trajectoryIndices)):
            for trajIndex in self.trajectoryIndices[i]:
                self.individualTrajectories[trajIndex].append(self.centres[i])

        self.individualTrajectories = np.array(self.individualTrajectories)
        for i in range(len(self.individualTrajectories)):
            if len(self.individualTrajectories[i]) > 0:
                self.individualTrajectories[i] = np.vstack(self.individualTrajectories[i])
        return self.individualTrajectories
        '''


    def plotTrajectoriesAltAxes(self, ax):
        trajectories = self.getTrajectories()
        for traj in trajectories:
            if len(traj) > 0:
                ax.scatter(traj[:, 3], traj[:, 1], traj[:, 2], marker='D', s=10)




class PlotlyGrapher:
    # Helper class that automatically generates Plotly graphs
    # for the PEPT data

    def __init__(self, rows=1, cols=1, xlim = [0, 500],
                 ylim = [0, 500], zlim = [0, 712], subplot_titles = ['  ']):
        self.rows = rows
        self.cols = cols

        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim

        self.subplot_titles = subplot_titles
        self.subplot_titles.extend(['  '] * (rows * cols - len(subplot_titles)))


    def createFigure(self):
        # Create subplots and set limits

        specs = [[{"type": "scatter3d"}] * self.cols] * self.rows

        self.fig = make_subplots(rows = self.rows, cols = self.cols,
                    specs = specs, subplot_titles = self.subplot_titles,
                    horizontal_spacing = 0.005, vertical_spacing = 0.05)

        self.fig['layout'].update(margin = dict(l=0,r=0,b=30,t=30), showlegend = False)

        # For every subplot (scene), set axes' ratios and limits
        # Also set the y axis to point upwards
        # Plotly naming convention of scenes: 'scene', 'scene2', etc.
        for i in range(self.rows):
            for j in range(self.cols):
                if i == j == 0:
                    scene = 'scene'
                else:
                    scene = 'scene{}'.format(i * self.cols + j + 1)

                # Justify subplot title on the left
                self.fig.layout.annotations[i * self.cols + j].update(x = (j + 0.08) / self.cols)
                self.fig['layout'][scene].update(aspectmode = 'manual',
                                                 aspectratio = {'x': 1, 'y': 1, 'z': 1},
                                                 camera = {'up': {'x': 0, 'y': 1, 'z':0},
                                                           'eye': {'x': 1, 'y': 1, 'z': 1}},
                                                 xaxis = {'range': self.xlim,
                                                          'title': {'text': "<i>x</i> (mm)"}},
                                                 yaxis = {'range': self.ylim,
                                                          'title': {'text': "<i>y</i> (mm)"}},
                                                 zaxis = {'range': self.zlim,
                                                          'title': {'text': "<i>z</i> (mm)"}}
                                                 )

        return self.fig


    def getFigure(self):
        return self.fig


    def addDataAsTrace(self, data, row, col, size = 2, color = None):
        # Expected data row: [time, x, y, z, ...]
        if len(data) != 0:
            trace = go.Scatter3d(
                x = data[:, 1],
                y = data[:, 2],
                z = data[:, 3],
                mode = 'markers',
                marker = dict(
                    size = size,
                    color = color,
                    opacity = 0.8
                )
            )

            self.fig.add_trace(trace, row = row, col = col)


    def addDataAsTraceColorbar(self, data, row, col, titleColorbar = None, size = 3):
        # Expected data row: [time, x, y, z, ...]
        if len(data) != 0:
            if titleColorbar != None:
                colorbar=dict(title=titleColorbar)
            else:
                colorbar = dict()

            trace = go.Scatter3d(
                x=data[:, 1],
                y=data[:, 2],
                z=data[:, 3],
                mode='markers',
                marker=dict(
                    size=size,
                    color=data[:, -1],   # set color to sample size
                    colorscale='Magma',     # choose a colorscale
                    colorbar=colorbar,
                    opacity=0.8
                )
            )

            self.fig.add_trace(trace, row = row, col = col)


    def addDataAsTraceLine(self, data, row, col):
        # Expected data row: [time, x, y, z, ...]
        if len(data) != 0:
            trace = go.Scatter3d(
                x=data[:, 1],
                y=data[:, 2],
                z=data[:, 3],
                mode='lines',
                line=dict(
                    width=4,
                )
            )

            self.fig.add_trace(trace, row = row, col = col)


    def addTrace(self, trace, row, col):
        # Add precomputed trace
        # Can accept HDBSCANclusterer.getCentresTrace() output
        if len(trace) != 0:
            self.fig.add_traces(trace, rows=row, cols=col)


    def addTraces(self, traces, row, col):
        # Add precomputed traces
        # Can accept HDBSCANclusterer.getSampleLabelsTraces() output
        if len(traces) != 0:
            self.fig.add_traces(traces, rows=[row]*len(traces), cols=[col]*len(traces))


    def showFigure(self):
        self.fig.show()









