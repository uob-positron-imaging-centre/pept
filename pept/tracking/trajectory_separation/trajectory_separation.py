#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : trajectory_separation.py
# License: License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 23.08.2019


'''The *peptml* module implements a hierarchical density-based clustering
algorithm for general Positron Emission Particle Tracking (PEPT)

The module aims to provide general classes which can
then be used in a script file as the user sees fit. For example scripts,
look at the base of the pept library.

The peptml subpackage accepts any instace of the LineData base class
and can create matplotlib- or plotly-based figures.

PEPTanalysis requires the following packages:

* **numpy**
* **math**
* **matplotlib.pyplot** and **mpl_toolkits.mplot3d** for 3D matplotlib-based plotting
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


import  math
import  time
import  numpy                                   as          np

from    scipy.spatial                           import      cKDTree
from    joblib                                  import      Parallel,       delayed
from    tqdm                                    import      tqdm

import  pept




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




class TrajectorySeparation:

    def __init__(self,
                 centres,
                 points_to_check = 25,
                 max_distance = 20,
                 max_cluster_size_diff = 500,
                 points_cluster_size = 50):

        # centres row: [time, x, y, z, clusterSize]
        # Make sure the trajectory is memory-contiguous for efficient
        # KDTree partitioning
        self.centres = np.ascontiguousarray(centres)
        self.points_to_check = points_to_check
        self.max_distance = max_distance
        self.max_cluster_size_diff = max_cluster_size_diff
        self.points_cluster_size = points_cluster_size

        # For every point in centres, save a set of the trajectory
        # indices of the trajectories that they are part of
        #   eg. centres[2] is part of trajectories 0 and 1 =>
        #   trajectory_indices[2] = {0, 1}
        # Initialise a vector of empty sets of size len(centres)
        self.trajectory_indices = np.array([ set() for i in range(len(self.centres)) ])

        # For every trajectory found, save a list of the indices of
        # the centres that are part of that trajectory
        #   eg. trajectory 1 is comprised of centres 3, 5 and 8 =>
        #   centresIndices[1] = [3, 5, 8]
        self.centres_indices = [[]]

        # Maximum trajectory index
        self.max_index = 0


    def find_trajectories(self):

        for i, current_point in enumerate(self.centres):

            if i == 0:
                # Add the first point to trajectory 0
                self.trajectory_indices[0].add(self.max_index)
                self.centres_indices[self.max_index].append(0)
                self.max_index += 1
                continue

            # Search for the closest previous pointsToCheck points
            # within a given maxDistance
            start_index = i - self.points_to_check
            end_index = i

            if start_index < 0:
                start_index = 0

            # Construct a KDTree from the x, y, z (1:4) of the
            # selected points. Get the indices for all the points within
            # maxDistance of the currentPoint
            tree = cKDTree(self.centres[start_index:end_index, 1:4])
            closest_indices = tree.query_ball_point(current_point[1:4],
                                                    self.max_distance,
                                                    n_jobs = -1)
            closest_indices = np.array(closest_indices) + start_index

            # If no point was found, it is a new trajectory. Continue
            if len(closest_indices) == 0:
                self.trajectory_indices[i].add(self.max_index)
                self.centres_indices.append([i])
                self.max_index += 1
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
            closest_trajectories = self.trajectory_indices[closest_indices]
            #print("closestTrajectories:")
            #print(closestTrajectories)

            # If all the closest points are part of the same trajectory
            # (just one!), then the currentPoint is part of it too
            if (np.all(closest_trajectories == closest_trajectories[0]) and
                len(closest_trajectories[0]) == 1):

                self.trajectory_indices[i] = closest_trajectories[0]
                self.centres_indices[ next(iter(closest_trajectories[0])) ].append(i)
                continue

            # Otherwise, check the points based on their cluster size
            else:
                # Create a list of all the trajectories that were found to
                # intersect
                #print('\nIntersection:')
                closest_traj_indices = list( set().union(*closest_trajectories) )

                #print("ClosestTrajIndices:")
                #print(closestTrajIndices)

                # For each close trajectory, calculate the mean cluster size
                # of the last points_cluster_size points

                # Keep track of the mean cluster size that is the closest to
                # the currentPoint's clusterSize
                current_cluster_size = current_point[4]
                #print("currentClusterSize = {}".format(currentClusterSize))
                closest_traj_index = -1
                cluster_size_diff = self.max_cluster_size_diff

                for traj_index in closest_traj_indices:
                    #print("trajIndex = {}".format(trajIndex))

                    traj_centres = self.centres[ self.centres_indices[traj_index] ]
                    #print("trajCentres:")
                    #print(trajCentres)
                    mean_cluster_size = traj_centres[-self.points_cluster_size:][:, 4].mean()
                    #print("meanClusterSize = {}".format(meanClusterSize))
                    #print("clusterSizeDiff = {}".format(clusterSizeDiff))
                    #print("abs diff = {}".format(np.abs( currentClusterSize - meanClusterSize )))
                    if np.abs( current_cluster_size - mean_cluster_size ) < cluster_size_diff:
                        closest_traj_index = traj_index
                        cluster_size_diff = np.abs( current_cluster_size - mean_cluster_size )

                if closest_traj_index == -1:
                    #self.trajectoryIndices[i] = set(closestTrajIndices)
                    #for trajIndex in closestTrajIndices:
                    #    self.centresIndices[trajIndex].append(i)

                    print("\n**** -1 ****\n")
                    break
                else:
                    #print("ClosestTrajIndex found = {}".format(closestTrajIndex))
                    self.trajectory_indices[i] = set([closest_traj_index])
                    self.centres_indices[closest_traj_index].append(i)

        individual_trajectories = []
        for traj_centres in self.centres_indices:
            individual_traj = pept.PointData(self.centres[traj_centres],
                                             sample_size = 0,
                                             overlap = 0,
                                             verbose = False)
            individual_trajectories.append(individual_traj)

        return individual_trajectories



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






