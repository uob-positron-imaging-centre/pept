#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : peptml.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 22.06.2020


import  textwrap

import  numpy   as      np
from    tqdm    import  tqdm

import  pept

from    pept.tracking                           import  peptml
import  pept.tracking.trajectory_separation     as      tsp

from    pept.visualisation                      import  PlotlyGrapher


# Colours for printing to the terminal, taken from blender build scripts.
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'        # Return to normal colour
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class PEPTMLUser:
    '''This is an example script analysing PEPT data using the PEPT-ML
    algorithm from the `pept` library.

    The script is encapsulated in a class such that, after running it (i.e.
    instantiating the class), all the relevant analysis objects can be accessed
    as class attributes (e.g. LoRs, clusterers, tracer locations).

    The class has three methods:

    1. __init__ : the class constructor that is run when instantiating it; it
       contains the actual PEPT-ML algorithm.
    2. typecheck_and_create_attributes : typecheck input parameters and create
       class attributes.
    3. minimal : the minimal PEPT-ML code, without pretty-printing to the
       terminal.

    It is more than recommended to look through the code, copy and hack into
    it! The class constructor (`__init__`) contains the PEPT-ML algorithm with
    printing functionality to the terminal; if you'd like the code without such
    additions, use the equivalent code in the `minimal` method.

    The main steps of the PEPT-ML-User algorithm are:

    -- Done beforehand, by the user --

    1. Initialise LoRs in a `pept.LineData` instance (if they are stored in a
       file, you can use `pept.utilities.read_csv`). **You must do this before
       using this script**.

    -- Done by this script --

    2. Compute the cutpoints from the LoRs using the `peptml.Cutpoints` class.
    3. Cluster them once using `peptml.HDBSCANClusterer` to find the tracer
       locations (i.e. cluster *centres*) - this is *1-pass clustering*.
    4. Cluster the tracer locations from the previous step again, using another
       `peptml.HDBSCANClusterer` - this is *2-pass clustering*.
    5. Segregate the intertwined points from the tracer trajectories using
       `trajectory_separation.segregate_trajectories`.
    6. Reconnect the trajectories of tracers with gaps in their tracks (such as
       when they collide) based on their cluster size using
       `trajectory_separation.connect_trajectories`.

    For more details check out the PEPT-ML paper (doi.org/10.1063/1.5129251)
    and the documentation of the functions used.

    The parameters below are, of course, easy to change. Their default values
    represent some (heuristically) good starting parameters that should work in
    the majority of cases. Perhaps the most important ones to change would be
    `k1` (a larger value represents harsher clustering; lower it for very
    noisy data), `trajectory_cut_distance` (for segregating pre-tracked points)
    and `max_signature_difference` (for reconnecting trajectories with gaps).

    Attributes
    ----------
    lors : pept.LineData instance
        The input lines of response (LoRs). As they can be stored in various
        sources (CSV files, binary data, different scanner geometries), it is
        expected that users initialise them as `pept.LineData` (or a subclass
        thereof) *before* using this script.
    max_tracers : int
        The maximum number of tracers that may be present in the field of view
        at any time. Note that the PEPT-ML algorithm does not *need* this
        number (and can work well even when tracers leave the FoV); however, it
        is useful when setting the sample sizes, overlaps and clustering
        parameters.
    sample_size : int, optional
        The `lors.sample_size`, used when transforming LoRs into cutpoints. By
        default, it is set to 200 * `max_tracers`.
    overlap : int, optional
        The `lors.overlap`, used when transforming LoRs into cutpoints. By
        default, it is set to 150 * `max_tracers`.
    sample_size2 : int, optional
        The `centres.sample_size`, used for the second pass of clustering (i.e.
        re-clustering the tracer locations found from the cutpoints). By
        default, it is set to 30 * `max_tracers`; this small value is fine as
        the data is already very clean after the first pass of clustering.
    overlap2 : int, optional
        The `centres.overlap`, used for the second pass of clustering (i.e.
        re-clustering the tracer locations found from the cutpoints). By
        default, it is set to `sample_size2` - 1, such that the loss of time
        resolution is minimised.
    max_distance : float, default 0.15
        The maximum allowed distance between two LoRs for their cutpoint to be
        considered; used by `peptml.Cutpoints`.
    k1 : float, default 0.15
        The fitting parameter for setting the `min_cluster_size` of the
        `clusterer` used for the first pass of clustering. As the number of
        cutpoints around a tracer scales with the sample_size and (inversely)
        `max_tracers`, the following formula is used -
        `min_cluster_size` = int(k1 * cutpoints.sample_size / max_tracers)`.
        A larger value represents "harsher" clustering. Lower it for very noisy
        data.
    k2 : float, default 0.6
        The fitting parameter for setting the `min_cluster_size` of the
        `clusterer2` used for the second pass of clustering. As the number of
        points around a tracer scales with the sample_size and (inversely)
        `max_tracers`, the following formula is used -
        `min_cluster_size2` = int(k2 * centres.sample_size / max_tracers)`.
        A larger value represents "harsher" clustering.
    points_window : int, optional
        The number of points included in the sliding window used when
        segregating the intertwined points from multiple trajectories using
        `pept.tracking.trajectory_separation.segregate_trajectories`. By
        default, it is set to 10 * `max_tracers`.
    trajectory_cut_distance : float, default 10.0
        Cut trajectories that are further apart than this value and consider
        the remaining ones distinct trajectories. This depends on how close the
        points in the data set are. Used by
        `pept.tracking.trajectory_separation.segregate_trajectories`.
    max_time_difference : float, default 150
        Only try to re-connect trajectories with gaps if their ends are closer
        time-wise than `max_time_difference`. Used by
        `pept.tracking.trajectory_separation.connect_trajectories`. Check the
        consistency of the units used for time.
    max_signature_difference : float, default 100
        Re-connect trajectories with gaps if the difference in cluster size
        (i.e. tracer signature) of the closest points is smaller than this
        value. An average of the 50 closest points is used. Used by
        `pept.tracking.trajectory_separation.connect_trajectories`.

    cutpoints : pept.tracking.peptml.Cutpoints instance
        The cutpoints computed from `lors`. It is a subclass of
        `pept.PointData`, such that all its methods and attributes can be used.

    min_cluster_size : int
        The minimum cluster size parameter used by the clusterer for the first
        pass of clustering.
    clusterer : pept.tracking.peptml.HDBSCANClusterer instance
        The clusterer used for the first pass of clustering. It also appends
        the "cluster size" (i.e. the number of points in the cluster
        surrounding a tracer) to `centres`.
    centres : pept.PointData instance
        The tracer locations (i.e. cluster *centres*) found after the first
        pass of clustering. The last column represents the cluster size; that
        is, the number of points in the cluster surrounding the tracer. This
        can act as a tracer signature for when their identity is temporarily
        lost, like when they collide.
    labelled_cutpoints : pept.PointData instance
        The cutpoints that were clustered, including their cluster labels as
        the last column.

    min_cluster_size2 : int
        The minimum cluster size parameter used by the clusterer for the second
        pass of clustering.
    clusterer2 : pept.tracking.peptml.HDBSCANClusterer instance
        The clusterer used for the second pass of clustering.
    centres2 : pept.PointData instance
        The tracer locations (i.e. cluster *centres*) found after the second
        pass of clustering.
    labelled_cetres : pept.PointData instance
        The centres that were clustered, including their cluster labels as the
        last column.

    segregated_trajectories : pept.PointData instance
        The segregated trajectories found from the intertwined tracer locations
        in `centres2`. It contains the same data as `centres2`, but with an
        added column representing the trajectory label (or index); they are
        indexed from 0, with -1 signifying noise. Returned by
        `pept.tracking.trajectory_separation.segregate_trajectories`.
    trajectories : pept.PointData instance
        The final separated trajectories, after reconnecting the
        `segregated_trajectories` based on their cluster size using
        `pept.tracking.trajectory_separation.connect_trajectories`. It contains
        the same data as `segregated_trajectories`, but the last column (i.e.
        trajectory label) is changed to reflect the final trajectories.
    trajectory_list : list of numpy.ndarray
        The trajectories from `trajectories` are separated into a list of
        individual trajectories, based on their last column (i.e. trajectory
        label).

    grapher : pept.visualisation.PlotlyGrapher instance
        The grapher organising the 3D subplots.

    Methods
    -------
    typecheck_and_create_attributes(\
        lors,\
        max_tracers,\
        sample_size = None,\
        overlap = None,\
        sample_size2 = None,\
        overlap2 = None,\
        max_distance = 0.15,\
        k1 = 0.15,\
        k2 = 0.6,\
        points_window = None,\
        trajectory_cut_distance = 10.0,\
        max_time_difference = 150.0,\
        max_signature_difference = 100.0\
    )
        Typecheck input parameters and create class attributes. Used by the
        class constructor for checks that are not relevant to the actual
        PEPT-ML aglorithm.
    minimal(\
        lors,\
        max_tracers,\
        sample_size = None,\
        overlap = None,\
        sample_size2 = None,\
        overlap2 = None,\
        max_distance = 0.15,\
        k1 = 0.15,\
        k2 = 0.6,\
        points_window = None,\
        trajectory_cut_distance = 10.0,\
        max_time_difference = 150,\
        max_signature_difference = 100\
    )
        The minimal code for using the PEPT-ML algorithm, without
        pretty-printing to the terminal. Useful for copy-pasting the code and
        hacking into it.

    Raises
    ------
    TypeError
        If `lors` is not an instance of `pept.LineData` (or subclass thereof!).
    AssertionError
        If `max_tracers` is smaller than 1.

    Notes
    -----
    The `pept` package is modular, meaning it can be used in many ways, to suit
    (m)any circumstances. This is just an example, which you can modify as you
    see fit.

    If you're not sure what a class or method does, you can always consult the
    documentation online, at "uob-positron-imaging-centre.github.io", or by
    adding a questions mark in front of a package / module / class / function
    name in the Python shell or using the `help` command. For example:

    >>> import pept
    >>> ?pept.LineData          # Short documentation
    >>> ??pept.LineData         # Long documentation
    >>> help(pept.LineData)     # Even more info

    Examples
    --------
    An example call would be:

    >>> lors_raw = pept.utilities.read_csv("csv_file.csv")  # two tracers CSV
    >>> lors = pept.LineData(lors_raw)
    >>> user = pept.cookbook.PEPTMLUser(lors, 2)            # max_tracers = 2

    '''

    def __init__(
        self,
        lors,                               # Essential, pept.LineData
        max_tracers,                        # Essential, int
        sample_size = None,                     # Default 200 * max_tracers
        overlap = None,                         # Default 150 * max_tracers
        sample_size2 = None,                    # Default 30 * max_tracers
        overlap2 = None,                        # Default sample_size2 - 1
        max_distance = 0.15,                # mm
        k1 = 0.15,                          # non-dimensional, important
        k2 = 0.6,                           # non-dimensional
        points_window = None,                   # Default 10 * max_tracers
        trajectory_cut_distance = 10.0,     # mm, important
        max_time_difference = 150,          # ms
        max_signature_difference = 100      # cluster size difference
    ):
        '''PEPTMLUser class constructor. This is the PEPT-ML algorithm with
        pretty-printing of the relevant objects to the terminal.

        Parameters
        ----------
        lors : pept.LineData instance
            The input lines of response (LoRs). As they can be stored in
            various sources (CSV files, binary data, different scanner
            geometries), it is expected that users initialise them as
            `pept.LineData` (or a subclass thereof) *before* using this script.
        max_tracers : int
            The maximum number of tracers that may be present in the field of
            view at any time. Note that the PEPT-ML algorithm does not *need*
            this number (and can work well even when tracers leave the FoV);
            however, it is useful to know when setting the sample sizes,
            overlaps and clustering parameters.
        sample_size : int, optional
            The `lors.sample_size`, used when transforming LoRs into cutpoints.
            By default, it is set to 200 * `max_tracers`.
        overlap : int, optional
            The `lors.overlap`, used when transforming LoRs into cutpoints. By
            default, it is set to 150 * `max_tracers`.
        sample_size2 : int, optional
            The `centres.sample_size`, used for the second pass of clustering
            (i.e. re-clustering the tracer locations found from the cutpoints).
            By default, it is set to 30 * `max_tracers`; this small value is
            fine as the data is already very clean after the first pass of
            clustering.
        overlap2 : int, optional
            The `centres.overlap`, used for the second pass of clustering (i.e.
            re-clustering the tracer locations found from the cutpoints). By
            default, it is set to `sample_size2` - 1, such that the loss of
            time resolution is minimised.
        max_distance : float, default 0.15
            The maximum allowed distance between two LoRs for their cutpoint to
            be considered; used by `peptml.Cutpoints`.
        k1 : float, default 0.15
            The fitting parameter for setting the `min_cluster_size` of the
            `clusterer` used for the first pass of clustering. As the number of
            cutpoints around a tracer scales with the sample_size and
            (inversely) `max_tracers`, the following formula is used -
            `min_cluster_size` = int(k1 * cutpoints.sample_size / max_tracers)`
            A larger value represents "harsher" clustering. Lower it for very
            noisy data.
        k2 : float, default 0.6
            The fitting parameter for setting the `min_cluster_size` of the
            `clusterer2` used for the second pass of clustering. As the number
            of points around a tracer scales with the sample_size and
            (inversely) `max_tracers`, the following formula is used -
            `min_cluster_size2` = int(k2 * centres.sample_size / max_tracers)`.
            A larger value represents "harsher" clustering.
        points_window : int, optional
            The number of points included in the sliding window used when
            segregating the intertwined points from multiple trajectories using
            `pept.tracking.trajectory_separation.segregate_trajectories`. By
            default, it is set to 10 * `max_tracers`.
        trajectory_cut_distance : float, default 10.0
            Cut trajectories that are further apart than this value and
            consider the remaining ones as distinct trajectories. This depends
            on how close the points in the data set are. Used by
            `pept.tracking.trajectory_separation.segregate_trajectories`.
        max_time_difference : float, default 150
            Only try to re-connect trajectories with gaps if their ends are
            closer time-wise than `max_time_difference`. Used by
            `pept.tracking.trajectory_separation.connect_trajectories`. Check
            the consistency of the units used for time.
        max_signature_difference : float, default 100
            Re-connect trajectories with gaps if the difference in cluster size
            (i.e. tracer signature) of the closest points is smaller than this
            value. An average of the 50 closest points is used. Used by
            `pept.tracking.trajectory_separation.connect_trajectories`.

        Raises
        ------
        TypeError
            If `lors` is not an instance of `pept.LineData` (or subclass
            thereof!).
        AssertionError
            If `max_tracers` is smaller than 1.
        '''

        # Initial type-checking of input parameters + properties
        # initialisation.
        self.typecheck_and_create_attributes(
            lors, max_tracers, sample_size, overlap, sample_size2, overlap2,
            max_distance, k1, k2, points_window, trajectory_cut_distance,
            max_time_difference, max_signature_difference
        )

        # Start the PEPT-ML algorithm; you can copy-paste the code below and
        # hack into it! It's recommended actually!
        print(
            bcolors.HEADER + "=" * 70 +     # Use colours when printing
            "\nPEPT-ML User Start\n" +
            "=" * 70 + bcolors.ENDC         # End colour
        )

        # Print input LoRs and max_tracers. For `lors`, print its printable
        # representation (i.e. long representation) using repr().
        self.lors.sample_size = self.sample_size
        self.lors.overlap = self.overlap
        print((
            f"\n{bcolors.OKBLUE}"
            f"LoRs:{bcolors.ENDC}\n"
            f"{repr(self.lors)}\n"
        ))
        print((
            f"{bcolors.OKBLUE}"
            f"max_tracers = {self.max_tracers}"
            f"{bcolors.ENDC}\n\n"
        ))

        # Transform the LoRs into cutpoints using the `Cutpoints` class. It is
        # a `pept.PointData` subclass, so we inherit all its attributes and
        # methods!
        print(
            bcolors.HEADER + "-" * 70 +
            "\nTransforming LoRs into Cutpoints\n" +
            "-" * 70 + bcolors.ENDC
        )
        print((
            f"\n{bcolors.OKBLUE}"
            f"max_distance = {self.max_distance}"
            f"{bcolors.ENDC}\n"
        ))

        self.cutpoints = peptml.Cutpoints(lors, max_distance)
        print((
            f"\n{bcolors.OKGREEN}"
            f"Finding cutpoints completed.\n\n"
            f"{bcolors.OKBLUE}"
            f"Cutpoints:{bcolors.ENDC}\n"
            f"{repr(self.cutpoints)}\n\n"
        ))

        # Find the tracer locations using peptml.HDBSCANClusterer. This is
        # 1-pass clustering.
        print(
            bcolors.HEADER + "-" * 70 +
            "\nClustering Cutpoints using HDBSCANClusterer " +
            "(1-pass clustering)\n" +
            "-" * 70 + bcolors.ENDC
        )

        # The cluster size (i.e. number of cutpoints around each tracer
        # location) scales with the maximum number of tracers that might be
        # present in the field of view at any time.
        self.min_cluster_size = int(self.k1 * self.cutpoints.sample_size /
                                    self.max_tracers)
        self.clusterer = peptml.HDBSCANClusterer(self.min_cluster_size)
        print((
            f"\n{bcolors.OKBLUE}"
            f"Clusterer for 1-pass clustering:{bcolors.ENDC}\n"
            f"{repr(self.clusterer)}\n"
        ))

        self.centres, self.labelled_cutpoints = self.clusterer.fit(
            self.cutpoints, get_labels = True
        )

        # Set the sample size and overlap of the tracer locations from the
        # first pass of clustering. The data is already very clean from the
        # previous steps, so we'll set a small sample size and large overlap to
        # minimise smoothing, or the temporal resolution lost.
        self.centres.sample_size = self.sample_size2
        self.centres.overlap = self.overlap2

        print((
            f"\n{bcolors.OKGREEN}"
            f"1-pass clustering completed.\n\n"
            f"{bcolors.OKBLUE}"
            f"Cluster centres (i.e. tracer locations):{bcolors.ENDC}\n"
            f"{repr(self.centres)}\n\n{bcolors.OKBLUE}"
            f"Labelled cutpoints:{bcolors.ENDC}\n"
            f"{repr(self.labelled_cutpoints)}\n\n"
        ))

        # Create smoother, tighter tracer locations using
        # peptml.HDBSCANClusterer. This is 2-pass clustering.
        print(
            bcolors.HEADER + "-" * 70 +
            "\nClustering Centres using HDBSCANClusterer " +
            "(2-pass clustering)\n" +
            "-" * 70 + bcolors.ENDC
        )

        self.min_cluster_size2 = int(self.k2 * self.centres.sample_size /
                                     self.max_tracers)
        self.clusterer2 = peptml.HDBSCANClusterer(self.min_cluster_size2)
        print((
            f"\n{bcolors.OKBLUE}"
            f"Clusterer for 2-pass clustering:{bcolors.ENDC}\n"
            f"{repr(self.clusterer2)}\n"
        ))

        self.centres2, self.labelled_centres = self.clusterer2.fit(
            self.centres, get_labels = True
        )

        print((
            f"\n{bcolors.OKGREEN}"
            f"2-pass clustering completed.\n\n"
            f"{bcolors.OKBLUE}"
            f"Tracer locations after 2-pass clustering:{bcolors.ENDC}\n"
            f"{repr(self.centres2)}\n\n"
        ))

        # Segregate the intertwined points of the previous step.
        print(
            bcolors.HEADER + "-" * 70 +
            "\nSegregate the intertwined points from the 2-pass clustering\n" +
            "-" * 70 + bcolors.ENDC
        )

        self.segregated_trajectories = tsp.segregate_trajectories(
            self.centres2, self.points_window, self.trajectory_cut_distance
        )

        print((
            f"\n{bcolors.OKGREEN}"
            f"Trajectory segregation completed.\n\n"
            f"{bcolors.OKBLUE}"
            f"Segregated trajectories:{bcolors.ENDC}\n"
            f"{repr(self.segregated_trajectories)}\n"
        ))

        print((
            f"\n{bcolors.OKBLUE}"
            "Number of trajectories found after segregation: "
            f"{self.segregated_trajectories.points[:, -1].max() + 1}\n\n"
        ))

        # Connect the segregated trajectories based on their cluster size.
        print(
            bcolors.HEADER + "-" * 70 +
            "\nConnect the segregated trajectories based on their cluster " +
            "size\n" + "-" * 70 + bcolors.ENDC
        )

        self.trajectories = tsp.connect_trajectories(
            self.segregated_trajectories,
            self.max_time_difference,
            self.max_signature_difference
        )

        self.trajectory_list = pept.utilities.group_by_column(
            self.trajectories.points, -1
        )

        print((
            f"\n{bcolors.OKGREEN}"
            f"Trajectory connecting completed.\n\n"
            f"{bcolors.OKBLUE}"
            f"Trajectories:{bcolors.ENDC}\n"
            f"{repr(self.trajectories)}\n"
        ))

        print((
            f"\n{bcolors.OKBLUE}"
            "Number of trajectories found: "
            f"{self.trajectories.points[:, -1].max() + 1}\n\n"
        ))

        # Plotting time!
        print(
            bcolors.HEADER + "-" * 70 +
            "\nPlotting Results using PlotlyGrapher\n" +
            "-" * 70 + bcolors.ENDC
        )

        # Instantiate a PlotlyGrapher instance with 2x3 subplots:
        self.grapher = PlotlyGrapher(rows = 2, cols = 3, subplot_titles = [
            "LoRs, first two samples",
            "Labelled cutpoints, first 10 samples, 1-pass",
            "Tracer locations, 1-pass (coloured cluster size)",
            "Tracer locations, 2-pass (coloured cluster size)",
            "Segregated tracer locations",
            "Individual trajectories"
        ])

        # Add traces to the grapher figure. row = 1, col = 1 by default.
        # If there are less than 2 samples of LoRs, plot all LoRs.
        if len(self.lors) >= 2:
            self.grapher.add_trace(self.lors.lines_trace([0, 1]))
        else:
            self.grapher.add_lines(self.lors)

        # If there are less than 10 samples of cutpoints, plot all of them
        if len(self.labelled_cutpoints) >= 10:
            self.grapher.add_trace(self.labelled_cutpoints.points_trace(
                range(0, 10)
            ), row = 1, col = 2)
        else:
            self.grapher.add_trace(self.labelled_cutpoints.points_trace(),
                                   col = 2)
        self.grapher.add_points(self.centres, row = 1, col = 3)
        self.grapher.add_points(self.centres2, colorbar_col = 4,
                                row = 2, col = 1)
        self.grapher.add_points(self.segregated_trajectories, row = 2, col = 2)
        self.grapher.add_points(self.trajectories, row = 2, col = 3)

        self.grapher.show()

        print((
            f"\n{bcolors.OKGREEN}"
            f"Plotting completed."
            f"{bcolors.ENDC}\n\n"
        ))

        print(
            bcolors.HEADER + "=" * 70 +     # Use colorrs when printing
            "\nPEPT-ML User End\n" +
            "=" * 70 + bcolors.ENDC         # End colour
        )


    @staticmethod
    def minimal(
        lors,                               # Essential, pept.LineData
        max_tracers,                        # Essential, int
        sample_size = None,                     # Default 200 * max_tracers
        overlap = None,                         # Default 150 * max_tracers
        sample_size2 = None,                    # Default 30 * max_tracers
        overlap2 = None,                        # Default sample_size2 - 1
        max_distance = 0.15,                # mm
        k1 = 0.15,                          # non-dimensional, important
        k2 = 0.6,                           # non-dimensional
        points_window = None,                   # Default 10 * max_tracers
        trajectory_cut_distance = 10.0,     # mm, important
        max_time_difference = 150,          # ms
        max_signature_difference = 100      # cluster size difference
    ):
        # Check `lors` is an instance of `pept.LineData`; this includes any
        # subclass! (e.g. `pept.scanners.ParallelScreens`). If it's not, raise
        # an error.
        if not isinstance(lors, pept.LineData):
            raise TypeError(
                textwrap.fill((
                    "\n[ERROR]: `lors` should be an instance of `pept."
                    f"LineData` (or subclass thereof). Received {type(lors)}."
                    "Note: you can read in LoR data from a CSV file using "
                    "`pept.utilities.read_csv`; initialise them in a `pept."
                    "LineData` before calling this function. Check this "
                    "script's documentation for more.\n"
                ), replace_whitespace = False)
            )

        # If any of sample_size, overlap, sample_size2, overlap2 were left
        # to the default `None`, set them to the default values.
        if sample_size is None:
            sample_size = 200 * max_tracers
        if overlap is None:
            overlap = 150 * max_tracers
        if sample_size2 is None:
            sample_size2 = 30 * max_tracers
        if overlap2 is None:
            overlap2 = sample_size2 - 1

        # Set points_window to a default value.
        if points_window is None:
            points_window = max_tracers * 10

        # Beginning of the PEPT-ML User algorithm:

        # Compute cutpoints from LoRs
        lors.sample_size = sample_size
        lors.overlap = overlap
        cutpoints = peptml.Cutpoints(lors, max_distance)

        # 1-pass clustering
        min_cluster_size = int(k1 * cutpoints.sample_size / max_tracers)
        clusterer = peptml.HDBSCANClusterer(min_cluster_size)
        centres, labelled_cutpoints = clusterer.fit(cutpoints,
                                                    get_labels = True)

        # 2-pass clustering
        centres.sample_size = sample_size2
        centres.overlap = overlap2

        min_cluster_size2 = int(k2 * centres.sample_size / max_tracers)
        clusterer2 = peptml.HDBSCANClusterer(min_cluster_size2)
        centres2, labelled_centres = clusterer2.fit(centres,
                                                    get_labels = True)

        # Segregate the intertwined points
        segregated_trajectories = tsp.segregate_trajectories(
            centres2, points_window, trajectory_cut_distance
        )

        # Connect segregated points based on cluster size
        trajectories = tsp.connect_trajectories(
            segregated_trajectories,
            max_time_difference,
            max_signature_difference
        )

        # Create a list of individual trajectories
        trajectory_list = pept.utilities.group_by_column(
            trajectories.points, -1
        )

        # Plot everything using PlotlyGrapher
        grapher = PlotlyGrapher(rows = 2, cols = 3, subplot_titles = [
            "LoRs, first two samples",
            "Labelled cutpoints, first 10 samples, 1-pass",
            "Tracer locations, 1-pass (coloured cluster size)",
            "Tracer locations, 2-pass (coloured cluster size)",
            "Segregated tracer locations",
            "Individual trajectories"
        ])

        grapher.add_trace(lors.lines_trace([0, 1]))
        grapher.add_trace(labelled_cutpoints.points_trace(range(0, 10)),
                          col = 2)
        grapher.add_points(centres, col = 3)
        grapher.add_points(centres2, colorbar_col = 4, row = 2, col = 1)
        grapher.add_points(segregated_trajectories, row = 2, col = 2)
        grapher.add_points(trajectories, row = 2, col = 3)

        grapher.show()


    def typecheck_and_create_attributes(
        self,
        lors,
        max_tracers,
        sample_size = None,
        overlap = None,
        sample_size2 = None,
        overlap2 = None,
        max_distance = 0.15,
        k1 = 0.15,
        k2 = 0.6,
        points_window = None,
        trajectory_cut_distance = 10.0,
        max_time_difference = 150.0,
        max_signature_difference = 100.0
    ):
        # Check `lors` is an instance of `pept.LineData`; this includes any
        # subclass! (e.g. `pept.scanners.ParallelScreens`). If it's not, raise
        # an error.
        if not isinstance(lors, pept.LineData):
            raise TypeError(
                textwrap.fill((
                    "\n[ERROR]: `lors` should be an instance of `pept."
                    f"LineData` (or subclass thereof). Received {type(lors)}."
                    "Note: you can read in LoR data from a CSV file using "
                    "`pept.utilities.read_csv`; initialise them in a `pept."
                    "LineData` before calling this function. Check this "
                    "script's documentation for more.\n"
                ), replace_whitespace = False)
            )

        # Attach lors to our class instance so we can access it as a class
        # property. Attach the next variables used too.
        self.lors = lors

        # Make sure max_tracers is an `int` that's larger or equal to 1
        max_tracers = int(max_tracers)
        assert max_tracers >= 1, textwrap.fill(
            "[ERROR]: `max_tracers` should be an `int` >= 1."
        )
        self.max_tracers = max_tracers

        # If any of sample_size, overlap, sample_size2, overlap2 were left
        # to the default `None`, set them to the default values. Otherwise
        # type-check them; also set lors.sample_size and lors.overlap.
        sample_size = 200 * max_tracers if sample_size is None \
                                        else int(sample_size)
        self.sample_size = sample_size

        overlap = 150 * max_tracers if overlap is None \
                                    else int(overlap)
        self.overlap = overlap

        sample_size2 = 30 * max_tracers if sample_size2 is None \
                                        else int(sample_size2)
        self.sample_size2 = sample_size2

        overlap2 = sample_size2 - 1 if overlap2 is None \
                                    else int(overlap2)
        self.overlap2 = overlap2

        # Type-check max_distance (for cutpoints), k1 (for 1-pass clusterer)
        # and k2 (for the 2-pass clusterer)
        max_distance = float(max_distance)
        self.max_distance = max_distance

        k1 = float(k1)
        self.k1 = k1

        k2 = float(k2)
        self.k2 = k2

        # Type-check parameters for segregate_trajectories
        if points_window is None:
            self.points_window = self.max_tracers * 10
        else:
            self.points_window = int(points_window)

        self.trajectory_cut_distance = float(trajectory_cut_distance)

        # Type-check parameters for connect_trajectories
        self.max_time_difference = float(max_time_difference)
        self.max_signature_difference = float(max_signature_difference)




class PEPTMLFindParameters:
    '''This is an example script that visually helps find the best clustering
    parameters for the PEPT-ML algorithm from the `pept` library.

    The script is encapsulated in a class such that, after running it (i.e.
    instantiating the class), all the relevant analysis objects can be accessed
    as class attributes (e.g. LoRs, clusterers, tracer locations).

    The class has three methods:

        1. __init__ : the class constructor that is run when instantiating it;
           it contains the actual PEPT-ML algorithm.
        2. typecheck_and_create_attributes : typecheck input parameters and
           create class attributes.
        3. minimal : the minimal PEPT-ML code, without pretty-printing to the
           terminal.

    It is more than recommended to look through the code, copy and hack into
    it! The class constructor (`__init__`) contains the PEPT-ML algorithm with
    printing functionality to the terminal; if you'd like the code without such
    additions, use the equivalent code in the `minimal` method.

    The main steps of the PEPT-ML-User algorithm are:

    -- Done beforehand, by the user --

        1. Initialise LoRs in a `pept.LineData` instance (if they are stored in
           a file, you can use `pept.utilities.read_csv`). **You must do this
           before using this script**.

    -- Done by this script --

        2. Compute the cutpoints from the LoRs using the `peptml.Cutpoints`
           class.
        3. For every value in `np.linspace(k1[0], k1[1], iterations)`, cluster
           the cutpoints using `peptml.HDBSCANClusterer` to find the tracer
           locations (i.e. cluster *centres*) - this is *1-pass clustering*.

    The function then plots the clustered cutpoints with colour-coded cluster
    labels and tracer locations for each value of k1 that was chosen. The user
    can then visually select the value of k1 that works best for clustering the
    given dataset.

    For more details check out the PEPT-ML paper (doi.org/10.1063/1.5129251)
    and the documentation of the functions used.

    Attributes
    ----------
    lors : pept.LineData instance
        The input lines of response (LoRs). As they can be stored in various
        sources (CSV files, binary data, different scanner geometries), it is
        expected that users initialise them as `pept.LineData` (or a subclass
        thereof) *before* using this script.
    max_tracers : int
        The maximum number of tracers that may be present in the field of view
        at any time. Note that the PEPT-ML algorithm does not *need* this
        number (and can work well even when tracers leave the FoV); however, it
        is useful when setting the sample sizes, overlaps and clustering
        parameters.
    sample_size : int, optional
        The `lors.sample_size`, used when transforming LoRs into cutpoints. By
        default, it is set to 200 * `max_tracers`.
    overlap : int, optional
        The `lors.overlap`, used when transforming LoRs into cutpoints. By
        default, it is set to 150 * `max_tracers`.
    max_distance : float, default 0.15
        The maximum allowed distance between two LoRs for their cutpoint to be
        considered; used by `peptml.Cutpoints`.
    k1 : list[float], default [0.05, 0.8]
        The range to explor of the fitting parameter for setting the
        `min_cluster_size` of the `clusterer` used for the first pass of
        clustering. As the number of cutpoints around a tracer scales with the
        sample_size and (inversely) `max_tracers`, the following formula is
        used -
        `min_cluster_size` = int(k1 * cutpoints.sample_size / max_tracers)`.
        A larger value represents "harsher" clustering. This class will iterate
        through `iterations` equally-spaced values between k1[0] and k1[1] for
        one-pass clustering.
    iterations : int, default 5
        The number of values of k1 that will be taken between k1[0] and k1[1]
        for the different clusterers.

    cutpoints : pept.tracking.peptml.Cutpoints instance
        The cutpoints computed from `lors`. It is a subclass of
        `pept.PointData`, such that all its methods and attributes can be used.

    ks : np.ndarray
        The values of k1 that will be used for the clusterers. It is equivalent
        to a call to `np.linspace(k1[0], k1[1], iterations, dtype = float)`.
    clusterers : list[pept.tracking.peptml.HDBSCANClusterer]
        The list of clusterers with different `min_cluster_size` values from
        `ks` that will be used for one-pass clustering.

    centres : list[pept.PointData]
        A list of the tracer locations (i.e. cluster *centres*) found after the
        first pass of clustering, for every value of k1 in `ks`. The last
        column in each `pept.PointData` represents the cluster size; that is,
        the number of points in the cluster surrounding the tracer. This can
        act as a tracer signature for when their identity is temporarily lost,
        like when they collide.
    labelled_cutpoints : list[pept.PointData]
        A list of the cutpoints that were clustered for every value of k1 in
        `ks`, including their cluster labels as the last column.

    grapher : pept.visualisation.PlotlyGrapher instance
        The grapher organising the 3D subplots.

    Methods
    -------
    typecheck_and_create_attributes(
        lors,                       # Essential, pept.LineData
        max_tracers,                # Essential, int
        sample_size = None,             # Default 200 * max_tracers
        overlap = None,                 # Default 150 * max_tracers
        max_distance = 0.15,        # mm
        k1 = [0.05, 0.8],           # non-dimensional, important
        iterations = 5              # number of points between k1[0] and k1[1]
    )
        Typecheck input parameters and create class attributes. Used by the
        class constructor for checks that are not relevant to the actual
        PEPT-ML aglorithm.
    minimal(
        lors,                       # Essential, pept.LineData
        max_tracers,                # Essential, int
        sample_size = None,             # Default 200 * max_tracers
        overlap = None,                 # Default 150 * max_tracers
        sample_size2 = None,            # Default 30 * max_tracers
        overlap2 = None,                # Default sample_size2 - 1
        max_distance = 0.15,        # mm
        k1 = [0.05, 0.8],           # non-dimensional, important
        iterations = 5              # number of points between k1[0] and k1[1]
    )
        The minimal code for using the algorithm in this class, without
        pretty-printing to the terminal. Useful for copy-pasting the code and
        hacking into it.

    Raises
    ------
    TypeError
        If `lors` is not an instance of `pept.LineData` (or subclass thereof!).
    AssertionError
        If `max_tracers` is smaller than 1.

    Notes
    -----
    The `pept` package is modular, meaning it can be used in many ways, to suit
    (m)any circumstances. This is just an example, which you can modify as you
    see fit.

    If you're not sure what a class or method does, you can always consult the
    documentation online, at "uob-positron-imaging-centre.github.io", or by
    adding a questions mark in front of a package / module / class / function
    name in the Python shell or using the `help` command. For example:

    >>> import pept
    >>> ?pept.LineData          # Short documentation
    >>> ??pept.LineData         # Long documentation
    >>> help(pept.LineData)     # Even more info

    Examples
    --------
    An example call would be:

    >>> lors_raw = pept.utilities.read_csv("csv_file.csv")  # two tracers CSV
    >>> lors = pept.LineData(lors_raw)
    >>> user = pept.cookbook.PEPTMLFindParameters(lors, 2)  # max_tracers = 2

    '''

    def __init__(
        self,
        lors,                       # Essential, pept.LineData
        max_tracers,                # Essential, int
        sample_size = None,             # Default 200 * max_tracers
        overlap = None,                 # Default 150 * max_tracers
        max_distance = 0.15,        # mm
        k1 = [0.05, 0.8],           # non-dimensional, important
        iterations = 5              # number of points between k1[0] and k1[1]
    ):
        '''The PEPTMLFindParameters class constructor.

        Parameters
        ----------
        lors : pept.LineData instance
            The input lines of response (LoRs). As they can be stored in
            various sources (CSV files, binary data, different scanner
            geometries), it is expected that users initialise them as
            `pept.LineData` (or a subclass thereof) *before* using this script.
        max_tracers : int
            The maximum number of tracers that may be present in the field of
            view at any time. Note that the PEPT-ML algorithm does not *need*
            this number (and can work well even when tracers leave the FoV);
            however, it is useful when setting the sample sizes, overlaps and
            clustering parameters.
        sample_size : int, optional
            The `lors.sample_size`, used when transforming LoRs into cutpoints.
            By default, it is set to 200 * `max_tracers`.
        overlap : int, optional
            The `lors.overlap`, used when transforming LoRs into cutpoints. By
            default, it is set to 150 * `max_tracers`.
        max_distance : float, default 0.15
            The maximum allowed distance between two LoRs for their cutpoint to
            be considered; used by `peptml.Cutpoints`.
        k1 : list[float], default [0.05, 0.8]
            The range to explor of the fitting parameter for setting the
            `min_cluster_size` of the `clusterer` used for the first pass of
            clustering. As the number of cutpoints around a tracer scales with
            the sample_size and (inversely) `max_tracers`, the following
            formula is used -
            `min_cluster_size` = int(k1 * cutpoints.sample_size / max_tracers)`
            A larger value represents "harsher" clustering. This class will
            iterate through `iterations` equally-spaced values between k1[0]
            and k1[1] for one-pass clustering.
        iterations : int, default 5
            The number of values of k1 that will be taken between k1[0] and
            k1[1] for the different clusterers.

        Raises
        ------
        TypeError
            If `lors` is not an instance of `pept.LineData` (or subclass
            thereof!).
        AssertionError
            If `max_tracers` is smaller than 1.
        '''

        # Initial type-checking of input parameters + properties
        # initialisation.
        self.typecheck_and_create_attributes(
            lors, max_tracers, sample_size, overlap, max_distance, k1,
            iterations
        )

        # Start the PEPT-ML algorithm; you can copy-paste the code below and
        # hack into it! It's recommended actually!
        print(
            bcolors.HEADER + "=" * 70 +     # Use colours when printing
            "\nPEPT-ML Find Parameters Start\n" +
            "=" * 70 + bcolors.ENDC         # End colour
        )

        # Print input LoRs and max_tracers. For `lors`, print its printable
        # representation (i.e. long representation) using repr().
        self.lors.sample_size = self.sample_size
        self.lors.overlap = self.overlap
        print((
            f"\n{bcolors.OKBLUE}"
            f"LoRs:{bcolors.ENDC}\n"
            f"{repr(self.lors)}\n"
        ))
        print((
            f"{bcolors.OKBLUE}"
            f"max_tracers = {self.max_tracers}"
            f"{bcolors.ENDC}\n\n"
        ))

        # Transform the LoRs into cutpoints using the `Cutpoints` class. It is
        # a `pept.PointData` subclass, so we inherit all its attributes and
        # methods!
        print(
            bcolors.HEADER + "-" * 70 +
            "\nTransforming LoRs into Cutpoints\n" +
            "-" * 70 + bcolors.ENDC
        )
        print((
            f"\n{bcolors.OKBLUE}"
            f"max_distance = {self.max_distance}"
            f"{bcolors.ENDC}\n"
        ))

        self.cutpoints = peptml.Cutpoints(lors, max_distance)
        print((
            f"\n{bcolors.OKGREEN}"
            f"Finding cutpoints completed.\n\n"
            f"{bcolors.OKBLUE}"
            f"Cutpoints:{bcolors.ENDC}\n"
            f"{repr(self.cutpoints)}\n\n"
        ))

        # Cluster our cutpoints using peptml.HDBSCANClusterer for every value
        # of k1 between k1[0] and k1[1], at `iterations` points.
        print(
            bcolors.HEADER + "-" * 70 +
            "\nClustering Cutpoints using HDBSCANClusterer " +
            "(1-pass clustering)\n" +
            "-" * 70 + bcolors.ENDC
        )

        self.ks = np.linspace(k1[0], k1[1], iterations, dtype = float)
        print((
            "\nFormula for setting the clusterer min_cluster_size:\n"
            "    min_cluster_size = int(k1 * cutpoints.sample_size / "
            "max_tracers)\n\n"
            f"{bcolors.OKBLUE}For values of k1 from:\n"
            f"    {self.ks}{bcolors.ENDC}\n\n"
        ))

        # Create a list of clusterers.
        self.clusterers = [peptml.HDBSCANClusterer(
            int(k * self.cutpoints.sample_size / self.max_tracers)
        ) for k in self.ks]

        # For each clusterer, fit the cutpoints and append the results to a
        # list. Use `tqdm` to show a progress bar.
        self.centres = []
        self.labelled_cutpoints = []
        for clusterer in tqdm(self.clusterers):
            centres1, labelled_cutpoints1 = clusterer.fit(
                self.cutpoints,
                get_labels = True,
                verbose = False
            )

            self.centres.append(centres1)
            self.labelled_cutpoints.append(labelled_cutpoints1)

        print((
            f"\n{bcolors.OKGREEN}"
            f"Clustering cutpoints for every value of k1 completed.\n\n"
            f"{bcolors.OKBLUE}" +
            textwrap.fill(
                "Tracer locations (i.e. cluster centres) for every value of "
                "k1 were saved as a list in the `centres` attribute of this "
                "class. Similarily, the labelled cutpoints (i.e. the "
                "cutpoints with an extra column of cluster labels) were saved "
                "as a list in the `labelled_cutpoints` attribute of this "
                "class."
            ) +
            f"{bcolors.ENDC}\n\n"
        ))

        # Plotting time!
        print(
            bcolors.HEADER + "-" * 70 +
            "\nPlotting Results using PlotlyGrapher\n" +
            "-" * 70 + bcolors.ENDC
        )

        # Instantiate a PlotlyGrapher instance with 2 rows and `iterations`
        # columns:
        self.grapher = PlotlyGrapher(
            rows = 2,
            cols = self.iterations,
            subplot_titles = [f"k1 = {k}, labelled cutpoints" for k in self.ks]
                              + [f"k1 = {k}, cluster centres" for k in self.ks]
        )

        # For every value of k1 we used, on the first row plot the labelled
        # cutpoints, and on the second row plot the tracer locations (i.e.
        # cluster centres).
        for i in range(self.iterations):
            self.grapher.add_trace(
                self.labelled_cutpoints[i].points_trace(), row = 1, col = i + 1
            )
            self.grapher.add_trace(
                self.centres[i].points_trace(), row = 2, col = i + 1
            )

        self.grapher.show()

        print((
            f"\n{bcolors.OKGREEN}"
            f"Plotting completed."
            f"{bcolors.ENDC}\n\n"
        ))

        print(
            bcolors.HEADER + "=" * 70 +     # Use colorrs when printing
            "\nPEPT-ML Find Parameters End\n" +
            "=" * 70 + bcolors.ENDC         # End colour
        )


    @staticmethod
    def minimal(
        lors,                       # Essential, pept.LineData
        max_tracers,                # Essential, int
        sample_size = None,             # Default 200 * max_tracers
        overlap = None,                 # Default 150 * max_tracers
        sample_size2 = None,            # Default 30 * max_tracers
        overlap2 = None,                # Default sample_size2 - 1
        max_distance = 0.15,        # mm
        k1 = [0.05, 0.8],           # non-dimensional, important
        iterations = 5              # number of points between k1[0] and k1[1]
    ):

        # Check `lors` is an instance of `pept.LineData`; this includes any
        # subclass! (e.g. `pept.scanners.ParallelScreens`). If it's not, raise
        # an error.
        if not isinstance(lors, pept.LineData):
            raise TypeError(
                textwrap.fill((
                    "\n[ERROR]: `lors` should be an instance of `pept."
                    f"LineData` (or subclass thereof). Received {type(lors)}."
                    "Note: you can read in LoR data from a CSV file using "
                    "`pept.utilities.read_csv`; initialise them in a `pept."
                    "LineData` before calling this function. Check this "
                    "script's documentation for more.\n"
                ), replace_whitespace = False)
            )

        # If any of sample_size, overlap, sample_size2, overlap2 were left
        # to the default `None`, set them to the default values.
        if sample_size is None:
            sample_size = 200 * max_tracers
        if overlap is None:
            overlap = 150 * max_tracers
        if sample_size2 is None:
            sample_size2 = 30 * max_tracers
        if overlap2 is None:
            overlap2 = sample_size2 - 1

        lors.sample_size = sample_size
        lors.overlap = overlap

        cutpoints = peptml.Cutpoints(lors, max_distance)

        ks = np.linspace(k1[0], k1[1], iterations, dtype = float)
        clusterers = [peptml.HDBSCANClusterer(
            int(k * cutpoints.sample_size / max_tracers)
        ) for k in ks]

        centres = []
        labelled_cutpoints = []
        for clusterer in tqdm(clusterers):
            centres1, labelled_cutpoints1 = clusterer.fit(
                cutpoints,
                get_labels = True,
                verbose = False
            )

            centres.append(centres1)
            labelled_cutpoints.append(labelled_cutpoints1)

        grapher = PlotlyGrapher(
            rows = 2,
            cols = iterations,
            subplot_titles = [f"k1 = {k}, labelled cutpoints" for k in ks] +
                             [f"k1 = {k}, cluster centres" for k in ks]
        )

        for i in range(iterations):
            grapher.add_trace(
                labelled_cutpoints[i].points_trace(), row = 1, col = i + 1
            )
            grapher.add_trace(
                centres[i].points_trace(), row = 2, col = i + 1
            )

        grapher.show()


    def typecheck_and_create_attributes(
        self,
        lors,                       # Essential, pept.LineData
        max_tracers,                # Essential, int
        sample_size = None,             # Default 200 * max_tracers
        overlap = None,                 # Default 150 * max_tracers
        max_distance = 0.15,        # mm
        k1 = [0.05, 0.8],           # non-dimensional, important
        iterations = 5              # number of points between k1[0] and k1[1]
    ):
        # Check `lors` is an instance of `pept.LineData`; this includes any
        # subclass! (e.g. `pept.scanners.ParallelScreens`). If it's not, raise
        # an error.
        if not isinstance(lors, pept.LineData):
            raise TypeError(
                textwrap.fill((
                    "\n[ERROR]: `lors` should be an instance of `pept."
                    f"LineData` (or subclass thereof). Received {type(lors)}."
                    "Note: you can read in LoR data from a CSV file using "
                    "`pept.utilities.read_csv`; initialise them in a `pept."
                    "LineData` before calling this function. Check this "
                    "script's documentation for more.\n"
                ), replace_whitespace = False)
            )

        # Attach lors to our class instance so we can access it as a class
        # property. Attach the next variables used too.
        self.lors = lors

        # Make sure max_tracers is an `int` that's larger or equal to 1
        max_tracers = int(max_tracers)
        assert max_tracers >= 1, textwrap.fill(
            "[ERROR]: `max_tracers` should be an `int` >= 1."
        )
        self.max_tracers = max_tracers

        # If any of sample_size, overlap, sample_size2, overlap2 were left
        # to the default `None`, set them to the default values. Otherwise
        # type-check them; also set lors.sample_size and lors.overlap.
        sample_size = 200 * max_tracers if sample_size is None \
                                        else int(sample_size)
        self.sample_size = sample_size

        overlap = 150 * max_tracers if overlap is None \
                                    else int(overlap)
        self.overlap = overlap

        # Type-check max_distance (for cutpoints), k1 (for 1-pass clusterer)
        # and k2 (for the 2-pass clusterer)
        self.max_distance = float(max_distance)

        # We will use `iterations` points between k1[0] and k1[1]. Make sure
        # k1 is list-like with length 2.
        k1 = list(k1)
        assert len(k1) == 2, textwrap.fill(
            "[ERROR]: k1 should be list-like with exactly two elements."
        )
        self.k1 = k1

        self.iterations = int(iterations)


