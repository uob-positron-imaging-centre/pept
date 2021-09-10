Feature Point Identification
============================

FPI is a modern voxel-based tracer-location algorithm that can reliably work with unknown numbers of tracers in fast and noisy environments.

It was successfully used to track fast-moving radioactive tracers in pipe flows at the Virginia Commonwealth University. If you use this algorithm in your work, please cite the following paper:

    *Wiggins C, Santos R, Ruggles A. A feature point identification method for positron emission particle tracking with multiple tracers. Nuclear Instruments and Methods in Physics Research Section A: Accelerators, Spectrometers, Detectors and Associated Equipment. 2017 Jan 21; 843:22-8.*




FPI Recipe
----------

As FPI works on voxelized representations of the LoRs, the ``Voxelize`` filter is first used before ``FPI`` itself:

::

    import pept
    from pept.tracking import *

    resolution = (100, 100, 100)

    pipeline = pept.Pipeline([
        Voxelize(resolution),
        FPI(w = 3, r = 0.4),
        Stack(),
    ])

    locations = pipeline.fit(lors)


