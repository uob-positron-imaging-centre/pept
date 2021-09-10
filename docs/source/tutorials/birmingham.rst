The Birmingham Method
=====================

The Birmingham Method is an efficient, analytical technique for tracking tracers using the LoRs from PEPT data.

If you are using it in your research, you are kindly asked to cite the following paper:


    *Parker DJ, Broadbent CJ, Fowles P, Hawkesworth MR, McNeil P. Positron emission particle tracking-a technique for studying flow within engineering equipment. Nuclear Instruments and Methods in Physics Research Section A: Accelerators, Spectrometers, Detectors and Associated Equipment. 1993 Mar 10;326(3):592-607.*



Birmingham Method recipe
------------------------

::

    import pept
    from pept.tracking import *

    pipeline = pept.Pipeline([
        BirminghamMethod(fopt = 0.5),
        Stack(),
    ])

    locations = pipeline.fit(lors)



Recipe with Trajectory Separation
---------------------------------

::

    import pept
    from pept.tracking import *

    pipeline = pept.Pipeline([
        BirminghamMethod(fopt = 0.5),
        Segregate(window = 20, cut_distance = 10),
        Stack(),
    ])

    locations = pipeline.fit(lors)


