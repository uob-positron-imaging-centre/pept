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


Birmingham Method GPU recipe
------------------------

The Birmingham Method can also be run on the GPU. This is done using CUDA and requires a GPU with compute capability 3.0 or higher.

For installation please see [pyCUDA](https://pypi.org/project/pycuda/) on PyPi.

::

    import pept
    from pept.tracking import *

    pipeline = pept.Pipeline([
        BirminghamMethodGPU(fopt = 0.5),
    ])

    locations = pipeline.fit(lors)


To manage memory usage, the Birmingham Method GPU will run in batches. The batch size is determined by the memory available in GPU.

Using `memory_usage` you can set the percentage of memory used by the Birmingham Method GPU. The default is 1.0 (100%), which will use (nearly) all available memory.

If you Nvidia GPU is older (compute capability < 2.0), you need to reduce the number of threads per block. This can be done by setting `threads_per_block` to a lower value. The default is 1024.

::

    import pept
    from pept.tracking import *

    pipeline = pept.Pipeline([
        BirminghamMethodGPU(fopt = 0.5, memory_usage = 0.5, threads_per_block = 512),
    ])

    locations = pipeline.fit(lors)