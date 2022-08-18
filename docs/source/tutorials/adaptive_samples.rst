Adaptive Sampling
=================

Perhaps the most important decision a PEPT user must make is how the LoRs are divided into samples. The two most common approaches are:

**Fixed sample size**: a constant number of elements per sample, with potential overlap between samples.

- Advantages: effectively adapts spatio-temporal resolution, with higher accuracy in more active PEPT scanner regions.
- Disadvantages: when a tracer exits the field of view, the last LoRs will be joined with the first LoRs when the tracer re-enters the scanner in the same samples.

**Fixed time window**: a constant time interval in which LoRs are aggregated, with potential overlap.

- Advantages: robust to tracers moving out of the field of view.
- Disadvantages: non-adaptive temporal resolution.

The two approaches can be combined into a single ``pept.AdaptiveWindow``, which works as a fixed time window, except when more LoRs are encountered than a given limit, in which case the time window is shrunk - hence adapting the time window depending on how many LoRs are intercepted in a given window.


::

    import pept

    # A time window of 5 ms shrinking when encountering more than 200 LoRs
    lors = pept.LineData(..., sample_size = pept.AdaptiveWindow(5.0, 200))

    # A time window of 12 ms with the number of LoRs capped at 400 LoRs and an overlap of 6 ms
    lors = pept.scanners.adac_forte(
        ...,
        sample_size = pept.AdaptiveWindow(12., 200),
        overlap = pept.AdaptiveWindow(6.),
    )



Moreover, if an ideal number of LoRs is selected, there exists an optimum time window for which most samples will have roughly this ideal number of LoRs, except when the tracer is out of the field of view, or it's static. This can be automatically selected using ``pept.tracking.OptimizeWindow``:


::

    import pept
    import pept.tracking as pt

    # Find an adaptive time window that is ideal for about 200 LoRs per sample
    lors = pept.LineData(...)
    lors = pt.OptimizeWindow(ideal_elems = 200).fit(lors)


`OptimizeWindow` can be used at the start of a pipeline; an optional `overlap` parameter can be used to define an overlap as a ratio to the ideal time window found. For example, if the ideal time window found is 100 ms, an overlap of 0.5 will result in an overlapping time interval of 50 ms:

::

    import pept
    from pept.tracking import *

    pipeline = pept.Pipeline([
        OptimizeWindow(200),
        BirminghamMethod(fopt = 0.5),
        Stack(),
    ])

    locations = pipeline.fit(lors)

