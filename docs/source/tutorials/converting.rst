Initialising PEPT Scanner Data
==============================

The ``pept.scanners`` submodule contains converters between scanner specific data formats
(e.g. parallel screens / ASCII, modular camera / binary) and the ``pept`` base classes,
allowing simple initialisation of ``pept.LineData`` from different sources.


ADAC Forte
----------

The parallel screens detector used at Birmingham can output binary `list-mode` data, which can
be converted using ``pept.scanners.adac_forte(binary_file)``:

::

    import pept

    lines = pept.scanners.adac_forte("binary_file.da01")


If you have multiple files from the same experiment, e.g. "data.da01", "data.da02", etc., you can stitch them all together using a *glob*, "data.da*":

::

    import pept

    # Multiple files starting with `binary_file.da`
    lines = pept.scanners.adac_forte("binary_file.da*")



Parallel Screens
----------------

If you have your data as a CSV containing 5 columns `[t, x1, y1, x2, y2]` representing the
coordinates of the two points defining an LoR on two parallel screens, you can use
``pept.scanners.parallel_screens``  to insert the missing coordinates and get the LoRs into
the general ``LineData`` format `[t, x1, y1, z1, x2, y2, z2]`:

::

    import pept

    screen_separation = 500
    lines = pept.scanners.parallel_screens(csv_or_array, screen_separation)


Modular Camera
--------------

Your modular camera data can be initialised using ``pept.scanners.modular_camera``:

::

    import pept

    lines = pept.scanners.modular_camera(filepath)



