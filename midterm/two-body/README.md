Basic usage: 

```sh
python orbits.py CelestialSeer --save_hdf5
```

To show the evolution:

```sh
python orbits.py CelestialSeer \
  --save_hdf5 \
  --gif_name "orbit.gif" \
  --frames 100
```

Full options:

```sh
python orbits.py CelestialSeer \
  --M 5e6 \
  --a 1.0 \
  --e 0.0167 \
  --N 2 \
  --relativistic True \
  --method "RK3" \
  --steps 2000 \
  --save_hdf5 \
  --gif_name "orbit.gif" \
  --frames 100

```
Running the above will output:
```sh
=========================================================

Initializing CelestialSeer (integrator class):
    Black hole mass:        5.0e+06 [M_sun]
    Semi-major axis (a):    1.0 [AU]
    Eccentricity (e):       0.017

    Integrating equations of motion:
        Periods (N):        20.0
        Method:             RK3
        Steps:              2000
        Relativistic:       True
        Save HDF5:          True

Saved to outputfolder/M5.0e+06-a1.0-e0.017-relat-RK3.h5
----------------------------------------------------------
                  STARTING GIF CREATION
----------------------------------------------------------
Saving 30 frames...
----------------------------------------------------------
Progress: 0% (0 images)
Progress: 10% (3 images)
Progress: 20% (6 images)
Progress: 30% (9 images)
Progress: 40% (12 images)
Progress: 50% (15 images)
Progress: 60% (18 images)
Progress: 70% (21 images)
Progress: 80% (24 images)
Progress: 90% (27 images)
----------------------------------------------------------
All 30 images saved to outputfolder/images.
----------------------------------------------------------
Saving GIF...
----------------------------------------------------------
GIF saved to outputfolder/orbit.gif.
----------------------------------------------------------
All images have been deleted.
----------------------------------------------------------
                   END OF GIF CREATION
----------------------------------------------------------

DONE. Simulated orbit up to 8.9e-03 yr.

==========================================================
```