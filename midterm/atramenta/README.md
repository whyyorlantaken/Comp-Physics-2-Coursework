```sh

 █████╗ ████████╗██████╗  █████╗ ███╗   ███╗███████╗███╗   ██╗████████╗ █████╗ 
██╔══██╗╚══██╔══╝██╔══██╗██╔══██╗████╗ ████║██╔════╝████╗  ██║╚══██╔══╝██╔══██╗
███████║   ██║   ██████╔╝███████║██╔████╔██║█████╗  ██╔██╗ ██║   ██║   ███████║
██╔══██║   ██║   ██╔══██╗██╔══██║██║╚██╔╝██║██╔══╝  ██║╚██╗██║   ██║   ██╔══██║
██║  ██║   ██║   ██║  ██║██║  ██║██║ ╚═╝ ██║███████╗██║ ╚████║   ██║   ██║  ██║
╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝
                                                                            

         /\                                                       
     _  / |       /                                      /        
    (  /  |  .---/---).--..-.  .  .-. .-.   .-..  .-.---/---.-.   
     `/.__|_.'  /   /    (  |   )/   )   )./.-'_)/   ) /   (  |   
 .:' /    |    /   /      `-'-''/   /   ( (__.''/   ( /     `-'-' 
(__.'     `-'                            `-'         `-           

                                                                        

```
**Atramenta** is our python package for simulating orbits of celestial bodies around a black hole with both *classical* and *relativistic* approaches. The implementation has been  divided into three main classes: `OrbitBirther` for initializing the system, `CelestialSeer` for solving the equations of motion, and `ChronoPainter` for creating animations of such orbits.

## Instructions

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
  --N 20 \
  --relativistic \
  --method "RK3" \
  --steps 2000 \
  --save_hdf5 \
  --gif_name "orbit.gif" \
  --frames 30

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

```sh
python orbits.py ChronoPainter \
  --orbits outputfolder/M3.0e+06-a1.0-e0.017-relat-RK3.h5 outputfolder/M3.0e+06-a1.0-e0.017-class-RK3.h5 \
  --labels "Relat" "Class" \
  --colors "magenta" "khaki" \
  --gif_name "orbit.gif" \
  --duration 0.1 \
  --frames 100 \
  --dpi 120
```
Running the above will output:
```sh
=========================================================

Initializing ChronoPainter (animation class):
    Orbits: 
        outputfolder/M3.0e+06-a1.0-e0.017-relat-RK3.h5
        outputfolder/M3.0e+06-a1.0-e0.017-class-RK3.h5
    Labels: Relat, Class
    Colors: magenta, khaki

    Creating animation:
        GIF name:   orbit.gif
        Duration:   0.1
        Frames:     100
        DPI:        120

----------------------------------------------------------
                  STARTING GIF CREATION
----------------------------------------------------------
Saving 100 frames...
----------------------------------------------------------
Progress: 0% (0 images)
Progress: 10% (10 images)
Progress: 20% (20 images)
Progress: 30% (30 images)
Progress: 40% (40 images)
Progress: 50% (50 images)
Progress: 60% (60 images)
Progress: 70% (70 images)
Progress: 80% (80 images)
Progress: 90% (90 images)
----------------------------------------------------------
All 100 frames saved to outputfolder/frames.
----------------------------------------------------------
Making GIF...
----------------------------------------------------------
GIF saved to outputfolder/orbit.gif.
----------------------------------------------------------
All frames have been deleted.
----------------------------------------------------------
                   END OF GIF CREATION
----------------------------------------------------------

DONE. Animation created successfully.

==========================================================
```