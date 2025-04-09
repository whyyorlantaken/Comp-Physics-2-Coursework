Basic usage: 

```sh
python orbits.py CelestialSeer --save_hdf5
```

To show the evolution:

```sh
python orbits.py CelestialSeer \
  --save_hdf5 \
  --gif_name "orbit.gif" \
  --frames 100 \
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
  --frames 100 \
```