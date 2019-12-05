# EGRSS

This project provides implementations of a set of algorithms for computations that involve higher-order *extended generator representable semiseparable* (EGRSS) matrices such as kernel matrices generated by the spline kernel. The algorithms and their application to smoothing spline regression are described in the following paper:

> Martin S. Andersen & Tianshi Chen, "Smoothing Splines and Rank Structured Matrices: Revisiting the Spline Kernel", submitted to SIAM Journal on Matrix Analysis and Applications, June, 2019. 

The project currently provides the following:

- C implementation/library (double precision)
- Julia reference implementation
- MATLAB reference implementation
- MATLAB MEX interface to C implementation

## Building and testing

```
$ cmake -S . -B build 
$ cmake --build build
```

## Building the MATLAB MEX interface

```
$ cmake -S . -B build -DBUILD_MEX=ON
$ cmake --build build
```

## License

This project is licensed under the [BSD 2-Clause](LICENSE) license.
