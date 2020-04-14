# EGRSS

[![Build Status](https://travis-ci.org/martinandersen/EGRSS.svg?branch=master)](https://travis-ci.org/martinandersen/EGRSS)

This project provides implementations of a set of algorithms for computations that involve higher-order *extended generator representable semiseparable* (EGRSS) matrices such as kernel matrices generated by the spline kernel. The algorithms and their application to smoothing spline regression are described in the following paper:

> Martin S. Andersen & Tianshi Chen, “Smoothing Splines and Rank
> Structured Matrices: Revisiting the Spline Kernel,” SIAM Journal on
> Matrix Analysis and Applications, vol. 42, no. 2,
> pp. 389–412, 2020.
> DOI: [10.1137/19M1267349](http://dx.doi.org/10.1137/19M1267349)

The project currently provides the following:

- C implementation/library (double precision)
- Julia reference implementation
- MATLAB reference implementation
- MATLAB MEX interface to C implementation

## Building and testing

Cmake 3.13.5 or later:

```
$ cmake -S . -B build 
$ cmake --build build
$ cd build && ctest .
```

Earlier versions of Cmake:

```
$ mkdir build 
$ cd build && cmake .. && cmake --build .
$ ctest .
```

## Building the MATLAB MEX interface

```
$ cmake -S . -B build -DBUILD_MEX=ON
$ cmake --build build
```

## License

This project is licensed under the [BSD 2-Clause](LICENSE) license.
