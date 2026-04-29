# Curve Fitting (Exercise 12.11)

Implementation of **Exercise 12.11 – Curve Fitting** from the book:

**H. Paul Williams – *Model Building in Mathematical Programming***

The program fits models to a dataset using **IBM CPLEX**.

## Models

Two approximations are computed:

* **Linear:** (y = bx + a)
* **Quadratic:** (y = cx^2 + bx + a)

Each model is solved with two objectives:

* minimize **sum of absolute deviations**
* minimize **maximum deviation**

Absolute values are linearized and solved as **linear programming problems in CPLEX**.

## Build with CMake

Requirements:

* **C++ compiler**
* **IBM ILOG CPLEX Optimization Studio**
* **CMake**

Before building, update the CPLEX path in CMakeLists.txt if needed:

```cmake
set(CPLEX_ROOT_PATH "C:/Program Files/IBM/ILOG/CPLEX_Studio_Community2212")
```

Build:

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

Run:

```bash
./CurveFittingCPLEX
```
