# Portfolio_Analysis
Analysis of portfolios as a multiple objective problem using Cauchy's steepest descent coupled with exterior penalty method. The file contains the following functions

## goldstein_armijo_line_search
The Goldstein-Armijo line search computes the optimum value for lambda

### Parameters
- f - objective function
- dims - list of variables in function
- start - starting point

## cauchy_steepest_descent
Computes the minimum of the function using lambda obtained from Goldstein-Armijo criteria

### Parameters
- fnc - objective function
- con - Constraint
- dims - list of variables in function
- start - starting point
- cw - Constraint weight

## ext_pen_method
Computes the minimum of the function by running cauchy_steepest_descent for increased penalty

### Parameters
- f - objective function 1
- f1 - objective function 2
- con - Constraint
- dims - list of variables in function

## portfolio
This function reads the excel file and constructs the objective function for the given portfolio analysis problem and plots the *"return vs volatility"* plot for the optimum solution. 
Here in this specific file 4 portfolios are analysed, this code can be customised for any number of stock returns.

### Parameters
- excel_name - Excel sheet containing the return values of the portfolios
- x - array containing the number of each stock bought ([a,b,c,d])
- con - Constraint
- dims - list of variables in function

## Dependency libraries
- sympy
- pandas
- numpy
- scipy
- matplotlib




