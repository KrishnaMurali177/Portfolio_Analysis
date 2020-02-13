import math
import numpy as np
import pandas as pd
import numpy.linalg as la
from sympy import *
from scipy.stats.mstats import gmean
import matplotlib.pyplot as plt                             # Required imports

def goldstein_armijo_line_search(f, dims, st):
    '''The parameters to the goldstein_armijo function to compute lambda_k are
    f - objective function
    dims - list of variables in function
    start - starting point;
    Returns coeffecient of descent lambda_k'''
    fn = Matrix([f])
    f = lambdify([dims], fn, "numpy")                   # Convert f to a lambda function for easy math processing
    initial_start = Matrix(st)
    start = initial_start
    grad = fn.jacobian([dims]).transpose()
    g = lambdify([dims], grad, "numpy")                 # Convert grad to a lambda function for easy math processing
    max_iter = 50                                       # Setting the maximum iterations to 50
    alpha = math.exp(-4)                                # Setting the alpha value to e-4
    lambda_k = 0.5                                      # Setting the initial lambda value to 1/2
    eta = 0.9                                           # Setting the eta value which is used in the second condition to 0.9
    dk = -1 * grad                                      # direction of descent = -grad(xk)
    d = lambdify([dims], dk, "numpy")                   # Convert dk to a lambda function for easy math processing
    l_iter = 1
    flag = False
    for i in range(1, max_iter):
        # Goldstein-Armijo Condition #1 - f(xk + lambda*dk) <= f(xk) + alpha*lambda*dk'*grad(xk)
        rhs_1 = np.squeeze(f(st)) + alpha * np.power(lambda_k, l_iter) * d(st).reshape(len(dims),1).transpose().dot(g(st).reshape(len(dims),1))
        next_pt = st + np.power(lambda_k, l_iter) * d(st).reshape(len(dims),1)    #Set next point to xk + lambda_k*dk
        next_pt_array = np.array(next_pt.tolist()).astype(np.float64)
        lhs_1 = f(next_pt_array)
        if lhs_1 <= rhs_1:
            # If Condition 1 is satisfied check Condition 2:
            # dk'*grad(xk+1) >= eta*dk'*grad(xk)
            rhs_2 = eta * d(st).reshape(len(dims),1).transpose().dot(g(st).reshape(len(dims),1))
            lhs_2 = d(st).reshape(len(dims),1).transpose().dot(g(next_pt_array).reshape(len(dims),1))
            if lhs_2 >= rhs_2:
                lambda_k = np.power(lambda_k, l_iter)   # If both the conditions are satisfied finalize the lambda value
                flag = True
                break
        else:
            l_iter += 1                                 # If lambda value does not satisfy the 1st condition, trying with 1/2^i in the next iteration
    if flag:
        return lambda_k
    else: return np.power(lambda_k, l_iter)

def cauchy_steepest_descent(fnc,con, dims, start,cw):
    '''The parameters to the cauchy function to compute minimum are
    fnc - objective function
    con - Constraint
    dims - list of variables in function
    start - starting point
    cw - Constraint weight;
    Returns minimum point'''
    x_o = start
    fnc_con = lambdify([dims], con, "numpy")
    const = 0
    a, b, c, d = Symbol('a'), Symbol('b'), Symbol('c'), Symbol('d')
    if fnc_con(x_o) > 0 : const = con**2
    if start[0].round(2) < 0: const+= a**2                               # If the constraints are violated, Multiply square term with the weight
    if start[1].round(2) < 0: const+= b**2                               # Quadratic loss function if constraint is violated
    if start[2].round(2) < 0: const+= c**2
    if start[3].round(2) < 0: const+= d**2
    fn = Matrix([fnc + cw * (const)])                  # Multiply the constraint weight by con^2 as it is an equality constraint
    f = lambdify([dims], fn, "numpy")                           # Convert f to a lambda function for easy numpy processing
    start_array = np.array(start).reshape(len(dims),1)
    grad = fn.jacobian([dims]).transpose()                      # Compute the gradient
    g = lambdify([dims], grad, "numpy")                         # Convert grad to a lambda function for easy numpy processing
    max_iter = 1
    dk = -1 * grad                                              # direction of descent = -grad(xk)
    d_fnc = lambdify([dims], dk, "numpy")
    while max_iter <= 100:
        if la.norm(g(start_array))/la.norm((1+f(start_array))) >= math.exp(-8):
            lambda_k = goldstein_armijo_line_search(fn, dims, start_array)      #Calling Goldstein armijo function to determine lambda_k
            start_array = start_array.reshape(len(dims),1) + lambda_k * d_fnc(start_array).reshape(len(dims),1)      #Setting the start array
            max_iter+=1
        else: break
    return np.array(start_array).reshape(len(dims),1)

def ext_pen_method(f,f1,con,dims):
    '''The parameters to the exterior penalty method function to compute minimum are
        f - objective function 1
        f1 - objective function 2
        con - Constraint
        dims - list of variables in function;
        Returns dictionary containing the alpha value,
        minimum point, value of return, value of volatility'''
    alpha = 0                                                       # Setting initial alpha to 0
    fnc = lambdify([dims], f, "numpy")                              # Lambdifying return function for easy numpy processing
    fnc1 = lambdify([dims], f1, "numpy")                            # Lambdifying volatility function for easy numpy processing
    output_dict = {}
    x_dic = {}
    for alpha in np.arange(0,1.05,.05):                             # Increasing alpha in steps of 0.05 to capture 20 points between 0 and 1
        i = 1                                                       # Setting exponent of the constraint weight as 1 and increasing for each iteration
        cw = 10**i                                                  # Constraint weight = 10^i
        x_o = np.array([0, 1, 0, 0]).reshape(4, 1)                  # Setting the starting point
        fn = (1 - alpha) * f + alpha * f1                           # Framing the objective function on the basis of alpha
        while(cw < 10**6):                                          # Iterating till constraint weight = 10^6
            x_n = cauchy_steepest_descent(fn,con,dims,x_o,cw)       # Calling the cauchy steepest descent method to compute minimum
            x_dic.update({i:x_n})
            i = i+1                                                 # Incrementing i value by 1
            cw = 10**i                                              # Setting the constraint weight to 10^i
            x_o = x_n                                               # Setting the old point as new one
        output_dict.update({alpha: {'point':x_dic[max(x_dic.keys())], 'return': fnc(x_n), 'volatility': fnc1(x_n)}})        # Appending the values of alpha, points, return function and risk function values to the output dictionary
    return output_dict

def portfolio(excel_name,x,con,dims):
    '''The parameters to the exterior penalty method function to compute minimum are
            excel_name - Excel sheet containing the return values of the portfolios
            x - array containing the number of each stock bought ([a,b,c,d])
            con - Constraint
            dims - list of variables in function;
            Returns dictionary containing the alpha value,
            minimum point, value of return, value of volatility and the plot'''
    stocks = pd.read_excel(excel_name, header=0)                                            # Reading the excel sheet
    ret_list = []
    for col in ['A', 'B', 'C', 'D']:
        ret_list.append(stocks[col].apply(lambda x: x + 1))                                 # Adding 1 to each return for geometric mean computation
    A, B, C, D = ret_list[0], ret_list[1], ret_list[2], ret_list[3]
    gm = np.array([gmean(A) - 1, gmean(B) - 1, gmean(C) - 1, gmean(D) - 1]).reshape(4, 1)   # Computing geometric mean
    f = -x.dot(gm)                                                                          # -0.0689083353898281*a - 0.0807760842481569*b - 0.0268752543004709*c - 0.0309861159295426*d
    cov_mat = np.cov(ret_list)                                                              # Computing the covariance matrix
    f1 = 0.5 * x.dot(cov_mat).dot(x.transpose())                                            # Constructing the risk function
    x = ext_pen_method(f, f1, con, dims)                                                    # Calling the exterior penalty method
    return_list = []
    vol_list = []
    for key, value in x.items():
        return_list.append(-1 * np.squeeze(x.get(key).get('return')))                       # Populating the return and risk values for plotting
        vol_list.append(np.squeeze(x.get(key).get('volatility')))
    fig, ax = plt.subplots()
    scatter = ax.scatter(vol_list, return_list, c=list(x.keys()))                           # Plotting the points
    plt.xlabel('Volatility')
    plt.ylabel('Returns')
    plt.title('Portfolio Analysis')
    ax.plot(vol_list, return_list)
    result = [(i,np.squeeze(x.get(i).get('point')[0]),np.squeeze(x.get(i).get('point')[1]),np.squeeze(x.get(i).get('point')[2]),np.squeeze(x.get(i).get('point')[3]),-1*np.squeeze(x.get(i).get('return')), np.squeeze(x.get(i).get('volatility'))) for i,j in x.items()]
    min_table = pd.DataFrame(result, columns = ['Alpha', 'a', 'b', 'c', 'd', 'Returns', 'Volatility'])  # Constructing the final table
    return min_table,plt


a,b,c,d = Symbol('a'), Symbol('b'), Symbol('c'), Symbol('d')            # Setting the symbols
x = np.array([a,b,c,d]).reshape(1,4)                                    # Setting the stock share array
dims = [a,b,c,d]                                                        # Setting the dimensions
con = a+b+c+d-1                                                         # Constructing the constraint function
excel_name = "portfolio.xlsx"                                           # Excel sheet name
solution = portfolio(excel_name,x,con,dims)                             # Calling the portfolio function
print(solution[0].to_string())                                          # Printing the output table
solution[1].show()                                                      # Displaying the graph

