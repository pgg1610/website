---
date: "2020-04-26"
description: Minimal example explaining how bayesian optimization works 
pubtype: Jupyter Notebook
fact: 
featured: true
image: /img/BO.png
link: https://github.com/pgg1610/misc_notebooks/blob/master/Bayesian_optimisation/bayesian_optimisation.ipynb
tags:
- Python
- Optimization

title: Bayesian Optimization using Gaussian Processes
---
* Notebook explaining the idea behind bayesian optimization alongside a small example showing its use. This notebook was adapted from Martin Krasser's [blogpost](http://krasserm.github.io/2018/03/21/bayesian-optimization/)
* Good introductory write-up on Bayesian optimization [here](https://distill.pub/2020/bayesian-optimization/)
* Nice lecture explaining the working of Gaussian Processes [here](https://www.youtube.com/watch?v=92-98SYOdlY&t=4827s)

## Setup 

If `f` (objective function) is cheap to evaluate we can sample various points and built a potential surface however, if the `f` is expensive -- like in case of first-principles electronic structure calculations, it is important to minimize the number of `f` calls and number of samples drawn from this evaluation. In that case, if an exact functional form for `f` is not available (that is, f behaves as a “black box”), what can we do? 

Bayesian optimization proceeds by maintaining a probabilistic belief about f and designing a so called **_acquisition function_** to determine where to evaluate the function next. Bayesian optimization is particularly well-suited to global optimization problems where `f` is an expensive black-box function. The idea is the find "global" minimum with least number of steps. Incorporating prior beliefs about the underlying process and update the prior with samples draw from the model to better estimate the posterior. Model used for approximating the objective function is called the **_surrogate model_**. 

### Surrogate model 

A popular surrogate model applied for Bayesian optimization, although strictly not required, are Gaussian Processes (GPs). These are used to define a prior beliefs about the objective function. The GP posterior is cheap to evaluate and is used to propose points in the search space where sampling is likely to yield an improvement. Herein, we could substitute this for a ANNs or other surrogate models. 

### Acquisition functions 

Used to propose sampling points in the search space. Trade-off between exploitation vs exploration. Exploitation == sampling where objective function value is high; exploration == where uncertainty is high. Both correspond to high `acquisition function` value. The goal is the maximize the acquisition value to determine next sampling point. 

Popular acquisition functions: 

* Maximum probability of improvement    
* Expected improvement
* Lower/Upper confidence bound (UCB)

*1. Expected Improvement*

```python
def EI(X_new, gpr, delta, noisy, minimize_objective):
    """
    Compute the expected improvement at points X_new, from a Gaussian
    process surrogate model fit to observed data (X_sample, Y_sample).
            
    Arguments
    ---------
    X_new : array_like; shape (num_new_pts, input_dimension)
    Locations at which to compute expected improvement.
                
    gpr : GaussianProcessRegressor
    Regressor object, pre-fit to the sample data via the command
    gpr.fit(X_sample, Y_sample).
                
    delta : float
    Trade-off parameter for exploration vs. exploitation. Must be
    a non-negative value. A value of zero corresponds to pure ex-
    ploitation, with more exploration at larger values of delta.
                
    noisy : bool
    If True, assumes a noisy model and predicts the expected
    outputs at X_sample, rather than using Y_sample.
                
    minimize_objective : bool
    Designates whether the objective function is to be minimized
    or maximized. By default, minimization is assumed. In either
    case, the expected improvement is defined such that its value            
    should be maximized.
            
    Returns
    -------
    ei : np.ndarray; shape (num_points,)
    The expected improvement at each of the points in X_new.
    """
    if delta < 0.0:
        raise ValueError("Exploration parameter must be non-negative.")

    if minimize_objective:
        best = np.min
        sign = -1.0
    else:
        best = np.max
        sign = 1.0
            
    (mu, sigma) = gpr.predict(X_new, return_std = True)
    
    if (mu.ndim > 1 and mu.shape[1] > 1) or mu.ndim > 2:
        raise RuntimeError("Invalid shape for predicted "
                                   "mean: %s" % (mu.shape,))
    else:
        mu = mu.flatten()

    sigma = np.maximum(1e-15, sigma.flatten())
    # Bump small variances to prevent divide-by-zero.
            
    if noisy:
        mu_sample = gpr.predict(gpr.X_train_)
        best_y = best(mu_sample)
    else:
        best_y = best(gpr.y_train_)
            
    improvement = sign*(mu - best_y + delta)
    Z = improvement/sigma
    return improvement*stats.norm.cdf(Z) + sigma*stats.norm.pdf(Z)
```

*2. Lower Confidence Bound*

```python
def LCB(X_new, gpr, sigma):
    """
    Compute the lower confidence bound at points X_new, from a Gaussian
    process surrogate model fit to observed data (X_sample, Y_sample).
            
    Arguments
    ---------
    X_new : array_like; shape (num_new_pts, input_dimension)
        Locations at which to compute confidence bound.
                
    gpr : GaussianProcessRegressor
        Regressor object, pre-fit to the sample data via the command
        gpr.fit(X_sample, Y_sample).
                
    sigma : float
        Trade-off parameter for exploration vs. exploitation. Must be
        a non-negative value. A value of zero corresponds to pure exploitation, with more                   exploration at larger values of sigma.
            
    Returns
    -------
    lcb : np.ndarray; shape (num_points,)
    The lower confidence bound at each of the points in X_new.
    """
    if sigma < 0.0:
        raise ValueError("Exploration parameter must be non-negative.")
            
    (mean, std_dev) = gpr.predict(X_new, return_std = True)
    
    if (mean.ndim > 1 and mean.shape[1] > 1) or mean.ndim > 2:
        raise RuntimeError("Invalid shape for predicted "
                            "mean: %s" % (mean.shape,))
    else:
        mean = mean.flatten()
            
    return mean - sigma*std_dev
```

### Objective function

Objective function `f` we are interested in optimizing is the `Egg Carton` function which has quite peculiar shape, as seen in the schematic below. While there are local 'swiggles' the overall function tends to a lower value around x = (4,6). We want to see if bayesian optimization can find this minimum value by optimizing not the ground function but rather a surrogate function which hypothetically would be 'cheaper' to evaluate and optimize on. 

The plot shown below has two main things: 1. The ground truth function which is the Egg carton function (shown by the black line) 2. The randomly sampled points which have some error built into them. Think of this like a sampling of surface with some error built-into the measuring the device, so it wont accurate sample the ground-truth function. We will use this 'noisy' function for optimization. 

Plot for the objective function: 
```python 
def egg_carton(x, f_noise = 0.0):
    x = np.asarray(x)
    return np.sin(4.25*x) + 0.25*(x - 4.8)**2.0 + f_noise * np.random.randn(*x.shape) 
```

Initial points are sampled from numpy's random number in a uniform distribution:
```python
num_sample_points = 10
noise_ = 0.1
generator = np.random.default_rng(42)
x_sample = generator.uniform(low, high, size = (num_sample_points, 1))
y_sample = objective(x_sample, noise_)
```

![objective_function](/img/bo/BO_New/test_function.png)

### Bayesian optimization

* Fit a surrogate function on initial points 

![initial_fit](/img/bo/BO_New/initial_gpr_eval.png)

Bayesian optimization runs for few iterations. 

For the inital points and the function value a GPR model as implemented in the `sklearn.gaussian_process.GaussianProcessRegressor` module is used. The prediction from the GPR is then used to optimize the acquisition function -- Expected Improvement Criterion or Lower Confidence Bound. 

#### Running a few more iterations: 

![iter_final](/img/bo/BO_New/final_iterations.png)

In total the noisy estimation of the ground-truth is conducted on 30 additional points. It is evident from the plot that most of those points are near the x = (4,6) since that is the minimum value region for the function.  