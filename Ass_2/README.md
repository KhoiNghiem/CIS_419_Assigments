# CIS_416_Assigment2
## Step by step tutorial for running assignment 2
### Clone repo: [**This repo**](https://github.com/KhoiNghiem/CIS_416_Assigments.git)

```bash
cd CIS_416_Assigments/Ass2
```

### 1. Polynomial Regression
#### 1.1. Implementing Polynomial Regression

```bash
python3 test_polyreg_univariate.py
```
to test your implementation, which will plot the learned function. 

### 1.2. Bias-Variance Tradeoff through Learning Curves 

```bash
python3 test_polyreg_learningCurve.py
```
script to plot the learning curves for various values of λ and d

### 2. Logistic Regression
#### 2.1. Testing Implementation

```bash
python3 test_logreg1.py 
```
trains a logistic regression model using your implementation 

### 3. Support Vector Machines
Implement various kernels for the support vector machine (SVM)
#### 3.1. Getting Started
```bash
python3 example_svm.py
```
Run `example svm.py` with **C = 0.01 and C = 1000**

#### 3.2. Implementing Custom Kernels
```bash
python3 example svmCustomKernel.py
```

#### 3.3. Implementing the Polynomial Kernel
Complete the `myPolynomialKernel()` function in `svmKernels.py`
```bash
python3 test_svmPolyKernel.py
```
