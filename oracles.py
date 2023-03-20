import numpy as np
import scipy

class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """
    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')
    
    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))


class QuadraticOracle(BaseSmoothOracle):
    """
    Oracle for quadratic function:
       func(x) = 1/2 x^TAx - b^Tx.
    """
    
    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        # your code here
        return np.dot(np.dot(x.T, self.A), x) * 1/2 - np.dot(self.b.T, x)

    def grad(self, x):
        # your code here
        return np.dot(self.A, x) - self.b.T

        
class LogRegL2Oracle(BaseSmoothOracle):
    """
    Oracle for logistic regression with l2 regularization:
         func(x) = 1/m sum_i log(1 + exp(-b_i * a_i^T x)) + regcoef / 2 ||x||_2^2.
    Let A and b be parameters of the logistic regression (feature matrix
    and labels vector respectively).
    For user-friendly interface use create_log_reg_oracle()
    Parameters
    ----------
        matvec_Ax : function
            Computes matrix-vector product Ax, where x is a vector of size n.
        matvec_ATx : function of x
            Computes matrix-vector product A^Tx, where x is a vector of size m.
        matmat_ATsA : function
            Computes matrix-matrix-matrix product A^T * Diag(s) * A,
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b
        self.regcoef = regcoef

    def func(self, x):
        # your code here
        m = self.b.shape[0]
        power = self.matvec_Ax(x) * self.b
        return (1/m) * np.sum(np.logaddexp(np.zeros(m), -power)) + (1/2) * self.regcoef * np.dot(x, x)
    
    def grad(self, x):
        # your code here
        m = self.b.shape[0]
        power = self.matvec_Ax(x) * self.b
        sigmoid_b = self.b * scipy.special.expit(-power)
        return (-1/m) * self.matvec_ATx(sigmoid_b) + self.regcoef * x
    

def create_log_reg_oracle(A, b, regcoef):
    """
    Auxiliary function for creating logistic regression oracles.
        `oracle_type` must be either 'usual' or 'optimized'
    """
    matvec_Ax = lambda x: np.dot(A, x) if type(A) == np.ndarray else A * x # your code here
    matvec_ATx = lambda x: np.dot(A.T, x) if type(A) == np.ndarray else A.T * x  # your code here

    def matmat_ATsA(s):
        # your code here
        if type(A) == np.ndarray:
            ATs = np.dot(A.T, np.diag(s))
            ATsA = np.dot(ATs, A)
        else:
            ATsA = A.T * np.diag(s) * A
        return ATsA #None

    return LogRegL2Oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)