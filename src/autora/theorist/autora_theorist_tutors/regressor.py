import sympy as sp
import random

class SymbolicRegressor:
    def __init__(self, variables, max_depth=3, population_size=1000):
        """
        Initialize the Symbolic Regressor.

        Parameters:
        variables (list): List of variable names.
        max_depth (int): Maximum depth for generated expressions.
        population_size (int): Number of expressions to sample.

        Example:
        >>> regressor = SymbolicRegressor(variables=['x1', 'x2'], max_depth=3, population_size=10)
        >>> isinstance(regressor, SymbolicRegressor)
        True
        """
        self.variables = variables
        self.max_depth = max_depth
        self.population_size = population_size

    def generate_random_expression(self, current_depth=0):
        """
        Generate a random expression in prefix notation.

        Parameters:
        current_depth (int): Current depth of the expression tree.

        Returns:
        list: Random expression in prefix notation.

        Example:
        >>> regressor = SymbolicRegressor(variables=['x1', 'x2'], max_depth=3)
        >>> expr = regressor.generate_random_expression()
        >>> isinstance(expr, list)
        True
        """
        if current_depth < self.max_depth:
            choice = random.choice(['+', '-', '*', '/', 'exp', 'log', 'const', 'var'])
            if choice in ['+', '-', '*', '/']:
                return [choice,
                        self.generate_random_expression(current_depth + 1),
                        self.generate_random_expression(current_depth + 1)]
            elif choice in ['exp', 'log']:
                return [choice, self.generate_random_expression(current_depth + 1)]
            elif choice == 'const':
                return random.uniform(-10, 10)
            else:  # 'var'
                return random.choice(self.variables)
        else:
            return random.choice(self.variables + [random.uniform(-10, 10)])

    def prefix_to_sympy(self, expr):
        """
        Convert a prefix notation expression to a SymPy expression.

        Parameters:
        expr (list or float): Expression in prefix notation or constant value.

        Returns:
        sympy.Basic: SymPy expression.

        Example:
        >>> regressor = SymbolicRegressor(variables=['x1', 'x2'])
        >>> sympy_expr = regressor.prefix_to_sympy(['+', 'x1', 2])
        >>> isinstance(sympy_expr, sp.Basic)
        True
        """
        if isinstance(expr, list):
            if expr[0] == '+':
                return self.prefix_to_sympy(expr[1]) + self.prefix_to_sympy(expr[2])
            elif expr[0] == '-':
                return self.prefix_to_sympy(expr[1]) - self.prefix_to_sympy(expr[2])
            elif expr[0] == '*':
                return self.prefix_to_sympy(expr[1]) * self.prefix_to_sympy(expr[2])
            elif expr[0] == '/':
                return self.prefix_to_sympy(expr[1]) / self.prefix_to_sympy(expr[2])
            elif expr[0] == 'exp':
                return sp.exp(self.prefix_to_sympy(expr[1]))
            elif expr[0] == 'log':
                return sp.log(self.prefix_to_sympy(expr[1]))
        else:
            return sp.sympify(expr)

    def get_depth(self, expr):
        """
        Calculate the depth of a SymPy expression.
        """
        if not expr.args:
            return 0
        return 1 + max(self.get_depth(arg) for arg in expr.args)

    def evaluate_expression(self, expr, data):
        """
        Evaluate a SymPy expression on given data.

        Parameters:
        expr (sympy.Basic): SymPy expression.
        data (dict): Dictionary of variable values.

        Returns:
        np.ndarray: Evaluated values.

        Example:
        >>> regressor = SymbolicRegressor(variables=['x1', 'x2'])
        >>> expr = regressor.prefix_to_sympy(['+', 'x1', 2])
        >>> data = {'x1': np.array([1, 2, 3]), 'x2': np.array([4, 5, 6])}
        >>> result = regressor.evaluate_expression(expr, data)
        >>> np.all(result == np.array([3, 4, 5]))
        True
        """
        f = sp.lambdify(self.variables, expr, 'numpy')
        predictions = f(*[data[var] for var in self.variables])
        return predictions

    def fitness_function(self, expr, x, y):
        """
        Compute the fitness (mean squared error) of an expression.

        Parameters:
        expr (sympy.Basic): SymPy expression.
        x (dict): Input data.
        y (np.ndarray): True output values.

        Returns:
        float: Mean squared error.

        Example:
        >>> regressor = SymbolicRegressor(variables=['x1', 'x2'])
        >>> expr = regressor.prefix_to_sympy(['+', 'x1', 2])
        >>> x = {'x1': np.array([1, 2, 3]), 'x2': np.array([4, 5, 6])}
        >>> y = np.array([3, 4, 5])
        >>> mse = regressor.fitness_function(expr, x, y)
        >>> isinstance(mse, float)
        True
        """
        predictions = self.evaluate_expression(expr, x)
        mse = ((predictions - y) ** 2).mean()
        return mse

    def fit(self, x, y):
        """
        Fit the symbolic regressor to the data.

        Parameters:
        x (dict): Input data.
        y (np.ndarray): True output values.

        Returns:
        self.best_expr (sympy.Basic): Best found expression.

        Example:
        >>> regressor = SymbolicRegressor(variables=['x1', 'x2'], max_depth=3, population_size=10)
        >>> x = {'x1': np.random.rand(100), 'x2': np.random.rand(100)}
        >>> y = 3 * x['x1'] ** 2 + 2 * x['x2'] - 1
        >>> regressor.fit(x, y)
        >>> regressor.best_expr is not None
        True
        """
        self.best_expr = None
        self.best_fitness = float('inf')

        for _ in range(self.population_size):
            expr = self.generate_random_expression()
            sympy_expr = self.prefix_to_sympy(expr)
            fitness = self.fitness_function(sympy_expr, x, y)
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_expr = sympy_expr
        return self.best_expr

    def predict(self, x):
        """
        Predict using the best found expression.

        Parameters:
        x (dict): Input data.

        Returns:
        np.ndarray: Predicted values.

        Example:
        >>> regressor = SymbolicRegressor(variables=['x1', 'x2'], max_depth=3, population_size=10)
        >>> x = {'x1': np.random.rand(100), 'x2': np.random.rand(100)}
        >>> y = 3 * x['x1'] ** 2 + 2 * x['x2'] - 1
        >>> regressor.fit(x, y)
        >>> predictions = regressor.predict(x)
        >>> len(predictions) == len(y)
        True
        """
        return self.evaluate_expression(self.best_expr, x)

    def print_eqn(self):
        """
        Print the best found expression.

        Example:
        >>> regressor = SymbolicRegressor(variables=['x1', 'x2'], max_depth=3, population_size=10)
        >>> x = {'x1': np.random.rand(100), 'x2': np.random.rand(100)}
        >>> y = 3 * x['x1'] ** 2 + 2 * x['x2'] - 1
        >>> regressor.fit(x, y)
        >>> regressor.print_eqn()
        Best Expression: ...
        """
        print(f"Best Expression: {self.best_expr}")