import sympy as sp
import random
import numpy as np
import pymc as pm
import arviz as az
import pytensor.tensor as pt


class SymbolicRegressor:
    def __init__(self, variables, max_depth=3, population_size=1000, seed=None):
        self.variables = variables
        self.max_depth = max_depth
        self.population_size = population_size
        self.seed = seed
        self.symbolic_vars = {var: sp.symbols(var) for var in self.variables}
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def generate_random_expression(self, current_depth=0):
        """
        Generate a random expression in prefix notation.

        Parameters:
        current_depth (int): Current depth of the expression tree.

        Returns:
        list: Random expression in prefix notation.
        
        Example:
        >>> variables = ['x1', 'x2']
        >>> regressor = SymbolicRegressor(variables=variables, max_depth=3, seed=42)
        >>> random_expr = regressor.generate_random_expression()
        Current depth: 0, Choice: -
        Current depth: 1, Choice: +
        Current depth: 2, Choice: exp
        Generated expression at depth 3: x1
        Generated expression at depth 2: ['exp', 'x1']
        Current depth: 2, Choice: -
        Generated expression at depth 3: 3.533989748458225
        Generated expression at depth 3: x2
        Generated expression at depth 2: ['-', 3.533989748458225, 'x2']
        Generated expression at depth 1: ['+', ['exp', 'x1'], ['-', 3.533989748458225, 'x2']]
        Current depth: 1, Choice: +
        Current depth: 2, Choice: +
        Generated expression at depth 3: x1
        Generated expression at depth 3: x1
        Generated expression at depth 2: ['+', 'x1', 'x1']
        Current depth: 2, Choice: /
        Generated expression at depth 3: 4.32039225844807
        Generated expression at depth 3: x1
        Generated expression at depth 2: ['/', 4.32039225844807, 'x1']
        Generated expression at depth 1: ['+', ['+', 'x1', 'x1'], ['/', 4.32039225844807, 'x1']]
        Generated expression at depth 0: ['-', ['+', ['exp', 'x1'], ['-', 3.533989748458225, 'x2']], ['+', ['+', 'x1', 'x1'], ['/', 4.32039225844807, 'x1']]]
        >>> isinstance(random_expr, list)
        True
        >>> len(random_expr) > 0
        True
        """
        
        if current_depth < self.max_depth:
            choice = random.choice(['+', '-', '*', '/', 'exp', 'log', 'const', 'var'])
            #print(f"Current depth: {current_depth}, Choice: {choice}")
            if choice in ['+', '-', '*', '/']:
                left = self.generate_random_expression(current_depth + 1)
                right = self.generate_random_expression(current_depth + 1)
                expr = [choice, left, right]
            elif choice in ['exp', 'log']:
                operand = self.generate_random_expression(current_depth + 1)
                expr = [choice, operand]
            elif choice == 'const':
                expr = random.uniform(-10, 10)
            else:  # 'var'
                expr = random.choice(self.variables)
        else:
            expr = random.choice(self.variables + [random.uniform(-10, 10)])
        #print(f"Generated expression at depth {current_depth}: {expr}")
        return expr

    def prefix_to_sympy(self, expr):
        """
        Convert a prefix notation expression to a SymPy expression.

        Parameters:
        expr (list or float): Expression in prefix notation or constant value.

        Returns:
        sympy.Basic: SymPy expression.

        Example:
            >>> regressor = SymbolicRegressor(variables=['x1', 'x2'])
            >>> expr = ['-', ['+', ['exp', 'x1'], ['-', 3.533989748458225, 'x2']], ['+', ['+', 'x1', 'x1'], ['/', 4.32039225844807, 'x1']]]
            >>> sympy_expr = regressor.prefix_to_sympy(expr)
            >>> expected_expr = sp.exp(sp.symbols('x1')) + 3.53398974845823 - sp.symbols('x2') - 2*sp.symbols('x1') - 4.32039225844807/sp.symbols('x1')
            >>> sympy_expr.equals(expected_expr)
            True
        """
        if isinstance(expr, list):
            # print(f"Converting list: {expr}")
            if expr[0] == '+':
                left = self.prefix_to_sympy(expr[1])
                right = self.prefix_to_sympy(expr[2])
                # print(f"Adding: {left} + {right}")
                result = left + right
            elif expr[0] == '-':
                left = self.prefix_to_sympy(expr[1])
                right = self.prefix_to_sympy(expr[2])
                # print(f"Subtracting: {left} - {right}")
                result = left - right
            elif expr[0] == '*':
                left = self.prefix_to_sympy(expr[1])
                right = self.prefix_to_sympy(expr[2])
                # print(f"Multiplying: {left} * {right}")
                result = left * right
            elif expr[0] == '/':
                left = self.prefix_to_sympy(expr[1])
                right = self.prefix_to_sympy(expr[2])
                # print(f"Dividing: {left} / {right}")
                result = left / right
            elif expr[0] == 'exp':
                operand = self.prefix_to_sympy(expr[1])
                # print(f"Exponential: exp({operand})")
                result = sp.exp(operand)
            elif expr[0] == 'log':
                operand = self.prefix_to_sympy(expr[1])
                # print(f"Logarithm: log({operand})")
                result = sp.log(operand)
            else:
                raise ValueError(f"Unrecognized operator: {expr[0]}")
            # print(f"Converted to SymPy: {result}")
            return result
        elif isinstance(expr, str):
            result = self.symbolic_vars.get(expr, None)
            # print(f"Converting variable: {expr} to {result}")
            return result
        else:
            result = sp.sympify(expr)
            # print(f"Converting constant: {expr} to {result}")
            return result

    def get_depth(self, expr):
        """
        Get the depth of an expression.

        Parameters:
        expr (str): expression in prefix notation

        Returns:
        int: Depth of the expression.

        Example:
        >>> regressor = SymbolicRegressor(variables=['x1', 'x2'])
        >>> prefix_expr = ['+', 'x1', 2]
        >>> depth = regressor.get_depth(expr)
        >>> depth
        1
        """
        if expr is None or not isinstance(expr, list):
            return 0
        if isinstance(expr, list):
            if len(expr) in [0, 1]:
                return 0
            if len(expr) == 2:
                return 1 + self.get_depth(expr[1])
            else:
                return 1 + max(self.get_depth(expr[1]), self.get_depth(expr[2]))

    def evaluate_expression(self, expr, data):
        """
        Evaluate a SymPy expression on given data.

        Parameters:
        expr (sympy.Basic): SymPy expression.
        data (dict): Dictionary of variable values.

        Returns:
        np.ndarray: Evaluated values or None if the evaluation leads to an error.
        """
        try:
            f = sp.lambdify(self.variables, expr, 'numpy')
            predictions = f(*[data[var] for var in self.variables])
            # Check for invalid values like inf, -inf, nan
            if np.any(np.isinf(predictions)) or np.any(np.isnan(predictions)):
                return None
            return predictions
        except Exception as e:
            return None

    def fit(self, x, y):
        # Generate random expressions
        expressions = [self.generate_random_expression() for _ in range(self.population_size)]
        sympy_exprs = [self.prefix_to_sympy(expr) for expr in expressions]

        # Ensure y is in float format
        y = np.array(y, dtype=float)

        # Ensure x values are in float format
        x = {k: np.array(v, dtype=float) for k, v in x.items()}

        # Evaluate the expressions and take only valid ones
        valid_exprs = []
        for expr in sympy_exprs:
            try:
                predictions = self.evaluate_expression(expr, x)
                if predictions is not None and np.all(np.isfinite(predictions)):
                    valid_exprs.append((expr, predictions))
            except Exception as e:
                continue

        if not valid_exprs:
            raise ValueError("No valid expressions generated. Try increasing the population size or max_depth.")

        print(len(valid_exprs))

        with pm.Model() as model:
            # Define coefficients as beta distributions between 0 and 1
            coefficients = [pm.Beta(f'coeff_{i}', alpha=2, beta=2) for i in range(len(valid_exprs))]

            # Linearly combine expressions with coefficients
            predictions = pt.zeros_like(y, dtype=float)
            for coeff, (_, pred) in zip(coefficients, valid_exprs):
                predictions += coeff * pred

            # Define the likelihood
            sigma = pm.HalfNormal('sigma', sigma=1)
            y_obs = pm.Normal('y_obs', mu=predictions, sigma=sigma, observed=y)

            # Perform MCMC sampling
            trace = pm.sample(1000, return_inferencedata=True)

        # Get the mean of coefficients from posterior distribution
        mean_coefficients = {f'coeff_{i}': trace.posterior[f'coeff_{i}'].mean().item() for i in range(len(valid_exprs))}

        # Find the expression with the highest coefficient
        best_expr_index = max(mean_coefficients, key=lambda k: abs(mean_coefficients[k]))
        self.best_expr = valid_exprs[int(best_expr_index.split('_')[1])][0]

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