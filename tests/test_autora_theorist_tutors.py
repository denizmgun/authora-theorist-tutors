from autora.theorist.autora_theorist_tutors.regressor import SymbolicRegressor


def test_get_depth():
    # Arrange
    theorist = SymbolicRegressor(variables=['x1', 'x2'])
    expr0 = 'x2'
    expr1 = ['+', 'x1', 2]
    expr2 = ['+', ['*', 'x1', 2], 3]
    expr3 = ['+', ['*', 'x1', 2], ['/', 3, 'x2']]
    expr4 = ['-', ['/', ['*', 'x1', 'x2'], ['exp', 'x2']], ['-', ['-', 'x2', 'x2'], 'x1']]

    # Act
    depth0 = theorist.get_depth(expr0)
    depth1 = theorist.get_depth(expr1)
    depth2 = theorist.get_depth(expr2)
    depth3 = theorist.get_depth(expr3)
    depth4 = theorist.get_depth(expr4)

    # Assert
    assert depth0 == 0
    assert depth1 == 1
    assert depth2 == 2
    assert depth3 == 2
    assert depth4 == 3


def test_random_expression_depth():
    # Arrange
    max_depth = 3
    variables = ['x1', 'x2']
    theorist = SymbolicRegressor(variables=variables, max_depth=max_depth)

    # Act
    random_expression = theorist.generate_random_expression()
    depth = theorist.get_depth(random_expression)

    # Assert
    assert 0 <= depth <= max_depth


def test_prefix_to_sympy():
    # Arrange
    theorist = SymbolicRegressor(variables=['x1', 'x2'])
    expr0 = 'x2'
    expr1 = ['+', 'x1', 2]
    expr2 = ['+', ['*', 'x1', 2], 3]
    expr3 = ['+', ['*', 'x1', 2], ['/', 3, 'x2']]
    expr4 = ['-', ['/', ['*', 'x1', 'x2'], ['exp', 'x2']], ['-', ['-', 'x2', 'x2'], 'x1']]
    expected_expr0 = 'x2'
    expected_expr1 = 'x1 + 2'
    expected_expr2 = '2*x1 + 3'
    expected_expr3 = '2*x1 + 3/x2'
    expected_expr4 = 'x1*x2*exp(-x2) + x1'

    # Act
    sympy_expr0 = theorist.prefix_to_sympy(expr0)
    sympy_expr1 = theorist.prefix_to_sympy(expr1)
    sympy_expr2 = theorist.prefix_to_sympy(expr2)
    sympy_expr3 = theorist.prefix_to_sympy(expr3)
    sympy_expr4 = theorist.prefix_to_sympy(expr4)

    # Assert
    assert str(sympy_expr0) == expected_expr0
    assert str(sympy_expr1) == expected_expr1
    assert str(sympy_expr2) == expected_expr2
    assert str(sympy_expr3) == expected_expr3
    assert str(sympy_expr4) == expected_expr4
