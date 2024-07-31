from autora.theorist.autora_theorist_tutors.regressor import SymbolicRegressor


def test_get_depth():
    # Arrange
    theorist = SymbolicRegressor(variables=['x1', 'x2'])
    expr0 = 'x2'
    expr1 = ['+', 'x1', 2]
    expr2 = ['+', ['*', 'x1', 2], 3]
    expr3 = ['+', ['*', 'x1', 2], ['/', 3, 'x2']]

    # Act
    depth0 = theorist.get_depth(theorist.prefix_to_sympy(expr0))
    depth1 = theorist.get_depth(theorist.prefix_to_sympy(expr1))
    depth2 = theorist.get_depth(theorist.prefix_to_sympy(expr2))
    depth3 = theorist.get_depth(theorist.prefix_to_sympy(expr3))

    # Assert
    assert depth0 == 0
    assert depth1 == 1
    assert depth2 == 2
    assert depth3 == 3


def test_random_expression_depth():
    # Arrange
    max_depth = 3
    variables = ['x1', 'x2']
    theorist = SymbolicRegressor(variables=variables, max_depth=max_depth)

    # Act
    random_expression = theorist.generate_random_expression()
    sympy_expression = theorist.prefix_to_sympy(random_expression)
    depth = theorist.get_depth(sympy_expression)

    # Assert
    assert 0 <= depth <= max_depth
