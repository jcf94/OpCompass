"""Uniform public parameter contracts for built-in operators."""

from opcompass.operators.base import Operator
from opcompass.registry import discover_operators


def test_all_builtin_operators_declare_explicit_specs():
    for name, operator_class in discover_operators().items():
        assert operator_class.spec is not Operator.spec, name
        operator = operator_class()
        assert operator.spec.name == name
        assert operator.spec.parameters
        assert all(parameter.value_type is int for parameter in operator.spec.parameters)
        assert all(parameter.minimum == 1 for parameter in operator.spec.parameters)


def test_builtin_specs_canonicalize_in_declaration_order():
    for operator_class in discover_operators().values():
        operator = operator_class()
        supplied = {
            parameter.name: parameter.default if parameter.default is not None else 1
            for parameter in reversed(operator.spec.parameters)
            if parameter.required or parameter.default is not None
        }
        canonical = operator.validate_dimensions(supplied)
        assert list(canonical) == [
            parameter.name for parameter in operator.spec.parameters
            if parameter.name in supplied
        ]
