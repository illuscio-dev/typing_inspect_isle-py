import pytest
from typing import Generic, TypeVar, Optional, Union, Any
from typing_inspect_isle import class_typevar_mapping, is_subclass
from dataclasses import dataclass


Var1 = TypeVar("Var1")
Var2 = TypeVar("Var2")
Var3 = TypeVar("Var2")


class VarGen(Generic[Var1]):
    pass


class ConcreteGen(VarGen[str]):
    pass


class VarInherited(VarGen[Var1]):
    pass


class ConcreteDoubleInherit(VarInherited[int]):
    pass


class DoubleGen(Generic[Var1, Var2]):
    pass


class DoubleConcrete(DoubleGen[int, float]):
    pass


class SingleFromDouble(DoubleGen[Var1, bytes]):
    pass


class CompletedDouble(SingleFromDouble[str]):
    pass


@dataclass
class GenericData(Generic[Var1]):
    pass


@dataclass
class ConcreteData(GenericData[str]):
    pass


class NonGeneric:
    pass


class DoubleHybridGeneric(NonGeneric, Generic[Var1]):
    pass


class ConcreteHybrid(DoubleHybridGeneric[float]):
    pass


@pytest.mark.parametrize(
    "type_in,result",
    [
        (ConcreteGen, {Var1: str}),
        (VarInherited, {}),
        (ConcreteDoubleInherit, {Var1: int}),
        (DoubleConcrete, {Var1: int, Var2: float}),
        (SingleFromDouble, {Var2: bytes}),
        (CompletedDouble, {Var1: str, Var2: bytes}),
        pytest.param(ConcreteData, {Var1: str}),
        (ConcreteHybrid, {Var1: float}),
    ],
)
def test_typevar_lookup(type_in, result):
    assert class_typevar_mapping(type_in, {}) == result


@pytest.mark.parametrize(
    "type_in, class_info, allow_union, allow_any, result, raises",
    [
        (int, int, False, False, True, False),
        (str, int, False, False, False, False),
        (Optional[int], int, False, False, True, False),
        (Union[str, int], int, True, False, True, False),
        (Union[str, int], int, False, False, True, True),
        (Union[str, int], int, False, False, True, True),
        (Union[Any, int], int, True, False, True, False),
        (Union[Any, str], int, True, False, False, False),
        (Any, int, False, True, True, False),
        (Union[Any, str], int, True, True, True, False),
        (Union[Any, str], int, False, True, True, True),
        (Optional[Any], int, False, True, True, False),
        (Optional[Any], int, False, False, True, True),
        (Any, int, False, False, True, True),
    ],
)
def test_is_subclass(
    type_in, class_info, allow_union: bool, allow_any: bool, result: bool, raises: bool
):
    if raises:
        with pytest.raises(TypeError):
            assert is_subclass(
                type_in, class_info, allow_unions=allow_union, allow_any=allow_any
            )
    else:
        answer = is_subclass(
            type_in, class_info, allow_unions=allow_union, allow_any=allow_any
        )
        assert answer is result
