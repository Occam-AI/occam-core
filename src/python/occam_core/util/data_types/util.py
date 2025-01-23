import inspect
import itertools
import types
import typing
from typing import Any, Type, Union, get_origin


def recursive_value_type_check(data: Any, spec_variable_type: type) -> bool:
    """
    This function checks if a value provided to the spec abides by
    the require value types dictated in the spec's type mappings.
    """
    if spec_variable_type == Any:
        return True

    args = getattr(spec_variable_type, '__args__', None)
    origin = get_origin(spec_variable_type)

    if (
        origin == dict
        and type(data) is dict
        and all(type(key) is args[0] for key in data)
        and all(recursive_value_type_check(value, args[1])
                for value in data.values())
    ):
        return True

    # For now we will not support mixed type lists eg List[int, str]
    # this is instead absorbed via accommodating UnionType below
    elif (
        origin == list
        and type(data) is list
        and len(args) == 1
        and all(recursive_value_type_check(datapoint, args[0])
                for datapoint in data)
    ):
        return True

    # UnionType is different from Union, so needs to be handled differently.
    # This will be the case of a union between primitive types, or classes,
    # or a mix of both. If any components of the union include typing data types
    # e.g. List or Dict, then those would be handled in the next recursion through
    # the above elif block.
    elif (
        (
            type(spec_variable_type) is types.UnionType
            or origin == typing.Union
        )
        and (
            typing.Any in args
            # either type is directly in args
            or any(is_compatible_type(type(data), arg)
                   for arg in args if isinstance(arg, type))
            # or, for more complex types (e.g. list[float]), recursively send it down
            or any(recursive_value_type_check(data, arg)
                   for arg in args if getattr(arg, '__args__', None))
        )
    ):
        return True

    return (
        origin not in (dict, list, typing.Union)
        and type(spec_variable_type) is not types.UnionType
        and is_compatible_type(
            variable_type=type(data),
            spec_variable_type=spec_variable_type
        )
    )


def recursive_type_check(variable_type: type, spec_variable_type: type) -> bool:
    """
    This function checks if value_type is
    compatible with spec_variable_type.
    """

    if spec_variable_type == Any:
        return True

    args = getattr(spec_variable_type, '__args__', None)
    origin = get_origin(spec_variable_type)

    value_args = getattr(variable_type, '__args__', None)
    value_origin = get_origin(variable_type)

    if (
        origin == dict
        and value_origin == dict
        and args[0] == value_args[0]
        and recursive_type_check(value_args[1], args[1])
    ):
        return True

    elif (
        origin == list
        and value_origin == list
        and len(args) == 1
        and recursive_type_check(value_args[0], args[0])
    ):
        return True

    spec_variable_type_is_union = type(spec_variable_type) is types.UnionType or origin == typing.Union
    variable_type_is_union = type(variable_type) is types.UnionType or value_origin == typing.Union

    if spec_variable_type_is_union or variable_type_is_union:
        return _union_type_compatibility_check(
            variable_type, variable_type_is_union,
            spec_variable_type, spec_variable_type_is_union)

    return (
        origin not in (dict, list)
        and is_compatible_type(
            variable_type=variable_type,
            spec_variable_type=spec_variable_type,
        )
    )


def recursive_model_convert(
    source_model_instance,
    target_model: Type
):
    from occam_core.util.data_types.occam import OccamDataType
    source_model_instance: OccamDataType
    target_model: Type[OccamDataType]
    """
    This function converts a source data model to a target model.

    This is needed over and above recursive_value_convert to
    create a three way recursion.

    Steps:
        load_model
        -> recursive_model_convert (expand the keys)
        -> recursive_value_convert (convert each key's value)
        which in turn may involve a load_model call if the value
        is a model as well.

    If instead we had a load_model -> recursive_value_convert only
    calls we would have an infinite loop where load_model calls
    recursive_value_convert which calls load_model again etc
    """
    source_model_data = source_model_instance.get_data_dict()
    target_model_data = source_model_data
    for field, field_type in target_model.get_variable_types_map().items():
        if field in source_model_data: 
            value = source_model_data[field]
            target_model_data[field] = recursive_value_convert(value, field_type)
    return target_model(**target_model_data)
 

def recursive_value_convert(
        value,
        spec_variable_type
    ) -> Any:

    from occam_core.util.data_types.occam import OccamDataType
    value: Union[Any | OccamDataType]
    spec_variable_type: Type[Union[Any | OccamDataType]]

    if spec_variable_type == Any:
        return value
    value_type = type(value)
    spec_type_args = getattr(spec_variable_type, '__args__', None)
    origin = get_origin(spec_variable_type)
    # in case of union, if type is a straight match to one in the union, we use that
    if type(spec_variable_type) is types.UnionType or origin == typing.Union:
        if value_type in spec_type_args:
            return value
        for arg in spec_type_args:
            if recursive_value_type_check(value, arg):
                return recursive_value_convert(
                    value, arg)
    elif inspect.isclass(spec_variable_type) and issubclass(spec_variable_type, OccamDataType):
        return spec_variable_type.load_model(value)
    elif inspect.isclass(value_type) and issubclass(value_type, OccamDataType):
        return value.transform_model(spec_variable_type)
    elif type(value) is list:
        return [recursive_value_convert(v, spec_type_args[0]) for v in value]
    elif type(value) is dict:
        return {
            recursive_value_convert(k, spec_type_args[0]):
            recursive_value_convert(v, spec_type_args[1])
            for k, v in value.items()
        }
    return value



def _union_type_compatibility_check(
        variable_type,
        variable_type_is_union,
        spec_variable_type,
        spec_variable_type_is_union
) -> bool:
    """
    This function checks if value_type is compatible with spec_variable_type, assuming that at least one of them
    is a UnionType. The check will return False if neither variable_type nor spec_variable_type is of type UnionType
    or has a type hint typing.Union.
    """

    # convert single args to list of 1 union elements
    # to do a pairwise permutation
    if not variable_type_is_union:
        variable_type_args = [variable_type]
    else:
        variable_type_args = getattr(variable_type, '__args__', None)

    if not spec_variable_type_is_union:
        spec_variable_type_args = [spec_variable_type]
    else:
        spec_variable_type_args = getattr(spec_variable_type, '__args__', None)

    return any(
        recursive_type_check(variable_arg, spec_variable_arg)
        for variable_arg, spec_variable_arg in itertools.product(
            variable_type_args, spec_variable_type_args)
    )


def is_compatible_type(variable_type, spec_variable_type) -> bool:
    from occam_core.util.data_types.occam import OccamDataType
    if not inspect.isclass(variable_type):
        return False

    return variable_type == spec_variable_type or (
        # ASSUMPTION: we only accept children if the
        # specified type hint is an abstract parent
        inspect.isabstract(spec_variable_type) and
        issubclass(variable_type, spec_variable_type)
    ) or (
        # value_category in [INPUTS_LABEL, PARAMS_LABEL] and
        issubclass(spec_variable_type, OccamDataType)
        and spec_variable_type.accepts(variable_type)
    ) or (
        # value_category in [INPUTS_LABEL, PARAMS_LABEL] and
        issubclass(variable_type, OccamDataType)
        and variable_type.emits(spec_variable_type)
    )
