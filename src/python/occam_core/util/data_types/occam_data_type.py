import inspect
import itertools
import types
import typing
from typing import Any, Type, Union, get_origin

from occam_core.util.error import (StrictRequiredVariablesViolated,
                                    TypeCheckFailedException)
from pydantic import BaseModel
from typing_extensions import Self

pydantic_class_name_label = 'pydantic_class_name'


def is_type(variable_type: type):
    # covers direct types, dict types and list types.
    return isinstance(variable_type, type) or (
        isinstance(variable_type, typing.GenericAlias)
        and variable_type.__origin__ in (dict, list, typing.Union)
    )


class OccamDataType(BaseModel):
    """
    Base class for all data types. This class is used to define
    the complex data types that are used by occam tools
    and need to be serializable for storage in the database.

    We wrap an init around pydantic's BaseModel to allow for
    custom input conversion. This is useful for example when
    we want to allow for a string or a Code object to be passed
    to a function. We can then convert the string to a Code
    object in the init function, or even more complex behavior
    like nested model conversions (eg. a model A containing another
    model B instance, where the input data is a model C containing
    a simple field D that can be converted to model B instances.

    The class' load_model function can accept instances of itself
    , its children, dictionaries, data classes, or types that
    can be transformed to the class, as dictated by overriding
    the _load_model function, which extends the functionality
    of the init function to allow for custom input conversion,
    eg. a Code class accepting a string as code.

    The _accepts object allows specifying the types that are
    additionally accepted by _load_model.

    The class' transform_model function reads in a variable type
    that we would like to convert to and attempts that conversion,
    raising an error if no transformation was possible.

    The _emits object allows specifying the types that are
    additionally emitted by transform_model.
    """

    @classmethod
    def load_model(cls, data: Any, throw_error: bool = True) -> Self:
        """
        This is different from a direct OccamDataType(data) as in
        the latter we only accept a dictionary. It is somewhat
        similar to model_validate, but supports more cases.

        Loads from.
        1. a cls instance or a child instance.
        2. A model whose schema (i.e. each field) can be mapped
        to the schema of the current model (i.e. each field
        is acceptable)
        3. a variable type that's accepted by cls _accepts (could be another model)
        """

        if is_type(data):
            raise TypeError(f"data must be a value, not a type {data}.")
        if isinstance(data, cls) or type(data) == dict:
            loaded_data = cls.model_validate(data)
            if type(loaded_data) != cls:
                loaded_data = cls(**loaded_data.model_dump())
            return loaded_data
        elif cls._accepts(type(data)):
            return cls._load_model(data)
        elif (
            isinstance(data, OccamDataType) and
            cls.accepts_model_schema(data, throw_error=throw_error)
        ):
            # if the target model accepts the schema of
            # the submitted model we recurse it field by field.
            # used instead of recursive_value_convert to expand a
            # model's keys and avoid an infinite loop between
            # load_model and recursive_value_convert.
            return recursive_model_convert(
                source_model_instance=data,
                target_model=cls
            )

        raise TypeError(f"Data of type {type(data)} cannot be loaded into {cls.__name__}.")

    @classmethod
    def accepts(cls, variable_type):
        """
        Accepts a cls instance or a child instance
        , or another model or variable type that can be
        transformed to cls.
        """
        if not is_type(variable_type):
            variable_type = type(variable_type)
        return cls._accepts(variable_type) or (
            inspect.isclass(variable_type) and (
                issubclass(variable_type, cls)
                or (
                    issubclass(variable_type, OccamDataType)
                    and cls.accepts_model_schema(variable_type)
                )
            )
        )

    def transform_model(self, variable_type):
        """
        Model can return as an output:
        1. variable_type = cls, a non transformed cls instance.
        2. variable_type = another model, another model loaded with cls data.
        3. variable_type = something cls can emit, the emitted format derived
        from the model.
        """

        # check that something is a type meaning it's a class or dict[str, Any] or list[str]
        # etc
        if not is_type(variable_type):
            raise TypeError(f"variable_type must be a type, not a value {variable_type}.")
        if variable_type == type(self):
            return self
        elif inspect.isclass(variable_type) and issubclass(variable_type, OccamDataType):
            return recursive_model_convert(
                source_model_instance=self,
                target_model=variable_type
            )

        output_model = self._transform_model(variable_type)
        if output_model:
            return output_model
        raise TypeError(f"Cannot transform {self.__class__.__name__} to {variable_type}.")

    @classmethod
    def emits(cls, variable_type):
        if not is_type(variable_type):
            raise TypeError(f"variable_type must be a type, not a value {variable_type}.")
        return (
            variable_type == cls or
            cls._emits(variable_type) or (
                inspect.isclass(variable_type)
                and issubclass(variable_type, OccamDataType)
                and variable_type.accepts_model_schema(cls)
            )
        )

    @classmethod
    def is_pydantic_occam_dump(cls, dump):
        return pydantic_class_name_label in dump

    def model_dump_w_type(self):
        data = self.model_dump()
        data[pydantic_class_name_label] = type(self).__name__
        return data

    @classmethod
    def _load_model(cls, data):
        # this can be over-ridden by a child to custom convert
        # other types for the model to accept.
        raise NotImplementedError(
            f"No _load_model implementation for {cls.__name__}. "
            f"Model can't be custom loaded from {type(data)}.")

    def _transform_model(self, variable_type) -> Self:
        # this can be over-ridden by a child to specify new
        # types that the model can be transformed to.
        ...

    @classmethod
    def _accepts(cls, data: Any) -> bool:
        # this can be over-ridden by a child to specify new
        # types that the model can accept.
        return False

    @classmethod
    def _emits(cls, variable_type: str) -> bool:
        return False

    @classmethod
    def get_variable_type(cls, variable_name):
        return cls.model_fields.get(variable_name).annotation

    @classmethod
    def get_variable_types_map(cls) -> dict[str, type]:
        return {
            field_name: field_info.annotation
            for field_name, field_info in cls.model_fields.items()
            if not field_name.startswith('_')  # Exclude private attributes
        }

    @classmethod
    def get_variable_defaults_map(cls) -> dict[str, type]:
        return {
            field_name: field_info.default
            for field_name, field_info in cls.model_fields.items()
            if not field_name.startswith('_')  # Exclude private attributes
        }

    @classmethod
    def filter_model_fields(cls, data: Self) -> Self:
        """
        This function filters the fields of the input data
        to ensure that only the fields present in the model
        are retained. This is necessary for loading the data
        into the model recursively.
        """
        cls_fields =set(cls.get_field_keys())
        return {
            field: field_type
            for field, field_type in data.items()
            if field in cls_fields
        }

    @classmethod
    def get_required_variables_types_map(cls) -> dict[str, type]:
        return {
            field_name: field_info.annotation
            for field_name, field_info in cls.model_fields.items()
            if get_origin(field_info.annotation) != typing.Union
        }

    @classmethod
    def get_field_keys(cls) -> set[str]:
        return cls.model_fields.keys()

    def get_data_dict(self) -> dict[str, Any]:
        return self.__dict__ | (self.model_extra or {})

    def get_data_keys(self) -> set[str]:
        return (self.__dict__ | (self.model_extra or {})).keys()

    @classmethod
    def accepts_model_schema(
        cls,
        source_model: Type[Self] | Self,
        model_category: str = "model",
        throw_error: bool = False
    ):
        """
        Check if the io model is compatible with the current
        model instance. This allows both evaluating this for an
        input model instance and for an input model schema.

        The reason we don't rely on pydantic for the former
        is that pydantic is not able to validate if a simple
        type in the class model can be instantiated from a
        complex type in the input model, without explicitly
        specifying this every time.
        """

        required_fields = cls.get_required_variables_types_map().keys()
        is_content = not is_type(source_model)

        if is_content:
            if not isinstance(source_model, OccamDataType):
                raise TypeError(
                    f"source_model must be an instance of OccamDataType, not {type(source_model)}."
                )
            if isinstance(source_model, cls):
                return True
            source_model_name = source_model.__class__.__name__
            type_check_function = recursive_value_type_check
            # we use this instead of model_dump so we get the values in their
            # model form, not serialized, this allows transforming them
            # as per the logic of the model they are to be loaded to.
            model_values = source_model.get_data_dict()
        else:
            if not inspect.isclass(source_model) or not issubclass(source_model, OccamDataType):
                raise TypeError(
                    f"source_model must be a subclass of OccamDataType, not {source_model}."
                )
            if issubclass(source_model, cls):
                return True
            source_model_name = source_model.__name__
            type_check_function = recursive_type_check
            model_values = source_model.get_variable_types_map()

        model_keys = model_values.keys()
        missing_required_fields = set(required_fields) - set(model_keys)
        if missing_required_fields:
            error =  StrictRequiredVariablesViolated(
                target_model_name=cls.__name__,
                source_model_name=source_model_name,
                model_category=model_category,
                missing_keys=missing_required_fields,
                is_content=is_content
            )
            if throw_error:
                raise error
            # logger.error(error)
            return False

        for field, field_type in cls.get_variable_types_map().items():
            if field in model_values:
                model_value = model_values[field]
                types_compatible = type_check_function(
                    model_value,
                    spec_variable_type=field_type
                )
                if not types_compatible:
                    error =  TypeCheckFailedException(
                        source_model_name=source_model_name,
                        target_model_name=cls.__name__,
                        model_category=model_category,
                        field=field,
                        checked_value=model_value,
                        expected_value_type=field_type,
                        is_content=is_content
                    )
                    if throw_error:
                        raise error
                    return False
        return True


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


def recursive_model_convert(
    source_model_instance: OccamDataType,
    target_model: Type[OccamDataType]
) -> OccamDataType:
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
        value: Union[Any | OccamDataType],
        spec_variable_type: Type[Union[Any | OccamDataType]]) -> Any:

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
