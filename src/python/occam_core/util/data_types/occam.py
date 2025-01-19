import inspect
import itertools
import types
import typing
from typing import Any, Type, Union, get_origin

from occam_core.util.error import (StrictRequiredVariablesViolated,
                                   TypeCheckFailedException)
from pydantic import BaseModel
from typing_extensions import Self

from python.occam_core.util.data_types.util import (recursive_model_convert,
                                                    recursive_type_check,
                                                    recursive_value_type_check)

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
                    error = TypeCheckFailedException(
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
