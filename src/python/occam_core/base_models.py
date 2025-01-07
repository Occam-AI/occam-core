import datetime
import enum
import inspect
import json
import random
import types
import typing
import uuid
from typing import (Any, Dict, List, Literal, Optional, Self, Type, Union,
                    get_args, get_origin)

from git import Optional
from occam_core.util.common import get_logger
from occam_core.util.data_types.occam_data_type import OccamDataType
from pydantic import BaseModel, Field, create_model, model_validator

INPUTS_LABEL = "inputs"
PARAMS_LABEL = "params"
OUTPUTS_LABEL = "outputs"
DATASETS_FIELD_NAME = "datasets"


# this is used to decide on the variable type
# in duckdb for any particular python field.
PYTHON_TO_DUCKDB = {
    int: "BIGINT",
    bytes: "BLOB",
    str: "VARCHAR",
    float: "DOUBLE",
    bool: "BOOLEAN",
    datetime.datetime: "DATETIME",
    datetime.date: "DATETIME",
    dict: "JSON",
    list: "JSON",
    uuid.UUID: "UUID",
    object: "JSON"
}


# this is used to map a field from json data schema
# to python code schema
JSON_TO_PYTHON_MAPPING = {
    'string': str,
    'date-time': datetime.datetime,
    'date': datetime.date,
    'uuid': uuid.UUID,
    'number': float,
    'integer': int,
    'object': object,
    'binary': bytes,
    'boolean': bool,
    'array': list,
}

# this is used to map a field from python code schema
# to json data schema
PYTHON_TO_JSON_MAPPING = {
    str: 'string',
    datetime.datetime: 'date-time',
    datetime.date: 'date',
    uuid.UUID: 'uuid',
    float: 'number',
    int: 'integer',
    bytes: 'binary',
    bool: 'boolean',
    list: 'array',
    object: 'object',
    dict: 'object',
}

# this defines the json types that are accepted by
# the SchemaParameter class.
AcceptedJSONLiteral = typing.Literal[
    'string',
    'date-time',
    'date',
    'uuid',
    'number',
    'integer',
    'object',
    'pydantic',
    'literal',
    'binary',
    'boolean',
    'array'
]


logger = get_logger()


class IOModel(OccamDataType):

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"  # This allows additional attributes

    @classmethod
    def __init_subclass__(cls, **kwargs):
        # we don't want to allow type unions
        for field_name, field_type in cls.__annotations__.items():
            origin = get_origin(field_type)
            args = get_args(field_type)
            if (
                type(field_type) is types.UnionType
                or origin == typing.Union
            ) and (
                # we only allow unions of the form
                # Union[None, Type], which is the case
                # for Optional[Type]
                len(args) > 2
                or types.NoneType not in args
            ):
                raise TypeError(f"IOModel field {field_name} cannot be a union of different types.")
        super().__init_subclass__(**kwargs)

    # assigns empty lists and dicts to non-assigned list
    # and dict fields respectively.
    @model_validator(mode="after")
    def fill_null_values(self) -> Self:
        for field_name, field_info in self.model_fields.items():
            field_type = field_info.annotation
            toplevel_types = (get_origin(field_type),)
            args = getattr(field_type, '__args__', None)
            if toplevel_types[0] == typing.Union:
                toplevel_types = (args[0], get_origin(args[0]))

            if (field_type == dict or dict in toplevel_types) \
                    and not getattr(self, field_name, None):
                setattr(self, field_name, {})

            if (field_type == list or list in toplevel_types) \
                    and not getattr(self, field_name, None):
                setattr(self, field_name, [])
        return self

    # # disallow 'dataset' to be used as a field name
    # @model_validator(mode="after")
    # def validate_dataset_field(self) -> Self:
    #     if DATASETS_FIELD_NAME in self.__dict__:
    #         raise ValueError(
    #             f"'{DATASETS_FIELD_NAME}' cannot be used as a field name for input/output models"
    #         )
    #     return self

    @classmethod
    def load_model(cls, data: Self) -> Self:
        if not isinstance(data, IOModel):
            raise TypeError(f"Data of type {type(data)} cannot be loaded into {cls.__name__}.")
        return super().load_model(data)

    def update_model(self, io_model: Self) -> Self:
        """
        Update the params model with data from another io model
        instance.

        We use get_data_dict, instead of model_dump,
        so as to preserve the data in its original form, not
        serialized, which is necessary for converting complex
        types to simple ones in case where the conversion logic
        is lost if the data is dumped in simple dict form.
        """
        for k in io_model.get_field_keys():
            setattr(self, k, getattr(io_model, k))
        base_model = self.get_data_dict() | io_model.get_data_dict()
        return self.__class__(**base_model)

    def format_as_str(self) -> str:
        output = ""

        def _format(val):
            result = val
            if isinstance(result, IOModel):
                result = result.format_as_str()
            elif type(result) is list:
                result = "\n".join([f"{_format(v)}" for v in result])
            elif type(result) is dict:
                result = "\n".join(
                    [
                        f"{k}: {_format(v)}" for k, v in result.items()
                    ]
                )
            return result

        for field, value in self.__dict__.items():
            if value is not None:
                value = _format(value)
                output += f"{field}: {value}\n"

        return output

    def concatenate_models(cls, io_models: List[Self]) -> List[Self]:
        # TODO: document how concatenate models works.
        """
        The function of this is to merge a list of models
        of the the respective type. this is relevant when the behavior of a tool
        may involve combining related records.
        """
        raise NotImplementedError(
            f"No combine_models implementation for {cls.__name__}.")


class InputsModel(IOModel):
    ...


class DatasetType(str, enum.Enum):
    """
    We specify datasets passed to tools for use.
    Tools can be passed either source dataset names,
    which are simply the names under which the records
    they are consuming are saved in the DB, OR, reference
    dataset names, that the tool may internally query, for
    e.g. like the querier tool.

    Source datasets are automatically passed to tools by
    the executor that's running the tool.

    Reference datasets are designated by the planner at
    planning time.
    """
    STATIC = "static"
    SOURCE_TOOL_OUTPUT = "source_tool_output"


class ReferenceDatasetsMode(str, enum.Enum):
    NONE = "none"
    WHITELIST = "whitelist"
    EVERYTHING = "everything"


class ParamsIOModel(IOModel):

    datasets: Optional[List[str]] = None
    reference_datasets_mode: ReferenceDatasetsMode = ReferenceDatasetsMode.NONE

    concatenate_inputs: bool = False

    # This is the max number of input records to run in parallel.
    threaded_run_batch_size: int = 1

    # @classmethod
    # def __init_subclass__(cls, **kwargs):
    #     super().__init_subclass__(**kwargs)
    #     # Validate datasets field type annotation if it exists in subclass
    #     if DATASETS_FIELD_NAME in cls.__annotations__:
    #         field_type = cls.__annotations__[DATASETS_FIELD_NAME]
    #         origin = get_origin(field_type)
    #         args = get_args(field_type)
    #         if origin != list or args != (str,):
    #             raise TypeError(
    #                 f"'{DATASETS_FIELD_NAME}' field must be annotated as List[str]"
    #             )

    @model_validator(mode="after")
    def validate_batch_size(self) -> Self:
        if self.threaded_run_batch_size < 1:
            raise ValueError("threaded_run_batch_size must be greater than 0.")
        return self

    # # allow param children to have a datasets field.
    @model_validator(mode="after")
    def validate_datasets(self) -> Self:
        self.datasets = self.datasets or []
        if self.reference_datasets_mode == ReferenceDatasetsMode.WHITELIST \
                and len(self.datasets) == 0:
            raise ValueError("datasets must be provided when reference_datasets_mode is WHITELIST.")
        if self.reference_datasets_mode != ReferenceDatasetsMode.WHITELIST \
                and len(self.datasets) > 0:
            raise ValueError("white_list datasets must be empty when reference_datasets_mode is not WHITELIST.")
        return self


class OutputsModel(IOModel):
    ...


class VectorEmbeddingIOModel(IOModel):
    text: str
    embedding: List[float]


def _construct_annotations_and_defaults(kwargs):
    annotations: Dict[str, type] = {}
    defaults: Dict[str, Any] = {}

    for field_name, field_type in kwargs.items():
        annotations[field_name] = field_type
        if get_origin(field_type) is Union and type(None) in get_args(field_type):
            defaults[field_name] = None
    return annotations, defaults


def construct_io_model(
    base_io_model: Type[IOModel] = IOModel,
    io_model_name: str = None,
    model_category: Literal[INPUTS_LABEL, PARAMS_LABEL] = INPUTS_LABEL,
    **kwargs: type
) -> Type[IOModel]:

    annotations, defaults = _construct_annotations_and_defaults(kwargs)
    ultimate_parent_model = (
        IOModel if model_category == INPUTS_LABEL else ParamsIOModel
    )

    if not issubclass(base_io_model, ultimate_parent_model):
        raise ValueError(
            f"base_io_model must be a subclass of {ultimate_parent_model.__name__}, got {base_io_model}"
        )

    if io_model_name is None:
        io_model_name = f'IOModel_{hash(random.random())}'

    # Create the Pydantic model dynamically with annotations and defaults
    model = type(io_model_name, (base_io_model,), {"__annotations__": annotations, **defaults})

    return model


def construct_params_io_model(
    base_io_model: Type[ParamsIOModel] = ParamsIOModel,
    io_model_name: str = None,
    **kwargs: type
) -> Type[ParamsIOModel]:
    
    return construct_io_model(
        base_io_model=base_io_model,
        io_model_name=io_model_name,
        model_category=PARAMS_LABEL,
        **kwargs
    )


def is_optional(field_type: type) -> bool:
    return (
        type(field_type) is types.UnionType
        or get_origin(field_type) == typing.Union
    ) and type(None) in get_args(field_type)


class SchemaParameter(BaseModel):
    """
    This is a class that's used to encapsulate how python/pydantic
    model schema is serialized into a json schema, and is deserialized
    back.

    Due to challenges in serialization of complex schemas, currently
    we support serialization that maintain the following schema details;

    (a) primitive types.
    (b) pydantic models that we have in code.
    (c) literal types.
    (d) lists and dicts, without retaining metadata about their sub-types

    This is used specifcally for loading the model of a dataset
    whose json schema is coming from a database, at the moment
    """
    name: str
    kind: AcceptedJSONLiteral
    allowed_values: Optional[list[str]] = None
    # if kind == 'pydantic', this is the json schema of the pydantic field
    pydantic_field_json_schema: Optional[str] = None
    description: Optional[str] = None
    # to signify Optional[type]
    optional: bool = False

    def to_python_type(self):
        from occam_tools.util.data_types.class_map import \
            MODEL_NAMES_TO_CLASSES_REGISTRY

        if self.kind == 'pydantic':
            custom_definition = json.loads(self.pydantic_field_json_schema)
            return MODEL_NAMES_TO_CLASSES_REGISTRY[custom_definition['title']]
        elif self.kind == 'literal':
            return Literal[tuple(self.allowed_values)]
        return JSON_TO_PYTHON_MAPPING[self.kind]

    def to_pydantic_field_definition(self):
        # currently structured oututs don't support default values
        # if self.optional:
        #     python_type = Optional[self.to_python_type()]
        #     return (python_type, Field(
        #         default=None,
        #         description=self.description,
        #     ))
        # else:
        #     python_type = self.to_python_type()
        #     return (python_type, Field(
        #         default=self.default if self.default is not None else ...,
        #         description=self.description,
        #     ))
        if self.optional:
            python_type = Optional[self.to_python_type()]
        else:
            python_type = self.to_python_type()
        return (python_type, Field(
            ...,
            description=self.description,
        ))

    @classmethod
    def from_code_schema(cls, code_schema: dict) -> list[typing.Self]:
        pydantic_model = create_model("CodeSchemaModel", **{
            column_name: (definition['type'], ...)
            for column_name, definition in code_schema.items()
        })
        return cls.from_model_type(pydantic_model)

    @classmethod
    def from_model_type(cls, model_type: type[BaseModel]) -> list[typing.Self]:
        from occam_tools.util.data_types.class_map import \
            MODEL_NAMES_TO_CLASSES_REGISTRY
        if not issubclass(model_type, BaseModel):
            raise ValueError(f"Model type {model_type} is not a subclass of BaseModel.")
        schema = []
        model_fields = {
            field_name: field_info.annotation
            for field_name, field_info in model_type.model_fields.items()
            if not field_name.startswith('_')  # Exclude private attributes
        }
        for name, variable_type in model_fields.items():
            description = model_type.model_fields[name].description
            base_variable_type = get_base_type(variable_type)
            is_optional_field = is_optional(variable_type)
            if (
                issubclass(base_variable_type, BaseModel)
                and base_variable_type.__name__ in MODEL_NAMES_TO_CLASSES_REGISTRY
            ):
            #if issubclass(base_variable_type, BaseModel):
                schema.append(
                    SchemaParameter(
                        name=name,
                        kind='pydantic',
                        pydantic_field_json_schema=json.dumps(base_variable_type.model_json_schema()),
                        description=description,
                        optional=is_optional_field
                    )
                )
            elif get_origin(variable_type) == Literal:
                args = getattr(variable_type, '__args__', None)
                schema.append(
                    SchemaParameter(
                        name=name,
                        kind='literal',
                        allowed_values=list(args),
                        description=description,
                        optional=is_optional_field
                    )
                )
            else:
                schema.append(
                    SchemaParameter(
                        name=name,
                        kind=PYTHON_TO_JSON_MAPPING.get(base_variable_type, 'object'),
                        description=description,
                        optional=is_optional_field
                    )
                )
        # schema.append(SchemaParameter(name='_row_index', kind='integer'))
        return schema


# def get_code_schema_from_model_type(model_type: type[IOModel]) -> dict:
#     """"
#     This extracts the base type of each variable in an IOModel such that
#     it can be mapped to a duckdb type. For e.g. dict[str, str] becomes
#     dict, Enum becomes str, etc.
#     """
#     if not issubclass(model_type, IOModel):
#         raise ValueError(f"Model type {model_type} is not a subclass of IOModel")
#     code_data_schema = {}
#     for key, variable_type in model_type.get_variable_types_map().items():
#         base_type = get_base_type(variable_type)
#         code_data_schema[key] = {'type': base_type}
#     # code_data_schema['_row_index'] = {'type': int}
#     return code_data_schema


def get_base_type(variable_type: type):
    if variable_type in PYTHON_TO_DUCKDB:
        return variable_type
    args = getattr(variable_type, '__args__', None)
    origin = typing.get_origin(variable_type)
    if origin in (dict, list):
        return origin
    elif origin == Literal:
        return str
    elif (
        type(variable_type) is types.UnionType
        or origin == typing.Union
    ):
        # this assumes that all types in the Union map
        # to the same db type.
        # filter out NoneType if in union.
        filtered_args = [arg for arg in args if arg is not types.NoneType]
        return get_base_type(filtered_args[0])
    elif inspect.isclass(variable_type) and issubclass(variable_type, enum.Enum):
        return str
    else:
        return variable_type


class JsonSchemaModel(BaseModel):
    """
    A template for a dynamic model, which is a model
    that is created at runtime based on a set of
    fields and field types, usually provided by a
    planner agent.

    The template specifies the name to be given to
    the model and the list of fields and their
    types.

    This is mainly needed for plan time output model
    creation, that can leverage structured outputs.

    Currently this only supports;

    - simple primitive types
    - literal
    - dictionaries and lists
    - custom pydantic models that are hard
    coded in occam-tools.

    It doesn't support nested objects mixing
    pydantic models and dictionaries.
    """
    model_category: Literal[INPUTS_LABEL, OUTPUTS_LABEL] = OUTPUTS_LABEL
    model_name: str
    fields: list[SchemaParameter]

    @classmethod
    def from_model_type(
        cls,
        model_category: Literal[INPUTS_LABEL, OUTPUTS_LABEL],
        model_type: type[IOModel]
    ) -> Self:
        return cls(
            model_category=model_category,
            model_name=model_type.__name__,
            fields=SchemaParameter.from_model_type(model_type)
        )

    @classmethod
    def from_json_schema(
        cls,
        model_category: Literal[INPUTS_LABEL, OUTPUTS_LABEL],
        model_name: str,
        json_schema: list[SchemaParameter]
    ) -> Self:
        return cls(
            model_category=model_category,
            model_name=model_name,
            fields=json_schema
        )

    def to_pydantic_model(self) -> type[IOModel]:
        field_definitions = {
            field.name: field.to_pydantic_field_definition()
            for field in self.fields
        }
        return create_model(
            self.model_name,
            **field_definitions,
            __base__=IOModel
        )
