


import enum
import types
import typing
from typing import List, Optional, Self, get_args, get_origin

from occam_core.util.occam_data_type import OccamDataType
from pydantic import model_validator


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
        """
        The function of this is to merge a list of models
        of the the respective type. this is relevant when the behavior of a tool
        may involve combining related records.
        """
        raise NotImplementedError(
            f"No combine_models implementation for {cls.__name__}.")


class InputsModel(IOModel):
    ...


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
