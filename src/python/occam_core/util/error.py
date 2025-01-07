from typing import Any, Set


class StrictRequiredVariablesViolated(ValueError):
    def __init__(
            self,
            target_model_name: str,
            source_model_name: str,
            model_category: str,
            missing_keys: Set[str],
            is_content: bool = True, *args, **kwargs):
        super().__init__(args, kwargs)
        if is_content:
            self.error_message = f"Required {model_category} fields {list(missing_keys)} of {target_model_name} not found in values of source model {source_model_name}."
        else:
            self.error_message = f"Required {model_category} fields {list(missing_keys)} of {target_model_name} not found in schema of source model {source_model_name}."
        self.missing_keys = missing_keys
        self.model_name = target_model_name
        self.test_model_name = source_model_name

    def __str__(self) -> str:
        return self.error_message


class TypeCheckFailedException(ValueError):
    def __init__(self,
                 source_model_name: str,
                 target_model_name: str,
                 model_category: str,
                 field: str,
                 checked_value: Any,
                 expected_value_type: str,
                 is_content: bool = False,
                 *args, **kwargs):
        super().__init__(args, kwargs)
        if is_content:
            self.error_message = f"{model_category} {source_model_name} variable: {field} with data {checked_value} is not of required type: {expected_value_type} of {target_model_name}."
        else:
            self.error_message = f"{model_category} {source_model_name} variable: {field} with type {checked_value} is not of required type: {expected_value_type} of {target_model_name}."
        self.model_name = target_model_name
        self.field = field
        self.checked_value = checked_value
        self.expected_value_type = expected_value_type
        self.is_content = is_content

    def __str__(self) -> str:
        return self.error_message