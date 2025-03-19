import abc
import enum
import unittest
from types import NoneType
from typing import Any, Dict, List, Optional, Self, Union

from occam_core.util.data_types.occam import (OccamDataType,
                                              recursive_type_check,
                                              recursive_value_type_check)
from occam_core.util.data_types.util import (is_compatible_type,
                                             recursive_type_hint_derivation,
                                             recursive_value_convert)


class TestDataType(OccamDataType):
    """
    A simple string wrapper class that accepts itself or a string
    and can emit a string.

    Used to test the ability of IO Models to recursively load
    and transform itself and nested classes.
    """

    value: str

    @classmethod
    def _load_model(cls, data) -> Self:
        if isinstance(data, str):
            return TestDataType(value=data)

    def _transform_model(self, variable_type):
        if variable_type == str:
            return self.value

    @classmethod
    def _accepts(cls, variable_type) -> bool:
        return variable_type == str

    @classmethod
    def _emits(cls, variable_type) -> bool:
        return variable_type == str


class TestOccamDataType(unittest.TestCase):
    class _DummyEnum(str, enum.Enum):
        A = "A"
        B = "B"

    class ChildModel(TestDataType):
        value: str
        optional_value: Optional[str] = None
    
    class SameAsTestDataType(OccamDataType):
            value: str

    def test_load_model(self):

        # test loading a model that's the same type
        model = TestDataType(value="test")
        self.assertEqual(TestDataType.load_model(model), model)

        # test loading a model from a child BaseModel
        self.assertEqual(TestDataType.load_model(self.ChildModel(value="A")), TestDataType(value="A"))

        # test loading a model from a child BaseModel with more fields
        # and ignoring the extra ones when loading it in
        self.assertEqual(
            TestDataType.load_model(self.ChildModel(value="A", optional_value="B")),
            TestDataType(value="A")
        )

        # test loading a model that's submitted as a dictionary
        self.assertEqual(TestDataType.load_model({"value": "test"}), TestDataType(value="test"))

        # test loading a model from a primitive value that the model
        # can accept.
        self.assertEqual(TestDataType.load_model("test"), TestDataType(value="test"))

    def test_accepts(self):
        self.assertTrue(TestDataType.accepts(TestDataType))
        self.assertTrue(TestDataType.accepts(self.ChildModel))
        self.assertTrue(TestDataType.accepts(self.SameAsTestDataType))
        self.assertTrue(TestDataType.accepts(str))
        self.assertFalse(TestDataType.accepts(int))

    def test_transform_model(self):

        model = TestDataType(value="test")

        # test just pass as code
        self.assertEqual(model.transform_model(TestDataType), model)

        # test transform to string
        self.assertEqual(model.transform_model(str), "test")

        # test transform to similar class
        self.assertEqual(model.transform_model(self.SameAsTestDataType), self.SameAsTestDataType(value="test"))

        # test reject int transformation.
        with self.assertRaises(TypeError) as context:
            model.transform_model(int)

        message = str(context.exception)
        self.assertEqual(message, "Cannot transform TestDataType to <class 'int'>.")
    
    def test_emits(self):
        self.assertTrue(TestDataType.emits(TestDataType))
        self.assertTrue(TestDataType.emits(str))
        self.assertTrue(TestDataType.emits(self.SameAsTestDataType))
        self.assertFalse(TestDataType.emits(int))


class _BaseDummy(abc.ABC):
    @abc.abstractmethod
    def __call__(self):
        """Dummy abstract method"""


class _DummyCallableAbstract(_BaseDummy):
    @abc.abstractmethod
    def __call__(self):
        """Dummy abstract method"""


class _Dummy:
    def __call__(self):
        pass


class _DummyCallable(_Dummy):
    def __call__(self):
        pass


class RecursiveDataTypeCheck(unittest.TestCase):
    def test_recursive_dict(self):
        # tests for basic dict
        self.assertTrue(recursive_value_type_check({"a": 1, "b": 2}, dict[str, int]))

        # test for mixed types
        self.assertTrue(recursive_value_type_check({"a": 1, "b": "2"}, dict[str, int | str]))
        self.assertFalse(recursive_value_type_check({"a": 1, "b": "2"}, dict[str, int]))

        # test for recursive dict value types
        self.assertTrue(recursive_value_type_check({"a": {"b": 1}, "c": {"d": 2}}, dict[str, dict[str, int]]))
        self.assertFalse(recursive_value_type_check({"a": {"b": 1}, "c": {"d": "2"}}, dict[str, dict[str, int]]))

        # upper case dict
        self.assertTrue(recursive_value_type_check({"a": 1, "b": 2}, Dict[str, int]))

    def test_recursive_list(self):
        # tests for basic list
        self.assertTrue(recursive_value_type_check([1, 2, 3], list[int]))

        # test for mixed types
        self.assertTrue(recursive_value_type_check([1, 2, "3"], list[int | str]))
        self.assertFalse(recursive_value_type_check([1, 2, "3"], list[int]))

        # test for recursive list value types
        self.assertTrue(recursive_value_type_check([[1, 2], [3, 4]], list[list[int]]))
        self.assertFalse(recursive_value_type_check([[1, 2], [3, "4"]], list[list[int]]))

        # upper case list
        self.assertTrue(recursive_value_type_check([1, 2, 3], List[int]))

    def test_complex_mixed_dict(self):
        data = [{
            'finish_reason': 'stop',
            'index': 0,
            'message': {'content': 'The capital of France is Paris.',
                        'role': 'assistant', 'function_call': None, 'tool_calls': None}
        }]

        spec_type_mapping = list[dict[str, int | str | dict]]
        self.assertTrue(recursive_value_type_check(data, spec_type_mapping))

    def test_complex_any(self):
        data = [{
            'finish_reason': 'stop',
            'index': 0,
            'message': {'content': 'The capital of France is Paris.',
                        'role': 'assistant', 'function_call': None, 'tool_calls': None}
        }]

        spec_type_mapping = list[dict[str, Any]]
        self.assertTrue(recursive_value_type_check(data, spec_type_mapping))

    def test_mixed_iterable_non_iterable(self):
        self.assertTrue(recursive_value_type_check([1, 2, 3], int | list[int]))

    def test_order_shouldnt_matter(self):
        self.assertTrue(recursive_value_type_check([1, 2, 3], list[int] | None))
        self.assertTrue(recursive_value_type_check([1, 2, 3], None | list[int]))

    def test_allow_Any(self):
        self.assertTrue(recursive_value_type_check({"test": 123}, Any))
        self.assertTrue(recursive_value_type_check({"test": 123}, dict[str, Any]))
        self.assertTrue(recursive_value_type_check({"test": 123, "value": None}, dict[str, int | Any]))
        self.assertTrue(recursive_value_type_check(1, int | Any))
        self.assertTrue(recursive_value_type_check("1", int | Any))

    def test_data_check_with_occam_data_types(self):

        codestring = "codestring"
        # we don't convert between classes and dicts for type checks
        # as its correctness if not guaranteed (plan might be broken)
        self.assertFalse(
            recursive_value_type_check({"value": codestring}, TestDataType)
        )
        self.assertFalse(
            recursive_value_type_check(
                TestDataType(value=codestring), dict[str, Any]
            )
        )

       # assert that same type is captured
        self.assertTrue(recursive_value_type_check(TestDataType(value="4"), TestDataType))

        # assert primitive type loading and transforming
        self.assertTrue(recursive_value_type_check(codestring, TestDataType))
        self.assertTrue(recursive_value_type_check(TestDataType(value=codestring), str))

        class SourceStringSub(OccamDataType):
            sub_content: str

        class SourceStringMain(OccamDataType):
            sub: SourceStringSub

        class SourceTestDataTypeSub(OccamDataType):
            sub_content: TestDataType

        class SourceTestDataTypeMain(OccamDataType):
            sub: SourceTestDataTypeSub

        class TargetStringSub(OccamDataType):
            sub_content: str

        class TargetTestDataTypeSub(OccamDataType):
            sub_content: TestDataType

        class TargetStringMain(OccamDataType):
            sub: TargetStringSub

        class TargetTestDataTypeMain(OccamDataType):
            sub: TargetTestDataTypeSub

        # assert that compatible, and nested OccamDataTypes pass the
        # data type check
        self.assertTrue(
            recursive_value_type_check(
                SourceStringMain(sub=SourceStringSub(sub_content="s")),
                TargetStringMain
            )
        )
        self.assertTrue(
            recursive_value_type_check(
                SourceTestDataTypeMain(sub=SourceTestDataTypeSub(sub_content=TestDataType(value="s"))),
                TargetTestDataTypeMain
            )
        )

        # assert that they pass the type check when a an occam data type
        # to str type conversion is needed.
        self.assertTrue(
            recursive_value_type_check(
                SourceStringMain(sub=SourceStringSub(sub_content="s")),
                TargetTestDataTypeMain
            )
        )
        self.assertTrue(
            recursive_value_type_check(
                SourceTestDataTypeMain(sub=SourceTestDataTypeSub(sub_content=TestDataType(value="s"))),
                TargetStringMain
            )
        )


class TestRecursiveTypeCheck(unittest.TestCase):
    def test_recursive_dict(self):
        # tests for basic dict
        self.assertTrue(recursive_type_check(dict[str, int], dict[str, int]))

        # test for mixed types
        self.assertTrue(recursive_type_check(dict[str, int], dict[str, int | str]))
        self.assertFalse(recursive_type_check(dict[str, str], dict[str, int]))

        # test for recursive dict value types
        self.assertTrue(recursive_type_check(dict[str, dict[str, int]], dict[str, dict[str, int | bool]]))
        self.assertFalse(recursive_type_check(dict[str, dict[str, float]], dict[str, dict[str, int]]))

        # upper case dict
        self.assertTrue(recursive_type_check(dict[str, int], Dict[str, int]))

    def test_recursive_list(self):
        # tests for basic list
        self.assertTrue(recursive_type_check(list[int], list[int]))

        # test for mixed types
        self.assertTrue(recursive_type_check(list[float], list[int | str | float]))
        self.assertFalse(recursive_type_check(list[str], list[int]))

        # test for recursive list value types
        self.assertTrue(recursive_type_check(List[list[bool]], list[list[int | bool]]))
        self.assertFalse(recursive_type_check(list[list[float]], list[list[int]]))

        # upper case list
        self.assertTrue(recursive_type_check(list[int], List[int]))

    def test_mixed_iterable_non_iterable(self):
        self.assertTrue(recursive_type_check(list[int], int | list[int]))
        self.assertTrue(recursive_type_check(int, int | list[int]))

    def test_mixed_lists_dicts_and_unions(self):
        # Real for the "pages" field of scraper tool
        self.assertTrue(recursive_type_check(list[dict[str, str | list[str]]], list[dict[str, str | list[str]]]))
        self.assertTrue(
            recursive_type_check(list[dict[str, str | list[str]]], list[dict[str, list[str]]]))
        # Same as above but order in union flipped
        self.assertTrue(
            recursive_type_check(list[dict[str, str | list[str]]], list[dict[str, list[str] | str]]))

        # Incompatible unions, recursive_type_check should return False
        self.assertFalse(
            recursive_type_check(list[dict[str, list[int] | bool]], list[dict[str, list[str] | str]]))

    def test_order_shouldnt_affect(self):
        self.assertTrue(recursive_type_check(list[int], list[int] | None))
        self.assertTrue(recursive_type_check(list[int], None | list[int]))

    def test_type_check_with_occam_data_types(self):
        # we don't convert between classes and dicts for type checks
        # as its correctness if not guaranteed (plan might be broken)
        self.assertFalse(recursive_type_check(dict[str, Any], TestDataType))
        self.assertFalse(recursive_type_check(TestDataType, dict[str, Any]))

        # assert that same type is captured
        self.assertTrue(recursive_type_check(TestDataType, TestDataType))

        # assert primitive type loading and transforming
        self.assertTrue(recursive_type_check(str, TestDataType))
        self.assertTrue(recursive_type_check(TestDataType, str))

        class SourceStringSub(OccamDataType):
            sub_content: str

        class SourceStringMain(OccamDataType):
            sub: SourceStringSub

        class SourceTestDataTypeSub(OccamDataType):
            sub_content: TestDataType

        class SourceTestDataTypeMain(OccamDataType):
            sub: SourceTestDataTypeSub

        class TargetStringSub(OccamDataType):
            sub_content: str

        class TargetTestDataTypeSub(OccamDataType):
            sub_content: TestDataType

        class TargetStringMain(OccamDataType):
            sub: TargetStringSub

        class TargetTestDataTypeMain(OccamDataType):
            sub: TargetTestDataTypeSub

        # assert that compatible, and nested OccamDataTypes pass the type check
        self.assertTrue(recursive_type_check(SourceStringMain, TargetStringMain))
        self.assertTrue(recursive_type_check(SourceTestDataTypeMain, TargetTestDataTypeMain))

        # including when a an occam data type to str conversion is needed
        # or vice versa.
        self.assertTrue(recursive_type_check(SourceStringMain, TargetTestDataTypeMain))
        self.assertTrue(recursive_type_check(SourceTestDataTypeMain, TargetStringMain))


class TestIsCompatibleType(unittest.TestCase):
    def test_primitive_type_compatibility(self):
        self.assertTrue(is_compatible_type(int, int))
        self.assertTrue(is_compatible_type(str, str))
        self.assertFalse(is_compatible_type(int, float))
        self.assertFalse(is_compatible_type(int, bool))

    def test_class_compatibility(self):
        self.assertTrue(is_compatible_type(_DummyCallableAbstract, _BaseDummy))
        self.assertFalse(is_compatible_type(_BaseDummy, _DummyCallableAbstract))
        self.assertFalse(is_compatible_type(_DummyCallable, _DummyCallableAbstract))

    def test_occam_data_type_compatibility(self):
        self.assertTrue(is_compatible_type(TestDataType, str))
        self.assertTrue(is_compatible_type(str, TestDataType))
        self.assertTrue(is_compatible_type(TestDataType, TestDataType))

        # Unacceptable type compatibility
        self.assertFalse(is_compatible_type(int, TestDataType))
        self.assertFalse(is_compatible_type(TestDataType, float))
        self.assertFalse(is_compatible_type(TestDataType, bool))
    
class TestRecursiveValueConvert(unittest.TestCase):
    def test_load_model_conversion(self):
        data = "test"
        converted = recursive_value_convert(data, TestDataType)
        self.assertEqual(converted, TestDataType(value=data))
        # assert that same type can be loaded
        self.assertEqual(
            recursive_value_convert(TestDataType(value=data), TestDataType),
            TestDataType(value=data)
        )

    def test_transform_model_conversion(self):
        data = TestDataType(value="test")
        converted = recursive_value_convert(data, str)
        self.assertEqual(converted, "test")

    def test_load_model_conversion_list(self):
        data = ["test1", "test2"]
        converted = recursive_value_convert(data, list[TestDataType])
        self.assertListEqual(converted, [TestDataType(value="test1"), TestDataType(value="test2")])

    def test_transform_model_conversion_list(self):
        data = [TestDataType(value="test1"), TestDataType(value="test2")]
        converted = recursive_value_convert(data, list[str])
        self.assertListEqual(converted, ["test1", "test2"])

    def test_load_model_conversion_dict(self):
        data = {"key1": "test1", "key2": "test2"}
        converted = recursive_value_convert(data, dict[str, TestDataType])
        self.assertEqual(converted, {"key1": TestDataType(value="test1"), "key2": TestDataType(value="test2")})

    def test_transform_model_conversion_dict(self):
        data = {"key1": TestDataType(value="test1"), "key2": TestDataType(value="test2")}
        converted = recursive_value_convert(data, dict[str, str])
        self.assertEqual(converted, {"key1": "test1", "key2": "test2"})

    def test_union_value_selection(self):
        data = "test"
        converted = recursive_value_convert(data, TestDataType | str)
        self.assertEqual(converted, "test")

        data = TestDataType(value="test")
        converted = recursive_value_convert(data, TestDataType | str)
        self.assertEqual(converted, TestDataType(value="test"))

    def test_union_value_selection_list(self):
        data = ["test1", "test2"]
        converted = recursive_value_convert(data, list[TestDataType | str])
        self.assertListEqual(converted, ["test1", "test2"])

        data = [TestDataType(value="test1"), TestDataType(value="test2")]
        converted = recursive_value_convert(data, list[TestDataType | str])
        self.assertListEqual(converted, [TestDataType(value="test1"), TestDataType(value="test2")])

    def test_union_value_conversion(self):
        data = [TestDataType(value="test1"), TestDataType(value="test2")]
        converted = recursive_value_convert(data, list[str | int])
        self.assertListEqual(converted, ["test1", "test2"])
    
    def test_load_same_model(self):
        data = TestDataType(value="test")
        converted = recursive_value_convert(data, TestDataType)
        self.assertEqual(converted, data)

    def test_recursive_occam_types_conversions(self):

        class SourceStringSub(OccamDataType):
            sub_content: str

        class SourceStringMain(OccamDataType):
            sub: SourceStringSub

        class SourceTestDataTypeSub(OccamDataType):
            sub_content: TestDataType

        class SourceTestDataTypeMain(OccamDataType):
            sub: SourceTestDataTypeSub

        class TargetStringSub(OccamDataType):
            sub_content: str

        class TargetTestDataTypeSub(OccamDataType):
            sub_content: TestDataType

        class TargetStringMain(OccamDataType):
            sub: TargetStringSub

        class TargetTestDataTypeMain(OccamDataType):
            sub: TargetTestDataTypeSub

        # assert that compatible, and nested OccamDataTypes are loadable
        self.assertEqual(
            recursive_value_convert(
                SourceStringMain(sub=SourceStringSub(sub_content="1")),
                TargetStringMain
            ),
            TargetStringMain(sub=TargetStringSub(sub_content="1"))
        )

        self.assertEqual(
            recursive_value_convert(
                SourceTestDataTypeMain(sub=SourceTestDataTypeSub(sub_content=TestDataType(value="1"))),
                TargetTestDataTypeMain
            ),
            TargetTestDataTypeMain(sub=TargetTestDataTypeSub(sub_content=TestDataType(value="1")))
        )

        #assert that a nested mixture of primitive and
        #occam data types can be converted
        self.assertEqual(
            recursive_value_convert(
                SourceStringMain(sub=SourceStringSub(sub_content="1")),
                TargetTestDataTypeMain
            ),
            TargetTestDataTypeMain(sub=TargetTestDataTypeSub(sub_content=TestDataType(value="1")))
        )

        self.assertEqual(
            recursive_value_convert(
                SourceTestDataTypeMain(sub=SourceTestDataTypeSub(sub_content=TestDataType(value="1"))),
                TargetStringMain
            ),
            TargetStringMain(sub=TargetStringSub(sub_content="1"))
        )


def test_recursive_type_hint_derivation():
    # Test 1: Simple primitives
    test_int = 42
    test_float = 3.14
    test_str = "hello"
    test_bool = True

    assert recursive_type_hint_derivation(test_int) == int
    assert recursive_type_hint_derivation(test_float) == float
    assert recursive_type_hint_derivation(test_str) == str
    assert recursive_type_hint_derivation(test_bool) == bool
    assert recursive_type_hint_derivation(None) == NoneType

    # Test 2: Homogeneous collections
    test_int_list = [1, 2, 3, 4, 5]
    test_str_list = ["a", "b", "c"]
    test_str_dict = {"a": "apple", "b": "banana", "c": "cherry"}

    assert recursive_type_hint_derivation(test_int_list) == List[int]
    assert recursive_type_hint_derivation(test_str_list) == List[str]
    assert recursive_type_hint_derivation(test_str_dict) == Dict[str, str]

    # Test 3: Heterogeneous collections
    test_mixed_list = [1, "two", 3.0, True]
    test_mixed_dict = {
        "name": "John",
        "age": 30,
        "is_active": True,
        "scores": [95, 87, 92]
    }

    mixed_list_type = recursive_type_hint_derivation(test_mixed_list)
    mixed_dict_type = recursive_type_hint_derivation(test_mixed_dict)

    assert mixed_list_type == List[Union[int, str, float, bool]]
    assert mixed_dict_type == Dict[str, Union[str, int, bool, List[int]]]

    # Test 4: Nested structures
    test_nested = {
        "users": [
            {
                "name": "Alice",
                "age": 25,
                "hobbies": ["reading", "hiking"],
                "contact": {"email": "alice@example.com", "phone": None}
            },
            {
                "name": "Bob",
                "age": 30,
            }
        ],
        "config": {
            "max_users": 10,
            "features_enabled": True,
            "version": 2.1
        }
    }

    nested_type = recursive_type_hint_derivation(test_nested)
    assert nested_type == Dict[
        str,
        Union[
            List[Union[Dict[str, Union[str, int, List[str], Dict[str, Union[str, NoneType]]]],
                       Dict[str, Union[str, int]]]],
            Dict[str, Union[float, int, bool]]
        ]
    ]

    # Test 5: Edge cases
    empty_list = []
    empty_dict = {}
    mixed_type_values = {"a": 1, "b": "string", "c": 3.14}
    mixed_type_keys = {1: "one", "two": 2, 3.0: "three"}

    assert recursive_type_hint_derivation(empty_list) == List[Any]
    assert recursive_type_hint_derivation(empty_dict) == Dict[Any, Any]
    assert recursive_type_hint_derivation(mixed_type_values) == Dict[str, Union[int, str, float]]
    assert recursive_type_hint_derivation(mixed_type_keys) == Dict[Union[int, float, str], Union[str, int]]

    # Test 6: Complex data with pydantic models
    complex_data = {
        "metadaa": TestDataType(value="test"),
        "x": 1,
        "y": True
    }

    complex_type = recursive_type_hint_derivation(complex_data)
    assert complex_type == Dict[str, Union[TestDataType, int, bool]]


if __name__ == "__main__":

    unittest.main()
