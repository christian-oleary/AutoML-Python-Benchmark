"""Lack of Cohesion of Methods (LCOM) metric calculation.

Adapted from https://github.com/potfur/lcom/
"""

from __future__ import annotations

import ast
import os
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from pathlib import Path

from loguru import logger


class ReflectionError(Exception):
    """Reflection error."""


class Reflection(object):
    """Abstract reflection class."""

    __metaclass__ = ABCMeta

    @abstractmethod
    def name(self):
        """Return the name of the reflected element."""
        raise NotImplementedError()


class ModuleReflection(Reflection):
    """Module reflection class."""

    @classmethod
    def from_file(cls, file: str) -> ModuleReflection | None:
        """Create a module reflection from a file.

        :param str file: File path.
        :return ModuleReflection | None: Module reflection instance or None if syntax error.
        """
        try:
            file = file.replace('/', os.path.sep).replace('\\', os.path.sep)
            with open(file, 'r', encoding='utf-8') as handle:
                content = handle.read()

            name = file.rsplit('.', 1)[0]
            replacements = ((os.path.sep, '.'), ('__init__', ''), ('__main__', ''))
            for needle, replace in replacements:
                name = name.replace(needle, replace)
            reflection = cls(name.strip('.'), ast.parse(content))
        except (UnicodeDecodeError, SyntaxError):
            reflection = None
        return reflection

    def __init__(self, name: str, node: ast.AST):
        self.__name = name
        self.__node = node

    def name(self) -> str:
        return self.__name

    def class_by_name(self, name: str) -> ClassReflection:
        """Get class reflection by name.

        :param str name: Class name.
        :raises ReflectionError: If the class is not found.
        :return ClassReflection: Class reflection instance.
        """
        for elem in self.classes():
            if elem.name() == name:
                return elem
        raise ReflectionError(f'Unknown class {name}')

    def classes(self) -> list[ClassReflection]:
        """Return all class reflections in the module.

        :return list[ClassReflection]: List of class reflection instances.
        """
        return [
            ClassReflection(self.__name, node)
            for node in ast.walk(self.__node)
            if isinstance(node, ast.ClassDef)
        ]


class ClassReflection(Reflection):
    """Class reflection class."""

    def __init__(self, module_name: str, node):
        self.__module_name = module_name
        self.__node = node

    def name(self) -> str:
        return f'{self.__module_name}.{self.__node.name}'

    def method_by_name(self, name: str) -> MethodReflection:
        """Get method reflection by name.

        :param str name: Method name.
        :raises ReflectionError: If the method is not found.
        :return MethodReflection: Method reflection instance.
        """
        nodes = [node for node in self.__class_methods() if node.name == name]
        try:
            return MethodReflection(self.__module_name, self.__node.name, nodes[0])
        except IndexError as e:
            raise ReflectionError(f'Unknown method {name}') from e

    def methods(self) -> list[MethodReflection]:
        """Return all method reflections in the class.

        :return list[MethodReflection]: List of method reflection instances.
        """
        return [
            MethodReflection(self.__module_name, self.__node.name, node)
            for node in self.__class_methods()
        ]

    def vars(self) -> list[str]:
        """Return all variable names in the class.

        :return list[str]: List of variable names.
        """
        result = self.__class_vars()
        result |= self.__instance_vars()
        result -= {node.name for node in self.__class_methods()}
        return list(result)

    def __class_vars(self):
        """Return class variable names.

        :return set[str]: Set of class variable names.
        """
        return {
            target.id
            for node in self.__node.body
            if isinstance(node, ast.Assign)
            for target in node.targets
        }

    def __instance_vars(self) -> set[str]:
        return {
            node.attr
            for node in ast.walk(self.__node)
            if isinstance(node, ast.Attribute) and node.value.id == 'self'  # type: ignore
        }

    def __class_methods(self) -> set[ast.FunctionDef]:
        return {node for node in self.__node.body if isinstance(node, ast.FunctionDef)}


class MethodReflection(Reflection):
    """Method reflection class."""

    def __init__(self, module_name: str, class_name: str, node):
        self.__module_name = module_name
        self.__class_name = class_name
        self.__node = node

    def name(self) -> str:
        """Return the method name."""
        return self.__call_name(self.__node.name)

    def is_constructor(self) -> bool:
        """Return True if the method is a constructor."""
        return self.__node.name == '__init__'

    def is_loose(self) -> bool:
        """Return True if the method does not access any class or instance variables."""
        return not (self.__calls() | self.__vars())

    def has_decorator(self, decorator_name: str) -> bool:
        """Return True if the method has a decorator.

        :param str decorator_name: Decorator name.
        :return bool: True if the method has the decorator, False otherwise.
        """
        if not hasattr(self.__node, 'decorator_list'):
            return False

        for decorator in self.__node.decorator_list:
            if self._decorator_name(decorator) == decorator_name:
                return True
        return False

    def calls(self) -> list[str]:
        """Return all method calls in the method."""
        return [self.__call_name(call) for call in self.__calls()]

    def vars(self) -> list[str]:
        """Return all variable names accessed in the method."""
        return list(self.__vars() - self.__calls())

    # ADDED
    def _decorator_name(self, decorator) -> str | None:
        """Return the name of the decorator.

        :param any decorator: Decorator node.
        :return str | None: Decorator name or None if not found.
        """
        if isinstance(decorator, ast.Name):
            return decorator.id
        if isinstance(decorator, ast.Attribute):
            return decorator.attr
        if isinstance(decorator, ast.Call):
            return self._decorator_name(decorator.func)
        return None

    def __vars(self) -> set[str]:
        """Return variable names accessed in the method."""
        return {
            node.attr
            for node in ast.walk(self.__node)
            if isinstance(node, ast.Attribute)
            and hasattr(node.value, 'id')
            and node.value.id in ('cls', 'self')
        }

    def __calls(self) -> set[str]:
        """Return method call names in the method."""
        return {
            node.func.attr
            for node in ast.walk(self.__node)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and hasattr(node.func.value, 'id')
            and node.func.value.id in ('cls', 'self')
        }

    def __call_name(self, node_name: str) -> str:
        """Return the fully qualified call name.

        :param str node_name: Node name.
        :return str: Fully qualified call name.
        """
        return f'{self.__module_name}.{self.__class_name}::{node_name}'


class LCOM4:
    """Lack of Cohesion of Methods (LCOM) metric calculation."""

    def calculate(self, ref: ClassReflection) -> int:
        """Calculate LCOM4 metric for the given class reflection.

        :param ClassReflection ref: Class reflection instance.
        :return int: LCOM4 metric value.
        """
        paths = self.__call_paths(ref)
        groups = self.__match_groups(paths.values())
        groups = self.__match_groups(groups)
        return len(groups)

    def __call_paths(self, ref: ClassReflection) -> dict[str, set[str]]:
        """Return call paths for methods in the class reflection.

        :param ClassReflection ref: Class reflection instance.
        :return dict[str, set[str]]: Dictionary mapping method names to sets of related method/variable names.
        """
        result: dict[str, set[str]] = defaultdict(set)
        # Iterate through methods
        for method in ref.methods():
            # Skip constructors, loose methods, and class methods
            if any(
                [method.is_constructor(), method.is_loose(), method.has_decorator('classmethod')]
            ):
                continue
            # Get method name and related names
            name = method.name()
            result[name] |= set([name] + method.vars())
            # Follow calls recursively
            for call in method.calls():
                result[name].add(call)
                result[call].add(name)
                result[name] |= self.__follow_call(ref, call)
        return result

    def __follow_call(self, ref, name):
        # Recursively follow method calls to gather related names
        try:
            method = ref.method_by_name(name)
        except ReflectionError:
            return set()
        # Gather variables and calls for the method
        result = set(method.vars() + method.calls())
        for call in method.calls():
            if call == name:
                continue
            result |= self.__follow_call(ref, call)

        return result

    def __match_groups(self, groups):
        result = list()
        # Iterate through groups and merge overlapping ones
        for group in groups:
            match = self.__find_matching_group(group, result)
            match |= group
        return result

    def __find_matching_group(self, path, groups):
        # Find a group that overlaps with the given path
        for other in groups:
            if other & path:
                return other
        # No match found, create a new group
        other = set()
        groups.append(other)
        return other


class LCOMRunner(object):
    """LCOM metric runner."""

    def __init__(self):
        self.__lcom = LCOM4()

    def handle(
        self, paths: list[str | Path], file_filter: str | None = None
    ) -> tuple[list[tuple[str, int]], float]:
        """Handle LCOM calculation for the given paths.

        :param list[str | Path] paths: List of file or directory paths.
        :param str | None file_filter: Optional filename filter.
        :return tuple[list[tuple[str, int]], float]: Tuple of list of (class name, LCOM value) and average LCOM.
        """
        refs = list()
        for path in paths:
            refs += self.__gather_refs(path, file_filter)
        result, average = self.__aggregate(refs)
        return result, average

    def find(self, path: str | Path, filename: str | None = None) -> list[ModuleReflection]:
        """Find modules in the given path.

        :param str | Path path: File or directory path.
        :param str | None filename: Optional filename filter.
        :return list[ModuleReflection]: List of module reflections.
        """
        if os.path.isfile(path):
            return self.__find_in_file(path, filename)
        return self.__find_in_directory(path, filename)

    def __aggregate(self, refs: list[ClassReflection]) -> tuple[list[tuple[str, int]], float]:
        result: list[tuple[str, int]] = list()
        # Return early if no class reflections
        if not refs:
            return result, 0
        # Calculate LCOM for each class reflection and compute average
        for ref in refs:
            result.append((ref.name(), self.__lcom.calculate(ref)))
        average = sum([elem[1] for elem in result]) / len(result)
        return result, average

    def __gather_refs(self, path: str | Path, file_filter: str | None) -> list[ClassReflection]:
        refs = list()
        for mod in self.find(path, file_filter):
            refs += mod.classes()
        return refs

    def __find_in_directory(
        self, path: str | Path, file_filter: str | None = None
    ) -> list[ModuleReflection]:
        result = []
        for root, _, files in os.walk(path):
            for file in files:
                result += self.__find_in_file(os.path.join(root, file), file_filter)
        return result

    def __find_in_file(
        self, path: str | Path, file_filter: str | None = None
    ) -> list[ModuleReflection]:
        """Find module reflections in a file.

        :param str | Path path: File path.
        :param str | None file_filter: Optional filename filter.
        :return list[ModuleReflection]: List of module reflections.
        """
        result: list[ModuleReflection] = []
        # Check if the file matches the filter and is a Python file
        path = str(path)
        if path.split('.')[-1] == 'py' and self.__matches(path, file_filter):
            # Create module reflection from the file
            reflection = ModuleReflection.from_file(path)
            if reflection is not None:
                result.append(reflection)
        return result

    def __matches(self, file: str, file_filter: str | None = None) -> bool:
        return file_filter is None or file_filter in file


if __name__ == '__main__':
    # Example usage:
    output, mean = LCOMRunner().handle(['./src/ml/tasks.py'])
    logger.success(output)
    logger.success(mean)
