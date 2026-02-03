import inspect
import re
from enum import Enum, auto

from pygptlink.tool_definition import ArgSpec, ToolDefinition
from pygptlink.logging import logger


def _validate_type(type: str) -> str:
    # Map python types to JSON Schema:
    # https://json-schema.org/understanding-json-schema
    if re.match(r"[Ll]ist\[.*?\]", type):
        type = "array"

    mapping = {"str": "string", "int": "integer", "bool": "boolean", "float": "number"}
    type = type.strip()
    type = mapping.get(type, type)
    if type in {"string", "integer", "number", "boolean", "array"}:
        return type
    raise RuntimeError(f"Unknown parameter type: {type} encountered!")


class DocSection(Enum):
    SUMMARY = auto()
    ARGS = auto()
    RETURNS = auto()
    RAISES = auto()


class ToolsBase:
    """This is a convenience class that provides an easy way to make python functions callable by LLMs (from GPTCompletion.complete()). It automatically maps Python methods from a subclass of this class to an array of GPTToolDefinitions suitable for use with GPTCompletion.

    To use this class, subclass it and add code and state necessary to implement the methods you wish the model to be able to call, document them with a standard docstring. That's it, when doing a completion, pass the output of `make_tools_list()` to the completion call.

    Methods starting with an '_' and methods without docstring are ignored, as are methods with a docstring that contain #NO_GPT_TOOL in the description.
    """

    def make_tools_list(self) -> list[ToolDefinition]:
        ans: list[ToolDefinition] = []
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            doc_string = inspect.getdoc(method)
            if not doc_string:
                continue
            if name[0] == "_":
                continue  # Skip "protected/private" methods

            doc_lines = doc_string.splitlines()
            previous_blank = False
            current_section = DocSection.SUMMARY

            summary = ""
            required_args: list[ArgSpec] = []
            optional_args: list[ArgSpec] = []
            current_arg: ArgSpec | None = None

            ignore_tool = False

            for line in doc_lines:
                if "#NO_GPT_TOOL" in line:
                    ignore_tool = True
                    break
                if "args:".casefold() == line.casefold() and previous_blank:
                    current_section = DocSection.ARGS
                elif "returns:".casefold() == line.casefold() and previous_blank:
                    current_section = DocSection.RETURNS
                elif "raises:".casefold() == line.casefold() and previous_blank:
                    current_section = DocSection.RAISES
                elif line:
                    if current_section == DocSection.SUMMARY:
                        if summary:
                            summary += " " + line.strip()
                        else:
                            summary = line.strip()
                    elif current_section == DocSection.ARGS:
                        match = re.match(
                            pattern=r"\s*(\w+) \(([^)]+)\): (.+)", string=line
                        )
                        if match:
                            # New argument found
                            arg_name = match.group(1).strip()
                            arg_description = match.group(3).strip()

                            opt_match = re.match(
                                pattern=r"Optional\[(\w+)\].*|(.*?), optional.*",
                                string=match.group(2).strip(),
                            )
                            if opt_match:
                                arg_type = _validate_type(
                                    opt_match.group(1) or opt_match.group(2)
                                )
                                current_arg = ArgSpec(
                                    name=arg_name,
                                    type=arg_type,
                                    description=arg_description,
                                )
                                optional_args.append(current_arg)
                            else:
                                arg_type = _validate_type(match.group(2).strip())
                                current_arg = ArgSpec(
                                    name=arg_name,
                                    type=arg_type,
                                    description=arg_description,
                                )
                                required_args.append(current_arg)
                        else:
                            if not current_arg:
                                # This is a bug at startup, crash early.
                                raise RuntimeError("Current arg is empty!")
                            # Continuation of previous arg's description
                            current_arg["description"] += " " + line.strip()
                        pass
                    elif current_section == DocSection.RETURNS:
                        # Ignore the returns section, we don't need it
                        pass
                    elif current_section == DocSection.RAISES:
                        # Ignore the raises section, we don't need it
                        pass
                    else:
                        # This is a bug at startup, crash early.
                        raise RuntimeError("Unknown enum, this is a bug")
                previous_blank = not line

            if ignore_tool:
                continue

            tool = ToolDefinition(
                name=name,
                callback=method,
                description=summary,
                required_args=required_args,
                optional_args=optional_args,
            )
            logger.debug(f"Parsed tool: {tool.__dict__}")
            ans.append(tool)
        return ans
