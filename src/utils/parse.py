"""Helper functions for argparsing."""

import argparse
from dataclasses import Field, fields, MISSING
from typing import Any, Literal, get_args, get_origin

from models.base import ModelConfig


def config_to_parser(
    parser: argparse.ArgumentParser,
    config: Any,  # noqa: ANN401
) -> None:
    """Add the config entries as options to parser."""

    def _format_name(name: str) -> tuple[str, str]:
        """Format the name as cli arguments."""
        underscore_split = name.split("_")
        long_name = f"--{'-'.join(underscore_split)}"
        short_name = (
            f"-{name[0]}"
            if len(underscore_split) == 1
            else f"--{''.join([u[0] for u in underscore_split])}"
        )
        return long_name, short_name

    def _get_kwargs(field: Field) -> dict:
        """Return the action string based on the type."""
        kwargs = {}

        if field.type is int or field.type is float:
            kwargs["type"] = field.type
        if field.type is bool:
            kwargs["action"] = "store_true"
        if get_origin(field.type) is list:
            kwargs["action"] = "append"
        if get_origin(field.type) is Literal:
            kwargs["choices"] = list(get_args(field.type))

        if field.default is not  MISSING:
            kwargs["default"] = field.default
        elif field.default_factory is not MISSING:
            kwargs["default"] = field.default_factory()
        else:
            kwargs["required"] = True

        if "help" in field.metadata:
            kwargs["help"] = field.metadata["help"]

        kwargs["dest"] = field.name

        return kwargs

    def _add_argument(*name_or_flags: str, field: Field) -> None:
        """Add a cli argument with given name_or_flags."""
        parser.add_argument(
            *name_or_flags,
            **_get_kwargs(field),
        )

    for field in fields(config):
        if field.metadata.get("fixed"):
            continue
        long, short = _format_name(field.name)
        try:
            _add_argument(long, short, field=field)
        except ValueError as err:
            print(err)
            _add_argument(long, field=field)
