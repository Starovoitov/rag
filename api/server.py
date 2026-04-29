from __future__ import annotations

import argparse
import contextlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, create_model

from main import REQUIRED_COMMAND_PARAMS, build_parser
from utils.cli_config import apply_config_defaults, load_cli_defaults, validate_required_command_params


DEFAULT_CLI_PARAMS_CONFIG = "cli.defaults.json"
SUPPORTED_COMMANDS = (
    "build_parser",
    "build_faiss",
    "demo_retrieval",
    "evaluation_runner",
    "reranker_pipeline",
    "run_rag",
    "cleanup_faiss",
)
CLI_DEFAULTS = load_cli_defaults(Path.cwd() / DEFAULT_CLI_PARAMS_CONFIG)


class CommandResponse(BaseModel):
    command: str
    argv: list[str]
    stdout: str
    stderr: str
    result: dict[str, Any] | None = None


@dataclass
class CommandSpec:
    model: type[BaseModel]
    actions: dict[str, argparse.Action]
    example_payload: dict[str, Any]


def _extract_subparsers(parser: argparse.ArgumentParser) -> argparse._SubParsersAction:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action
    raise RuntimeError("No subparsers found in CLI parser.")


def _action_field_type(action: argparse.Action) -> Any:
    if isinstance(action, argparse._StoreTrueAction):
        return bool

    if action.choices:
        literals = tuple(action.choices)
        return Literal.__getitem__(literals)  # type: ignore[misc]

    value_type = action.type if action.type in {int, float, str} else str
    return value_type | None


def _build_command_spec(
    subparser: argparse.ArgumentParser,
    config_defaults: dict[str, Any],
) -> CommandSpec:
    fields: dict[str, tuple[Any, Any]] = {}
    actions: dict[str, argparse.Action] = {}
    for action in subparser._actions:
        if not action.option_strings:
            continue
        if action.dest in {"help"}:
            continue

        field_type = _action_field_type(action)
        if action.dest in config_defaults:
            schema_default = config_defaults[action.dest]
        elif isinstance(action, argparse._StoreTrueAction):
            schema_default = False
        else:
            schema_default = None
        default = Field(default=schema_default, description=action.help)

        fields[action.dest] = (field_type, default)
        actions[action.dest] = action

    model_name = f"{subparser.prog.replace(' ', '_')}_Request"
    example_payload = {key: value for key, value in config_defaults.items() if key in fields}
    model = create_model(
        model_name,
        __base__=BaseModel,
        __config__=ConfigDict(
            extra="forbid",
            json_schema_extra={"example": example_payload} if example_payload else None,
        ),
        **fields,
    )
    return CommandSpec(model=model, actions=actions, example_payload=example_payload)


def _build_command_specs() -> dict[str, CommandSpec]:
    parser = build_parser()
    subparsers = _extract_subparsers(parser)
    specs: dict[str, CommandSpec] = {}
    command_defaults = CLI_DEFAULTS
    for command_name in SUPPORTED_COMMANDS:
        subparser = subparsers.choices.get(command_name)
        if subparser is None:
            raise RuntimeError(f"Command parser not found for '{command_name}'.")
        specs[command_name] = _build_command_spec(
            subparser=subparser,
            config_defaults=command_defaults.get(command_name, {}),
        )
    return specs


COMMAND_SPECS = _build_command_specs()

app = FastAPI(
    title="RAG Agent Command API",
    description=(
        "REST wrapper over primary CLI commands with full flag coverage.\n\n"
        "Swagger UI: `/docs`\n"
        "OpenAPI schema: `/openapi.json`"
    ),
    version="0.1.0",
    openapi_tags=[
        {"name": "commands", "description": "Execute primary CLI workflows via REST."},
        {"name": "meta", "description": "Service health and metadata endpoints."},
    ],
)


def _build_argv(command: str, payload: BaseModel, actions: dict[str, argparse.Action]) -> list[str]:
    argv = [command]
    values = payload.model_dump(exclude_unset=True)
    for dest, value in values.items():
        action = actions[dest]
        if isinstance(action, argparse._StoreTrueAction):
            if value:
                argv.append(action.option_strings[0])
            continue
        if value is None:
            continue
        argv.extend([action.option_strings[0], str(value)])
    return argv


def execute_cli_command(command: str, payload: BaseModel) -> CommandResponse:
    parser = build_parser()
    spec = COMMAND_SPECS[command]
    argv = _build_argv(command, payload, spec.actions)

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    try:
        args = parser.parse_args(argv)
        config_path = Path.cwd() / DEFAULT_CLI_PARAMS_CONFIG
        config_defaults = load_cli_defaults(config_path)
        apply_config_defaults(parser, args, argv, config_defaults)
        validate_required_command_params(parser, args, REQUIRED_COMMAND_PARAMS)
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            args.handler(args)
    except SystemExit as exc:
        raise HTTPException(
            status_code=400,
            detail={"command": command, "argv": argv, "error": f"Argument parsing failed: {exc}"},
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail={"command": command, "argv": argv, "error": str(exc)},
        ) from exc

    stdout = stdout_buffer.getvalue()
    stderr = stderr_buffer.getvalue()
    parsed_json: dict[str, Any] | None = None
    if stdout.strip().startswith("{"):
        try:
            parsed = json.loads(stdout)
            if isinstance(parsed, dict):
                parsed_json = parsed
        except json.JSONDecodeError:
            parsed_json = None

    return CommandResponse(
        command=command,
        argv=argv,
        stdout=stdout,
        stderr=stderr,
        result=parsed_json,
    )


def _register_command_route(command: str, spec: CommandSpec) -> None:
    body = (
        Body(..., examples={"default": {"summary": "Config defaults", "value": spec.example_payload}})
        if spec.example_payload
        else Body(...)
    )

    def _run(payload: BaseModel = body) -> CommandResponse:
        return execute_cli_command(command, payload)

    _run.__annotations__ = {"payload": spec.model, "return": CommandResponse}
    app.post(
        f"/{command}",
        response_model=CommandResponse,
        tags=["commands"],
        operation_id=f"run_{command}",
        summary=f"Run `{command}` command",
        description=(
            f"Executes the `{command}` CLI command. "
            "Request body fields map directly to command flags."
        ),
    )(_run)


for command_name, command_spec in COMMAND_SPECS.items():
    _register_command_route(command_name, command_spec)


@app.get("/health", tags=["meta"])
def health() -> dict[str, str]:
    return {"status": "ok"}

