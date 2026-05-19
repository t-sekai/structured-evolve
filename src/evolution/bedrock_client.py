"""Small AWS Bedrock runtime wrapper for schedule evolution."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.evolution.config import config_value, load_api_config


@dataclass(frozen=True)
class BedrockConfig:
    """Runtime settings for one Bedrock inference request."""

    model_id: str
    region_name: str
    temperature: float = 0.7
    max_tokens: int = 4096
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_session_token: str | None = None
    aws_bedrock_bearer_token: str | None = None


class BedrockClient:
    """Invoke a Bedrock-hosted chat model and return text."""

    def __init__(self, config: BedrockConfig):
        self.config = config
        try:
            import boto3
        except ImportError as err:  # pragma: no cover - depends on optional install
            raise RuntimeError(
                "boto3 is required for --use-bedrock. Install it with "
                "`.venv/bin/python -m pip install boto3`."
            ) from err

        if config.aws_bedrock_bearer_token:
            os.environ["AWS_BEARER_TOKEN_BEDROCK"] = config.aws_bedrock_bearer_token

        client_kwargs = {"region_name": config.region_name}
        if config.aws_access_key_id:
            client_kwargs["aws_access_key_id"] = config.aws_access_key_id
        if config.aws_secret_access_key:
            client_kwargs["aws_secret_access_key"] = config.aws_secret_access_key
        if config.aws_session_token:
            client_kwargs["aws_session_token"] = config.aws_session_token

        self._client = boto3.client("bedrock-runtime", **client_kwargs)

    @classmethod
    def from_env(
        cls,
        *,
        model_id: str | None,
        region_name: str | None,
        temperature: float,
        max_tokens: int,
    ) -> "BedrockClient":
        resolved_model_id = model_id or os.environ.get("BEDROCK_MODEL_ID")
        if not resolved_model_id:
            raise ValueError("Set --bedrock-model-id or BEDROCK_MODEL_ID.")

        resolved_region = (
            region_name
            or os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
            or "us-west-2"
        )
        return cls(
            BedrockConfig(
                model_id=resolved_model_id,
                region_name=resolved_region,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )

    @classmethod
    def from_sources(
        cls,
        *,
        config_path: Path | None,
        model_id: str | None,
        region_name: str | None,
        temperature: float | None,
        max_tokens: int | None,
    ) -> "BedrockClient":
        """Create a client from CLI args, config file, and environment variables."""
        config = load_api_config(config_path) if config_path is not None else {}

        resolved_model_id = (
            model_id
            or config_value(config, "bedrock", "model_id")
            or os.environ.get("BEDROCK_MODEL_ID")
        )
        if not resolved_model_id:
            raise ValueError(
                "Set --bedrock-model-id, bedrock.model_id in config, or BEDROCK_MODEL_ID."
            )

        resolved_region = (
            region_name
            or config_value(config, "aws", "region")
            or os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
            or "us-west-2"
        )
        resolved_temperature = _coalesce_float(
            temperature,
            config_value(config, "bedrock", "temperature"),
            0.7,
        )
        resolved_max_tokens = _coalesce_int(
            max_tokens,
            config_value(config, "bedrock", "max_tokens"),
            4096,
        )

        return cls(
            BedrockConfig(
                model_id=str(resolved_model_id),
                region_name=str(resolved_region),
                temperature=resolved_temperature,
                max_tokens=resolved_max_tokens,
                aws_access_key_id=(
                    config_value(config, "aws", "access_key_id")
                    or os.environ.get("AWS_ACCESS_KEY_ID")
                ),
                aws_secret_access_key=(
                    config_value(config, "aws", "secret_access_key")
                    or os.environ.get("AWS_SECRET_ACCESS_KEY")
                ),
                aws_session_token=(
                    config_value(config, "aws", "session_token")
                    or os.environ.get("AWS_SESSION_TOKEN")
                ),
                aws_bedrock_bearer_token=(
                    config_value(config, "aws", "bedrock_bearer_token")
                    or os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
                ),
            )
        )

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        """Generate text using a few Bedrock-compatible request shapes."""
        body = _converse_like_body(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        response = self._client.invoke_model(
            modelId=self.config.model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        payload = json.loads(response["body"].read())
        return _extract_text(payload)


def _converse_like_body(
    *,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    """Build a broadly compatible chat-style Bedrock request body.

    Bedrock providers differ in exact schemas. This shape works for models that
    accept OpenAI-style chat payloads through Bedrock. If the selected model id
    needs a provider-specific schema, this function is the only place to adjust.
    """
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


def _extract_text(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]

    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message", {})
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )

    output = payload.get("output")
    if isinstance(output, dict):
        message = output.get("message", {})
        content = message.get("content")
        if isinstance(content, list):
            return "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )

    content = payload.get("content")
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))

    raise ValueError(f"Could not extract generated text from Bedrock response keys: {sorted(payload)}")


def _coalesce_float(*values: Any) -> float:
    for value in values:
        if value is not None:
            return float(value)
    raise ValueError("No float value provided")


def _coalesce_int(*values: Any) -> int:
    for value in values:
        if value is not None:
            return int(value)
    raise ValueError("No int value provided")
