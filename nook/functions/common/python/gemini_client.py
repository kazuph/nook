"""
Gemini API Client for Lambda functions.

This module provides a common interface for interacting with the Gemini API.
"""

import os
import time
from dataclasses import dataclass
from typing import Any, List

from google import genai
from google.genai import types
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential
from google.genai.errors import ClientError, TransportError, DeadlineExceeded, ServerError
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GeminiClientConfig:
    """Configuration for the Gemini client."""

    model: str = "gemini-2.0-flash"
    fallback_model: str = "gemini-2.0-flash-lite"
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 40
    max_output_tokens: int = 8192
    response_mime_type: str = "text/plain"
    timeout: int = 60000
    use_search: bool = False
    # デフォルトでは無効化（既存コードとの互換性のため）
    enable_model_fallback: bool = False
    # リトライ時の待機時間（秒）
    retry_backoff_times: List[int] = None
    
    def __post_init__(self):
        if self.retry_backoff_times is None:
            self.retry_backoff_times = [5, 15, 30]

    def update(self, **kwargs) -> None:
        """Update the configuration with the given keyword arguments."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid configuration key: {key}")


class GeminiClient:
    """Client for interacting with the Gemini API."""

    def __init__(self, config: GeminiClientConfig | None = None, **kwargs):
        """
        Initialize the Gemini client.

        Parameters
        ----------
        config : GeminiClientConfig | None
            Configuration for the Gemini client.
            If not provided, default values will be used.
        """
        self._api_key = os.environ.get("GEMINI_API_KEY")
        if not self._api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")

        self._config = config or GeminiClientConfig()
        self._config.update(**kwargs)

        self._client = genai.Client(
            api_key=self._api_key,
            http_options=types.HttpOptions(timeout=self._config.timeout),
        )
        self._chat = None

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception(lambda e: isinstance(e, ClientError)),  # ClientErrorを条件に
        before_sleep=lambda retry_state: logger.info(f"Retrying due to {retry_state.outcome.exception()}...")
    )
    def generate_content(
        self,
        contents: str | list[str],
        system_instruction: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        max_output_tokens: int | None = None,
        response_mime_type: str | None = None,
        use_fallback: bool | None = None,
    ) -> str:
        """
        Generate content using the Gemini API.

        Parameters
        ----------
        contents : str | list[str]
            The content to generate from.
        system_instruction : str | None
            The system instruction to use.
        model : str | None
            The model to use.
            If not provided, the model from the config will be used.
        temperature : float | None
            The temperature to use.
            If not provided, the temperature from the config will be used.
        top_p : float | None
            The top_p to use.
            If not provided, the top_p from the config will be used.
        top_k : int | None
            The top_k to use.
            If not provided, the top_k from the config will be used.
        max_output_tokens : int | None
            The max_output_tokens to use.
            If not provided, the max_output_tokens from the config will be used.
        response_mime_type : str | None
            The response_mime_type to use.
            If not provided, the response_mime_type from the config will be used.
        use_fallback : bool | None
            Whether to use the fallback model if the primary model fails.
            If not provided, the enable_model_fallback from the config will be used.

        Returns
        -------
        str
            The generated content.
        """
        # モデルフォールバックが有効な場合はフォールバックロジックを使用
        should_use_fallback = use_fallback if use_fallback is not None else self._config.enable_model_fallback
        if should_use_fallback:
            try:
                return self.generate_content_with_fallback(
                    contents=contents,
                    system_instruction=system_instruction,
                    model=model,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    max_output_tokens=max_output_tokens,
                    response_mime_type=response_mime_type,
                )
            except Exception as e:
                logger.warning(f"Fallback logic failed: {str(e)}. Falling back to legacy retry logic.")
                # フォールバックロジックが失敗した場合は、以下の従来のロジックを使用
        
        # 以下は従来のロジック
        if isinstance(contents, str):
            contents = [contents]

        config_params = {
            "temperature": temperature or self._config.temperature,
            "top_p": top_p or self._config.top_p,
            "top_k": top_k or self._config.top_k,
            "max_output_tokens": max_output_tokens or self._config.max_output_tokens,
            "response_mime_type": response_mime_type or self._config.response_mime_type,
            "safety_settings": self._get_default_safety_settings(),
        }

        if system_instruction:
            config_params["system_instruction"] = system_instruction

        response = self._client.models.generate_content(
            model=model or self._config.model,
            contents=contents,
            config=types.GenerateContentConfig(**config_params),
        )

        return response.candidates[0].content.parts[0].text

    def generate_content_with_fallback(
        self,
        contents: str | list[str],
        system_instruction: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        max_output_tokens: int | None = None,
        response_mime_type: str | None = None,
    ) -> str:
        """
        Generate content using the Gemini API with model fallback and custom retry logic.

        Attempts to use the primary model first. If that fails, tries the fallback model.
        If both fail, implements a custom retry strategy with increasing sleep times.

        Parameters are the same as generate_content.

        Returns
        -------
        str
            The generated content.
        """
        # モデルの準備 - デフォルト/引数から取得
        primary_model = model or self._config.model
        fallback_model = self._config.fallback_model

        # 最初の試行 - プライマリモデル
        try:
            logger.info(f"Trying with primary model: {primary_model}")
            return self._generate_content_internal(
                contents=contents,
                system_instruction=system_instruction,
                model=primary_model,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_output_tokens=max_output_tokens,
                response_mime_type=response_mime_type,
            )
        except (ClientError, TransportError, DeadlineExceeded, ServerError) as e:
            logger.warning(f"Primary model {primary_model} failed: {str(e)}. Trying fallback model.")

        # 2回目の試行 - フォールバックモデル
        try:
            logger.info(f"Trying with fallback model: {fallback_model}")
            return self._generate_content_internal(
                contents=contents,
                system_instruction=system_instruction,
                model=fallback_model,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_output_tokens=max_output_tokens,
                response_mime_type=response_mime_type,
            )
        except (ClientError, TransportError, DeadlineExceeded, ServerError) as e:
            logger.warning(f"Fallback model {fallback_model} also failed: {str(e)}. Starting retry sequence.")

        # リトライシーケンス - 指定された待機時間で交互にモデルを試す
        last_exception = None

        for wait_time in self._config.retry_backoff_times:
            for current_model in [primary_model, fallback_model]:
                try:
                    logger.info(f"Retrying after {wait_time}s sleep with model: {current_model}")
                    time.sleep(wait_time)
                    return self._generate_content_internal(
                        contents=contents,
                        system_instruction=system_instruction,
                        model=current_model,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        max_output_tokens=max_output_tokens,
                        response_mime_type=response_mime_type,
                    )
                except (ClientError, TransportError, DeadlineExceeded, ServerError) as e:
                    logger.warning(f"Retry with {current_model} after {wait_time}s sleep failed: {str(e)}")
                    last_exception = e

        # すべてのリトライが失敗した場合、最後に発生した例外を再度発生させる
        if last_exception:
            raise last_exception
        raise RuntimeError("All retry attempts failed without specific exception")

    def _generate_content_internal(
        self, 
        contents: str | list[str],
        system_instruction: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        max_output_tokens: int | None = None,
        response_mime_type: str | None = None,
    ) -> str:
        """Internal method to generate content without retry logic."""
        if isinstance(contents, str):
            contents = [contents]

        config_params = {
            "temperature": temperature or self._config.temperature,
            "top_p": top_p or self._config.top_p,
            "top_k": top_k or self._config.top_k,
            "max_output_tokens": max_output_tokens or self._config.max_output_tokens,
            "response_mime_type": response_mime_type or self._config.response_mime_type,
            "safety_settings": self._get_default_safety_settings(),
        }

        if system_instruction:
            config_params["system_instruction"] = system_instruction

        response = self._client.models.generate_content(
            model=model or self._config.model,
            contents=contents,
            config=types.GenerateContentConfig(**config_params),
        )

        return response.candidates[0].content.parts[0].text

    def create_chat(
        self,
        model: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        max_output_tokens: int | None = None,
    ) -> None:
        """
        Create a new chat session.

        Parameters
        ----------
        model : str | None
            The model to use.
            If not provided, the model from the config will be used.
        temperature : float | None
            The temperature to use.
            If not provided, the temperature from the config will be used.
        top_p : float | None
            The top_p to use.
            If not provided, the top_p from the config will be used.
        top_k : int | None
            The top_k to use.
            If not provided, the top_k from the config will be used.
        max_output_tokens : int | None
            The max_output_tokens to use.
            If not provided, the max_output_tokens from the config will be used.
        """
        config_params = {
            "temperature": temperature or self._config.temperature,
            "top_p": top_p or self._config.top_p,
            "top_k": top_k or self._config.top_k,
            "max_output_tokens": max_output_tokens or self._config.max_output_tokens,
            "response_modalities": ["TEXT"],
        }

        if self._config.use_search:
            google_search_tool = types.Tool(google_search=types.GoogleSearch())
            config_params["tools"] = [google_search_tool]

        self._chat = self._client.chats.create(
            model=model or self._config.model,
            config=types.GenerateContentConfig(**config_params),
        )

    def send_message(self, message: str) -> str:
        """
        Send a message to the chat and get the response.

        Parameters
        ----------
        message : str
            The message to send.

        Returns
        -------
        str
            The response from the chat.

        Raises
        ------
        ValueError
            If no chat has been created.
        """
        if not self._chat:
            raise ValueError("No chat has been created. Call create_chat() first.")

        response = self._chat.send_message(message)
        return response.text

    def chat_with_search(self, message: str, model: str | None = None) -> str:
        """
        Create a new chat with search capability and send a message.

        This is a convenience method that combines create_chat() and send_message().

        Parameters
        ----------
        message : str
            The message to send.
        model : str | None
            The model to use.
            If not provided, the model from the config will be used.

        Returns
        -------
        str
            The response from the chat.
        """
        original_use_search = self._config.use_search
        self._config.use_search = True

        try:
            self.create_chat(model=model)
            return self.send_message(message)
        finally:
            self._config.use_search = original_use_search

    def _get_default_safety_settings(self) -> list[types.SafetySetting]:
        """
        Get the default safety settings.

        Returns
        -------
        list[types.SafetySetting]
            The default safety settings.
        """
        return [
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
        ]


def create_client(config: dict[str, Any] | None = None, **kwargs) -> GeminiClient:
    """
    Create a Gemini client with the given configuration.

    Parameters
    ----------
    config : dict[str, Any] | None
        Configuration for the Gemini client.
        If not provided, default values will be used.

    Returns
    -------
    GeminiClient
        The Gemini client.
    """
    if config:
        client_config = GeminiClientConfig(
            model=config.get("model", "gemini-2.0-flash"),
            temperature=config.get("temperature", 1.0),
            top_p=config.get("top_p", 0.95),
            top_k=config.get("top_k", 40),
            max_output_tokens=config.get("max_output_tokens", 8192),
            response_mime_type=config.get("response_mime_type", "text/plain"),
            timeout=config.get("timeout", 60000),
            use_search=config.get("use_search", False),
        )
    else:
        client_config = None

    return GeminiClient(client_config, **kwargs)
