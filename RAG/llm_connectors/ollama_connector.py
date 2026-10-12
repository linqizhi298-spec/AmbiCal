# llm_connectors/ollama_connector.py
import json
import time
from typing import Dict, Any, Mapping
import ollama
from llm_connectors.base_llm_connector import (
    BaseLLMConnector,
    TokenUsageUnavailableError,
)
from ollama._types import ResponseError


class OllamaConnector(BaseLLMConnector):
    """Connector for Ollama LLMs."""

    def __init__(self, model_name: str, config: Dict[str, Any]):
        """
        Initializes the OllamaConnector.

        Args:
            model_name (str): The *key* of the model in the config (can be a user-defined alias).
            config (Dict[str, Any]): Configuration parameters for the Ollama model.
                                    Should include 'name' (the *actual* Ollama model name),
                                    and optionally 'temperature', 'num_predict', and 'context_window'.
        """
        super().__init__(model_name, config)  # model_name is the *key*
        # config["name"] is the *actual* Ollama model name.
        if "name" not in config:
            raise ValueError("Ollama config must include a 'name' key with the actual model name.")

        self.ollama_model_name = config["name"]  # Store the *actual* model name
        self.model_configs = config
        self.ollama_client = ollama.Client(host=config.get("host"))

    @staticmethod
    def _response_value(response: Any, key: str) -> Any:
        """Read a value from either an Ollama response object or a mapping."""

        value = getattr(response, key, None)
        if value is None and isinstance(response, Mapping):
            value = response.get(key)
        return value

    def _response_audit_metadata(self, response: Any) -> Dict[str, Any]:
        """Return non-content response fields safe to include in an error log."""

        keys = (
            "model",
            "done",
            "done_reason",
            "prompt_eval_count",
            "prompt_eval_cached_count",
            "prompt_eval_duration",
            "eval_count",
            "eval_duration",
            "total_duration",
            "load_duration",
        )
        metadata = {key: self._response_value(response, key) for key in keys}
        response_text = self._response_value(response, "response")
        metadata["response_character_count"] = (
            len(str(response_text)) if response_text is not None else None
        )
        metadata["response_type"] = type(response).__name__
        return metadata

    def invoke(self, prompt: str) -> str:
        """
        Invokes the Ollama LLM with the given prompt and returns the response.
        Implements retry logic for handling potential connection errors.

        Args:
            prompt (str): The prompt to send to the Ollama model.

        Returns:
            str: The Ollama model's response.
        """
        # retries = 5
        # wait_times = [5, 25, 125, 625, 3125]  # Exponential backoff in seconds

        retries = 2
        wait_times = [5, 25]  # Exponential backoff in seconds

        for attempt in range(retries):
            self._begin_token_usage_call()
            try:
                options = {
                    "temperature": self.temperature,
                    "num_predict": self.num_predict,
                }
                if self.context_window is not None:
                    options["num_ctx"] = self.context_window
                result_stream = self.ollama_client.generate(
                    model=self.ollama_model_name,
                    prompt=prompt,
                    # Ollama has occasionally returned a non-final response object
                    # even when stream=False.  Explicitly consume the stream so the
                    # connector can require the terminal done=True chunk that owns
                    # the exact prompt/output token counts.
                    stream=True,
                    options=options,
                )
                if hasattr(result_stream, "__next__"):
                    chunks = list(result_stream)
                else:
                    # Keep compatibility with test doubles and defensive support
                    # for clients that return one response despite stream=True.
                    chunks = [result_stream]
                if not chunks:
                    raise TokenUsageUnavailableError(
                        "Ollama streaming response contained no chunks; aborting "
                        "before incomplete token accounting can reach the run "
                        f"manifest. model_alias={self.model_name!r}; actual_model="
                        f"{self.ollama_model_name!r}"
                    )
                result = chunks[-1]
                prompt_tokens = self._response_value(result, "prompt_eval_count")
                output_tokens = self._response_value(result, "eval_count")
                response_parts = [
                    str(value)
                    for chunk in chunks
                    if (value := self._response_value(chunk, "response")) is not None
                ]
                response_text = "".join(response_parts)
                done = self._response_value(result, "done")
                if done is not True or prompt_tokens is None or output_tokens is None:
                    raise TokenUsageUnavailableError(
                        "Ollama response omitted exact prompt_eval_count or "
                        "eval_count, or was not final; aborting immediately before "
                        "incomplete token accounting can reach the run manifest. "
                        f"model_alias={self.model_name!r}; actual_model="
                        f"{self.ollama_model_name!r}; response_metadata="
                        + json.dumps(
                            self._response_audit_metadata(result),
                            ensure_ascii=True,
                            sort_keys=True,
                        )
                    )
                if not response_parts:
                    raise RuntimeError("Ollama response omitted generated text.")
                self._complete_token_usage_call(prompt_tokens, output_tokens)
                return str(response_text)
            except ResponseError as e:
                self._mark_token_usage_unreported()
                if attempt < retries - 1:  # Only retry if not the last attempt
                    wait_time = wait_time = wait_times[min(attempt, len(wait_times) - 1)] # prevent index out of range
                    print(
                        f"Ollama ResponseError: {e}. Retry attempt {attempt + 1}/{retries}. Waiting {wait_time} seconds..."
                    )
                    time.sleep(wait_time)
                else:
                    print(f"Ollama ResponseError: {e}. Max retries reached. Raising error.")
                    raise  # Re-raise the exception after max retries
            except Exception as e: # Catch other potential exceptions
                self._mark_token_usage_unreported()
                print(f"Unexpected error during Ollama invoke: {e}")
                raise # Re-raise unexpected exceptions

if __name__ == "__main__":
    # Example usage (assuming Ollama is running and you have llama3.2 model):
    ollama_config = {
        "name": "llama3.2:1B-128k",  # The *actual* Ollama model name
        "temperature": 0.0,
        "num_predict": 256,
        "context_window": 131072,
    }
    # Use a key that might be different from the actual model name
    ollama_connector = OllamaConnector("llama3.2:1B-128k", ollama_config)

    prompt_text = "What is the capital of France?"
    response = ollama_connector.invoke(prompt_text)
    print(f"Prompt: {prompt_text}")
    print(f"Response from Ollama: {response}")
