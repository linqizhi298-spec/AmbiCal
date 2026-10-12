from abc import ABC, abstractmethod
from typing import Dict, Any


class TokenUsageUnavailableError(RuntimeError):
    """A provider call completed without auditable exact token metadata."""


class BaseLLMConnector(ABC):
    """Abstract base class for LLM connectors."""

    @abstractmethod
    def __init__(self, model_name: str, config: Dict[str, Any]):
        """
        Initializes the LLM connector.

        Args:
            model_name (str): The name of the LLM model to use.
            config (Dict[str, Any]): Configuration parameters for the LLM.
        """
        self.model_name = model_name
        self.config = config
        self.temperature = config.get("temperature", 0.0) # Default temperature
        self.num_predict = config.get("num_predict", 256)     # Default max tokens
        self.context_window = config.get("context_window")   # Context window size, might be None or specific value
        self._token_usage = {
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "unreported_calls": 0,
        }

    def _begin_token_usage_call(self) -> None:
        """Register one real provider request before it is attempted."""

        self._token_usage["calls"] += 1

    def _complete_token_usage_call(
        self,
        input_tokens: Any = None,
        output_tokens: Any = None,
        total_tokens: Any = None,
    ) -> None:
        """Finish a previously registered request with provider token metadata."""

        if input_tokens is None or output_tokens is None:
            self._token_usage["unreported_calls"] += 1
            return
        input_count = int(input_tokens)
        output_count = int(output_tokens)
        total_count = (
            int(total_tokens)
            if total_tokens is not None
            else input_count + output_count
        )
        if min(input_count, output_count, total_count) < 0:
            raise ValueError("Token counts must be non-negative.")
        self._token_usage["input_tokens"] += input_count
        self._token_usage["output_tokens"] += output_count
        self._token_usage["total_tokens"] += total_count

    def _mark_token_usage_unreported(self) -> None:
        """Mark a registered provider request whose usage cannot be proven."""

        self._token_usage["unreported_calls"] += 1

    def _record_token_usage(
        self,
        input_tokens: Any = None,
        output_tokens: Any = None,
        total_tokens: Any = None,
    ) -> None:
        """Record an already completed call (compatibility convenience)."""

        self._begin_token_usage_call()
        self._complete_token_usage_call(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )

    def token_usage(self) -> Dict[str, int]:
        return dict(self._token_usage)

    @abstractmethod
    def invoke(self, prompt: str) -> str:
        """
        Invokes the LLM with the given prompt and returns the response.

        Args:
            prompt (str): The prompt to send to the LLM.

        Returns:
            str: The LLM's response.
        """
        pass
