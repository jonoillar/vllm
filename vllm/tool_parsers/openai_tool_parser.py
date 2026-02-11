# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from collections.abc import Sequence

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaMessage,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.parser.harmony_utils import parse_output_into_messages
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import (
    ToolParser,
)

logger = init_logger(__name__)


class OpenAIToolParser(ToolParser):
    """
    Tool parser for GPT-OSS Harmony models.

    Supports tool_choice="required" by using bad_words token sequences
    to block non-tool-call generation paths.
    """

    # Harmony special token names
    HARMONY_END_TOKEN = "<|end|>"
    HARMONY_START_TOKEN = "<|start|>"
    HARMONY_CHANNEL_TOKEN = "<|channel|>"
    HARMONY_MESSAGE_TOKEN = "<|message|>"

    def __init__(self, tokenizer: TokenizerLike):
        super().__init__(tokenizer)
        # Cache special token IDs from vocab (dynamic lookup, not hardcoded)
        self._harmony_token_ids = self._get_harmony_token_ids()
        # Cache channel name token IDs
        self._channel_token_ids = self._get_channel_token_ids()

    def _get_harmony_token_ids(self) -> dict[str, int | None]:
        """Get Harmony special token IDs from vocabulary."""
        vocab = self.vocab
        return {
            "end": vocab.get(self.HARMONY_END_TOKEN),
            "start": vocab.get(self.HARMONY_START_TOKEN),
            "channel": vocab.get(self.HARMONY_CHANNEL_TOKEN),
            "message": vocab.get(self.HARMONY_MESSAGE_TOKEN),
            "assistant": self._encode_single_token("assistant"),
        }

    def _get_channel_token_ids(self) -> dict[str, int | list[int] | None]:
        """Get channel name token IDs."""
        return {
            "final": self._encode_single_token("final"),
            "analysis": self._encode_single_token("analysis"),
            " final": self._encode_single_token(" final"),
            " analysis": self._encode_single_token(" analysis"),
            "commentary": self._encode_tokens("commentary"),
        }

    def _encode_single_token(self, text: str) -> int | None:
        """Encode text and return token ID if it's a single token."""
        try:
            ids = self.model_tokenizer.encode(text, add_special_tokens=False)
            return ids[0] if len(ids) == 1 else None
        except Exception:
            return None

    def _encode_tokens(self, text: str) -> list[int] | None:
        """Encode text and return all token IDs."""
        try:
            ids = self.model_tokenizer.encode(text, add_special_tokens=False)
            return ids if ids else None
        except Exception:
            return None

    def _build_bad_words_sequences(self) -> list[list[int]]:
        """
        Build bad_words token sequences to block non-tool-call paths.

        Blocks:
        - <|end|><|start|>assistant<|channel|>final (direct final response)
        - <|end|><|start|>assistant<|channel|>analysis (new analysis message)
        - commentary<|message|> (commentary without recipient = not a tool call)

        This allows first message to use analysis channel for reasoning,
        but forces subsequent messages to use commentary with recipient (tool call).
        """
        bad_sequences: list[list[int]] = []

        end_id = self._harmony_token_ids.get("end")
        start_id = self._harmony_token_ids.get("start")
        assistant_id = self._harmony_token_ids.get("assistant")
        channel_id = self._harmony_token_ids.get("channel")
        message_id = self._harmony_token_ids.get("message")

        # Validate required tokens exist
        if (
            end_id is None
            or start_id is None
            or assistant_id is None
            or channel_id is None
        ):
            logger.warning(
                "Missing Harmony special tokens in vocabulary. "
                "tool_choice='required' may not work correctly."
            )
            return []

        # Common prefix for new assistant message (all validated as int above)
        new_msg_prefix: list[int] = [end_id, start_id, assistant_id, channel_id]

        # Block transitions to final/analysis channels
        for channel_key in ["final", "analysis", " final", " analysis"]:
            channel_token = self._channel_token_ids.get(channel_key)
            if isinstance(channel_token, int):
                seq = new_msg_prefix + [channel_token]
                bad_sequences.append(seq)
                logger.debug("Blocking new message to %s channel: %s", channel_key, seq)

        # Block commentary without recipient (not a tool call)
        commentary_tokens = self._channel_token_ids.get("commentary")
        if isinstance(commentary_tokens, list) and message_id is not None:
            seq = commentary_tokens + [message_id]
            bad_sequences.append(seq)
            logger.debug("Blocking commentary without recipient: %s", seq)

        return bad_sequences

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        """
        Adjust request for GPT-OSS tool_choice="required" support.

        For tool_choice="required", builds bad_words token sequences
        and stores them on the request for later application to SamplingParams.
        """
        if not request.tools:
            return request

        # For tool_choice != "required", use default behavior
        if request.tool_choice != "required":
            return super().adjust_request(request)

        logger.debug("GPT-OSS tool_choice=required: building bad_words sequences")

        bad_sequences = self._build_bad_words_sequences()

        if bad_sequences:
            # Store as a temporary attribute on the request object
            # This avoids misusing vllm_xargs which is meant for user input
            request._tool_parser_bad_words_token_ids = bad_sequences  # type: ignore[attr-defined]
            logger.debug(
                "Stored %d bad_words sequences for tool_choice=required",
                len(bad_sequences),
            )

        return request

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
        token_ids: Sequence[int] | None = None,
    ) -> ExtractedToolCallInformation:
        if token_ids is None:
            raise NotImplementedError(
                "OpenAIToolParser requires token IDs and does not support "
                "text-based extraction."
            )

        parser = parse_output_into_messages(token_ids)
        tool_calls = []
        final_content = None
        commentary_content = None

        if len(parser.messages) > 0:
            for msg in parser.messages:
                if len(msg.content) < 1:
                    continue
                msg_text = msg.content[0].text
                if msg.recipient and msg.recipient.startswith("functions."):
                    # If no content-type is given assume JSON, as that's the
                    # most common case with gpt-oss models.
                    if not msg.content_type or "json" in msg.content_type:
                        # load and dump the JSON text to check validity and
                        # remove any extra newlines or other odd formatting
                        try:
                            tool_args = json.dumps(json.loads(msg_text))
                        except json.JSONDecodeError:
                            logger.exception(
                                "Error decoding JSON tool call from response."
                            )
                            tool_args = msg_text
                    else:
                        tool_args = msg_text
                    tool_calls.append(
                        ToolCall(
                            type="function",
                            function=FunctionCall(
                                name=msg.recipient.split("functions.")[1],
                                arguments=tool_args,
                            ),
                        )
                    )
                elif msg.channel == "final":
                    final_content = msg_text
                elif msg.channel == "commentary" and not msg.recipient:
                    commentary_content = msg_text

        # Extract partial content from the parser state if generation was truncated
        if parser.current_content:
            if parser.current_channel == "final":
                final_content = parser.current_content
            elif (
                parser.current_channel == "commentary" and not parser.current_recipient
            ):
                commentary_content = parser.current_content

        return ExtractedToolCallInformation(
            tools_called=len(tool_calls) > 0,
            tool_calls=tool_calls,
            # prefer final content over commentary content if both are present
            # commentary content is tool call preambles meant to be shown to the user
            content=final_content or commentary_content,
        )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        raise NotImplementedError("Not being used, manual parsing in serving.py")
