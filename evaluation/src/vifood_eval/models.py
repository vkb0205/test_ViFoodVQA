from __future__ import annotations

import base64
import mimetypes
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class VisionModel(ABC):
    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        max_new_tokens: int,
        temperature: float,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        raise NotImplementedError


def make_model(name: str, cfg: dict[str, Any]) -> VisionModel:
    model_type = cfg.get("type")
    if model_type == "openai_compatible":
        return OpenAICompatibleModel(cfg)
    if model_type == "hf":
        return HFVisionModel(cfg)
    raise ValueError(f"Unsupported model type for {name}: {model_type}")


class OpenAICompatibleModel(VisionModel):
    def __init__(self, cfg: dict[str, Any]) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the 'api' extra to use OpenAI-compatible models.") from exc

        api_key = os.getenv(cfg.get("api_key_env", "OPENAI_COMPAT_API_KEY"))
        base_url = os.getenv(cfg.get("base_url_env", "OPENAI_COMPAT_BASE_URL")) or None
        if not api_key:
            raise RuntimeError("Missing OpenAI-compatible API key environment variable.")

        self.model_id = cfg["model_id"]
        client_kwargs: dict[str, Any] = {"api_key": api_key, "base_url": base_url}
        if cfg.get("default_headers"):
            client_kwargs["default_headers"] = cfg["default_headers"]
        self.client = OpenAI(**client_kwargs)
        self.use_response_format = bool(cfg.get("json_response_format", True))
        self.response_format_fallback = bool(cfg.get("json_response_format_fallback", True))

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        max_new_tokens: int,
        temperature: float,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        request: dict[str, Any] = {
            "model": self.model_id,
            "messages": _messages_to_openai(messages),
            "temperature": temperature,
            "max_tokens": max_new_tokens,
        }
        if response_format and self.use_response_format:
            request["response_format"] = response_format

        try:
            response = self.client.chat.completions.create(**request)
        except Exception as exc:
            if (
                "response_format" not in request
                or not self.response_format_fallback
                or not _looks_like_response_format_error(exc)
            ):
                raise
            request.pop("response_format", None)
            response = self.client.chat.completions.create(**request)
        return response.choices[0].message.content or ""


class HFVisionModel(VisionModel):
    def __init__(self, cfg: dict[str, Any]) -> None:
        try:
            import torch
            from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
            try:
                from transformers import AutoModelForImageTextToText
            except ImportError:
                AutoModelForImageTextToText = None
            try:
                from transformers import Qwen3VLForConditionalGeneration
            except ImportError:
                Qwen3VLForConditionalGeneration = None
        except ImportError as exc:
            raise RuntimeError("Install the 'hf' extra to use local Hugging Face models.") from exc

        self.torch = torch
        self.adapter = cfg.get("adapter", "qwen_vl")
        self.use_cache = cfg.get("use_cache", False) if self.adapter == "phi3_vision" else cfg.get("use_cache")
        self.generation_kwargs = dict(cfg.get("generation_kwargs") or {})
        if self.adapter == "phi3_vision":
            _patch_dynamic_cache_legacy_api()

        trust_remote_code = cfg.get("trust_remote_code", True)
        processor_kwargs = {
            "trust_remote_code": trust_remote_code,
        }
        if self.adapter == "phi3_vision":
            processor_kwargs["num_crops"] = cfg.get("num_crops", 16)
        if "processor_use_fast" in cfg:
            processor_kwargs["use_fast"] = cfg["processor_use_fast"]
        if "processor_kwargs" in cfg:
            processor_kwargs.update(cfg["processor_kwargs"])
        self.processor_kwargs = processor_kwargs # save for manual resizing
        self.processor = AutoProcessor.from_pretrained(cfg["model_id"], **processor_kwargs)

        model_config = None
        if cfg.get("load_config_first", False):
            model_config = AutoConfig.from_pretrained(
                cfg["model_id"],
                trust_remote_code=trust_remote_code,
            )
            _force_attention_implementation(model_config, cfg.get("attn_implementation"))
            _force_use_cache(model_config, self.use_cache)

        model_kwargs = {
            "device_map": cfg.get("device_map", "auto"),
            "trust_remote_code": trust_remote_code,
        }
        if model_config is not None:
            model_kwargs["config"] = model_config
        if "attn_implementation" in cfg:
            model_kwargs["attn_implementation"] = cfg["attn_implementation"]
        torch_dtype = cfg.get("torch_dtype")
        if torch_dtype:
            import torch
            if torch_dtype == "float16": torch_dtype = torch.float16
            elif torch_dtype == "bfloat16": torch_dtype = torch.bfloat16
            elif torch_dtype == "float32": torch_dtype = torch.float32
            dtype_key = "dtype" if self.adapter == "qwen3_vl" else "torch_dtype"
            model_kwargs[dtype_key] = torch_dtype
        if "model_kwargs" in cfg:
            model_kwargs.update(cfg["model_kwargs"])

        auto_model = cfg.get("auto_model", "image_text_to_text")
        if self.adapter == "qwen3_vl":
            if Qwen3VLForConditionalGeneration is None:
                raise RuntimeError(
                    "Install a recent transformers build with Qwen3VLForConditionalGeneration "
                    "to use the qwen3_vl adapter."
                )
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(cfg["model_id"], **model_kwargs)
        elif auto_model == "causal_lm":
            self.model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], **model_kwargs)
        elif auto_model == "image_text_to_text":
            if AutoModelForImageTextToText is None:
                self.model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], **model_kwargs)
            else:
                try:
                    self.model = AutoModelForImageTextToText.from_pretrained(cfg["model_id"], **model_kwargs)
                except ValueError:
                    self.model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], **model_kwargs)
        else:
            raise ValueError(f"Unsupported Hugging Face auto_model: {auto_model}")
        _force_use_cache(getattr(self.model, "config", None), self.use_cache)
        self.model.eval()

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        max_new_tokens: int,
        temperature: float,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        _ = response_format
        max_pixels = self.processor_kwargs.get("max_pixels", None) if hasattr(self, "processor_kwargs") else None
        
        if self.adapter == "qwen3_vl":
            inputs = _messages_to_qwen3_inputs(self.processor, messages, max_pixels=max_pixels)
        elif self.adapter == "phi3_vision":
            prompt, images = _messages_to_phi(messages)
            inputs = self.processor(text=prompt, images=images, return_tensors="pt")
        else:
            prompt, images = _messages_to_chat_template(self.processor, messages)
            inputs = self.processor(text=[prompt], images=images, return_tensors="pt")

        inputs = _move_inputs_to_device(inputs, self.model.device)
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
        }
        if self.use_cache is not None:
            generate_kwargs["use_cache"] = bool(self.use_cache)
        if temperature > 0:
            generate_kwargs["temperature"] = temperature
        generate_kwargs.update(_resolve_generation_kwargs(self.generation_kwargs, self.processor))
        if self.adapter == "phi3_vision" and "eos_token_id" not in generate_kwargs:
            eos_token_id = _tokenizer_attr(self.processor, "eos_token_id")
            if eos_token_id is not None:
                generate_kwargs["eos_token_id"] = eos_token_id

        with self.torch.no_grad():
            output_ids = self.model.generate(**inputs, **generate_kwargs)

        new_tokens = _trim_generated_ids(output_ids, inputs["input_ids"])
        decoded = self.processor.batch_decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0].strip()


def _messages_to_openai(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for message in messages:
        parts: list[dict[str, Any]] = []
        for part in message["content"]:
            if part["type"] == "text":
                parts.append({"type": "text", "text": part["text"]})
            elif part["type"] == "image":
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": _image_to_data_url(Path(part["path"]))},
                    }
                )
        converted.append({"role": message["role"], "content": parts})
    return converted


def _looks_like_response_format_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "response_format" in message or "response format" in message


def _force_attention_implementation(config: Any, attn_implementation: object) -> None:
    if not attn_implementation:
        return
    value = str(attn_implementation)
    for attr in [
        "attn_implementation",
        "_attn_implementation",
        "_attn_implementation_internal",
    ]:
        try:
            setattr(config, attr, value)
        except Exception:
            pass
    for attr in ["use_flash_attention_2", "flash_attn_2_enabled"]:
        if hasattr(config, attr):
            try:
                setattr(config, attr, False)
            except Exception:
                pass


def _force_use_cache(config: Any, use_cache: object) -> None:
    if config is None or use_cache is None:
        return
    try:
        setattr(config, "use_cache", bool(use_cache))
    except Exception:
        pass


def _patch_dynamic_cache_legacy_api() -> None:
    try:
        from transformers.cache_utils import DynamicCache
    except ImportError:
        return

    if not hasattr(DynamicCache, "from_legacy_cache"):
        @classmethod
        def from_legacy_cache(cls: type[Any], past_key_values: Any = None) -> Any:
            if past_key_values is None or isinstance(past_key_values, cls):
                return past_key_values if past_key_values is not None else cls()
            return cls(ddp_cache_data=past_key_values)

        DynamicCache.from_legacy_cache = from_legacy_cache

    if not hasattr(DynamicCache, "get_usable_length"):
        def get_usable_length(
            self: Any,
            new_seq_length: int | None = None,
            layer_idx: int = 0,
        ) -> int:
            previous_seq_length = _cache_seq_length(self, layer_idx)
            max_length = _cache_max_length(self, layer_idx)
            if max_length is None or new_seq_length is None:
                return previous_seq_length
            if previous_seq_length + new_seq_length > max_length:
                return max(max_length - new_seq_length, 0)
            return previous_seq_length

        DynamicCache.get_usable_length = get_usable_length

    if not hasattr(DynamicCache, "seen_tokens"):
        DynamicCache.seen_tokens = property(_cache_seen_tokens)

    if not hasattr(DynamicCache, "get_max_length"):
        def get_max_length(self: Any) -> int | None:
            return _cache_legacy_max_length(self, 0)

        DynamicCache.get_max_length = get_max_length

    if not hasattr(DynamicCache, "to_legacy_cache"):
        def to_legacy_cache(self: Any) -> tuple[Any, ...]:
            return _cache_to_legacy(self)

        DynamicCache.to_legacy_cache = to_legacy_cache


def _cache_seen_tokens(cache: Any) -> int:
    return _cache_seq_length(cache, 0)


def _cache_seq_length(cache: Any, layer_idx: int) -> int:
    if not hasattr(cache, "get_seq_length"):
        return 0
    try:
        return int(cache.get_seq_length(layer_idx))
    except TypeError:
        return int(cache.get_seq_length())


def _cache_max_length(cache: Any, layer_idx: int) -> int | None:
    for getter_name in ["get_max_cache_shape", "get_max_length"]:
        if hasattr(cache, getter_name):
            getter = getattr(cache, getter_name)
            try:
                value = getter(layer_idx)
            except TypeError:
                value = getter()
            if value is not None:
                max_length = int(value)
                return max_length if max_length > 0 else None
    value = getattr(cache, "max_cache_len", None)
    if value is None:
        return None
    max_length = int(value)
    return max_length if max_length > 0 else None


def _cache_legacy_max_length(cache: Any, layer_idx: int) -> int | None:
    if hasattr(cache, "get_max_cache_shape"):
        getter = getattr(cache, "get_max_cache_shape")
        try:
            value = getter(layer_idx)
        except TypeError:
            value = getter()
        if value is not None:
            max_length = int(value)
            return max_length if max_length > 0 else None

    for attr in ["max_cache_len", "max_length"]:
        value = getattr(cache, attr, None)
        if value is not None:
            max_length = int(value)
            return max_length if max_length > 0 else None
    return None


def _cache_to_legacy(cache: Any) -> tuple[Any, ...]:
    if hasattr(cache, "ddp_cache_data") and getattr(cache, "ddp_cache_data") is not None:
        return tuple(getattr(cache, "ddp_cache_data"))

    legacy_layers = []
    for layer in getattr(cache, "layers", []):
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if keys is None or values is None:
            continue
        legacy_layers.append((keys, values))
    return tuple(legacy_layers)


def _move_inputs_to_device(inputs: Any, device: Any) -> Any:
    if hasattr(inputs, "to"):
        return inputs.to(device)
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


def _resolve_generation_kwargs(kwargs: dict[str, Any], processor: Any) -> dict[str, Any]:
    resolved = dict(kwargs)
    for key, attr in [
        ("eos_token_id", "eos_token_id"),
        ("pad_token_id", "pad_token_id"),
    ]:
        if resolved.get(key) == "tokenizer":
            token_id = _tokenizer_attr(processor, attr)
            if token_id is not None:
                resolved[key] = token_id
            else:
                resolved.pop(key, None)
    return resolved


def _tokenizer_attr(processor: Any, attr: str) -> Any:
    tokenizer = getattr(processor, "tokenizer", None)
    return getattr(tokenizer, attr, None) if tokenizer is not None else None


def _trim_generated_ids(output_ids: Any, input_ids: Any) -> Any:
    try:
        input_len = input_ids.shape[-1]
        return output_ids[:, input_len:]
    except Exception:
        return [
            output[len(input_id):]
            for input_id, output in zip(input_ids, output_ids)
        ]


def _messages_to_qwen3_inputs(processor: Any, messages: list[dict[str, Any]], max_pixels: int | None = None) -> Any:
    converted = []
    for message in messages:
        content = []
        for part in message["content"]:
            if part["type"] == "text":
                content.append({"type": "text", "text": part["text"]})
            elif part["type"] == "image":
                content.append({"type": "image", "image": _load_image(Path(part["path"]), max_pixels=max_pixels)})
        converted.append({"role": message["role"], "content": content})
    inputs = processor.apply_chat_template(
        converted,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    if hasattr(inputs, "pop"):
        inputs.pop("token_type_ids", None)
    return inputs


def _messages_to_chat_template(processor: Any, messages: list[dict[str, Any]]) -> tuple[str, list[Any]]:
    images = []
    converted = []
    for message in messages:
        content = []
        for part in message["content"]:
            if part["type"] == "text":
                content.append({"type": "text", "text": part["text"]})
            elif part["type"] == "image":
                image = _load_image(Path(part["path"]))
                images.append(image)
                content.append({"type": "image"})
        converted.append({"role": message["role"], "content": content})
    prompt = processor.apply_chat_template(converted, tokenize=False, add_generation_prompt=True)
    return prompt, images


def _messages_to_phi(messages: list[dict[str, Any]]) -> tuple[str, list[Any]]:
    images = []
    chunks: list[str] = []
    image_idx = 1
    for message in messages:
        if message["role"] == "system":
            text = "\n".join(part["text"] for part in message["content"] if part["type"] == "text")
            chunks.append(f"<|system|>\n{text}<|end|>\n")
            continue
        if message["role"] == "assistant":
            text = "\n".join(part["text"] for part in message["content"] if part["type"] == "text")
            chunks.append(f"<|assistant|>\n{text}<|end|>\n")
            continue

        user_parts = ["<|user|>\n"]
        for part in message["content"]:
            if part["type"] == "image":
                images.append(_load_image(Path(part["path"])))
                user_parts.append(f"<|image_{image_idx}|>\n")
                image_idx += 1
            elif part["type"] == "text":
                user_parts.append(part["text"])
        user_parts.append("<|end|>\n")
        chunks.append("".join(user_parts))
    chunks.append("<|assistant|>\n")
    return "".join(chunks), images


def _image_to_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _load_image(path: Path, max_pixels: int | None = None) -> Any:
    from PIL import Image

    image = Image.open(path).convert("RGB")
    if max_pixels is not None:
        w, h = image.size
        if w * h > max_pixels:
            scale = (max_pixels / (w * h)) ** 0.5
            new_w, new_h = int(w * scale), int(h * scale)
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return image
