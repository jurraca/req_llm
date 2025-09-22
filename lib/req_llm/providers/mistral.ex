defmodule ReqLLM.Providers.Mistral do
  @moduledoc """
  Mistral provider – 100% OpenAI Chat Completions compatible with Mistral's language models.

  ## Protocol Usage

  Uses the generic `ReqLLM.Context.Codec` and `ReqLLM.Response.Codec` protocols.
  No custom wrapper modules – leverages the standard OpenAI-compatible codecs.

  ## Mistral Models

  Supports all Mistral models including:
  - Text generation models (Mistral Large, Medium, Small, Nemo)
  - Code generation models (Codestral, Devstral)
  - Reasoning models (Magistral series)
  - Efficient models (Ministral 3B/8B, Mixtral 8x7B/8x22B)

  Note: This provider excludes image-capable models (Pixtral series and vision variants).

  ## Configuration

      # Add to .env file (automatically loaded)
      MISTRAL_API_KEY=your_api_key_here

  ## Examples

      # Simple text generation
      model = ReqLLM.Model.from("mistral:mistral-large-latest")
      {:ok, response} = ReqLLM.generate_text(model, "Hello!")

      # Streaming
      {:ok, stream} = ReqLLM.stream_text(model, "Tell me a story", stream: true)

      # Tool calling
      tools = [%ReqLLM.Tool{name: "get_weather", ...}]
      {:ok, response} = ReqLLM.generate_text(model, "What's the weather?", tools: tools)

      # Using reasoning models
      model = ReqLLM.Model.from("mistral:magistral-medium-latest")
      {:ok, response} = ReqLLM.generate_text(model, "Solve this logic puzzle...")

      # Code generation
      model = ReqLLM.Model.from("mistral:codestral-latest")
      {:ok, response} = ReqLLM.generate_text(model, "Write a Python function to sort a list")
  """

  @behaviour ReqLLM.Provider

  use ReqLLM.Provider.DSL,
    id: :mistral,
    base_url: "https://api.mistral.ai/v1",
    metadata: "priv/models_dev/mistral.json",
    default_env_key: "MISTRAL_API_KEY"

  use ReqLLM.Provider.Defaults

  # No custom implementations needed - Mistral is fully OpenAI-compatible
  # All standard operations (chat, embedding, streaming, tool calling) work with defaults
end