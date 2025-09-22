defmodule ReqLLM.Providers.Mistral do
  @moduledoc """
  Mistral provider with optimized structured output support for Mistral's language models.

  ## Protocol Usage

  Uses the generic `ReqLLM.Context.Codec` and `ReqLLM.Response.Codec` protocols.
  Implements Mistral's native structured output approach for `generate_object/4` operations.

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

      # Structured object generation (uses Mistral's native approach)
      schema = [name: [type: :string, required: true], age: [type: :pos_integer, required: true]]
      {:ok, response} = ReqLLM.generate_object(model, "Generate a person", schema)
  """

  @behaviour ReqLLM.Provider

  use ReqLLM.Provider.DSL,
    id: :mistral,
    base_url: "https://api.mistral.ai/v1",
    metadata: "priv/models_dev/mistral.json",
    default_env_key: "MISTRAL_API_KEY"

  #use ReqLLM.Provider.Defaults

  @impl ReqLLM.Provider
  def prepare_request(:object, model_spec, prompt, opts) do
    compiled_schema = Keyword.fetch!(opts, :compiled_schema)

    structured_output_tool =
      ReqLLM.Tool.new!(
        name: "structured_output",
        description: "Generate structured output matching the provided schema",
        parameter_schema: compiled_schema.schema,
        callback: fn _args -> {:ok, "structured output generated"} end
      )

    json_schema = structured_output_tool
      |> ReqLLM.Tool.to_schema(:openai)
      |> get_in(["function", "parameters", "properties"])
      |> Jason.encode!(pretty: true)

    # Mistral's recommended system prompt with schema
    schema_instruction = """
    Your output should be an instance of a JSON object following this schema: #{json_schema}

    Please ensure your response is valid JSON that strictly adheres to the provided schema.
    """

    # Add schema instruction to system prompt or create one if none exists
    enhanced_opts =
      case opts[:system_prompt] do
        nil ->
          Keyword.put(opts, :system_prompt, schema_instruction)
        existing_system ->
          Keyword.put(opts, :system_prompt, existing_system <> "\n\n" <> schema_instruction)
      end
      |> Keyword.put(:response_format, %{type: "json_object"})
      |> Keyword.put_new(:max_tokens, 4096)
      |> Keyword.put(:operation, :object)

    # Use regular chat preparation with enhanced prompts
    ReqLLM.Provider.Defaults.prepare_request(__MODULE__, :chat, model_spec, prompt, enhanced_opts)
  end

  # Delegate all other operations to defaults
  def prepare_request(operation, model_spec, prompt, opts) do
    ReqLLM.Provider.Defaults.prepare_request(__MODULE__, operation, model_spec, prompt, opts)
  end

  # Custom decode for object responses when using Mistral's JSON format
  @impl ReqLLM.Provider
  def decode_response({request, response}) do
    case request.options[:operation] do
      :object ->
        decode_object_response({request, response})
      _ ->
        ReqLLM.Provider.Defaults.default_decode_response({request, response})
    end
  end

  # Parse JSON response directly instead of extracting from tool calls
  defp decode_object_response({request, %Req.Response{status: status} = response}) when status in 200..299 do
    with {_req, %Req.Response{body: decoded_response}} <-
           ReqLLM.Provider.Defaults.default_decode_response({request, response}),
         content when is_list(content) <- decoded_response.message.content,
         text_content <- Enum.find(content, &(&1.type == :text)),
         {:ok, parsed_object} <- Jason.decode(text_content.text) do

      # Add the parsed object to the response
      enhanced_response = %{decoded_response | object: parsed_object}
      {request, %{response | body: enhanced_response}}
    else
      {:error, %Jason.DecodeError{}} ->
        # Fall back to default tool calling if JSON parsing fails
        ReqLLM.Provider.Defaults.default_decode_response({request, response})
      error ->
        error
    end
  end

  defp decode_object_response({request, response}) do
    # Handle error responses with default behavior
    ReqLLM.Provider.Defaults.default_decode_response({request, response})
  end
end
