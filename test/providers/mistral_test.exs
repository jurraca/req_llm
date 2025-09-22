defmodule ReqLLM.Providers.MistralTest do
  @moduledoc """
  Provider-level tests for Mistral implementation.

  Tests the provider contract directly without going through Generation layer.
  Focus: prepare_request -> attach -> request -> decode pipeline.
  """

  use ReqLLM.ProviderCase, provider: ReqLLM.Providers.Mistral

  import ReqLLM.ProviderTestHelpers

  alias ReqLLM.Context
  alias ReqLLM.Providers.Mistral

  describe "provider contract" do
    test "provider identity and configuration" do
      assert is_atom(Mistral.provider_id())
      assert is_binary(Mistral.default_base_url())
      assert String.starts_with?(Mistral.default_base_url(), "http")
    end

    test "provider schema separation from core options" do
      schema_keys = Mistral.provider_schema().schema |> Keyword.keys()
      core_keys = ReqLLM.Provider.Options.generation_schema().schema |> Keyword.keys()

      # Provider-specific keys should not overlap with core generation keys
      overlap = MapSet.intersection(MapSet.new(schema_keys), MapSet.new(core_keys))

      assert MapSet.size(overlap) == 0,
             "Schema overlap detected: #{inspect(MapSet.to_list(overlap))}"
    end

    test "supported options include core generation keys" do
      supported = Mistral.supported_provider_options()
      core_keys = ReqLLM.Provider.Options.all_generation_keys()

      # All core keys should be supported (except meta-keys like :provider_options)
      core_without_meta = Enum.reject(core_keys, &(&1 == :provider_options))
      missing = core_without_meta -- supported
      assert missing == [], "Missing core generation keys: #{inspect(missing)}"
    end

    test "provider_extended_generation_schema includes both base and provider options" do
      extended_schema = Mistral.provider_extended_generation_schema()
      extended_keys = extended_schema.schema |> Keyword.keys()

      # Should include all core generation keys
      core_keys = ReqLLM.Provider.Options.all_generation_keys()
      core_without_meta = Enum.reject(core_keys, &(&1 == :provider_options))

      for core_key <- core_without_meta do
        assert core_key in extended_keys,
               "Extended schema missing core key: #{core_key}"
      end

      # Should include provider-specific keys
      provider_keys = Mistral.provider_schema().schema |> Keyword.keys()

      for provider_key <- provider_keys do
        assert provider_key in extended_keys,
               "Extended schema missing provider key: #{provider_key}"
      end
    end
  end

  describe "request preparation & pipeline wiring" do
    test "prepare_request creates configured request" do
      model = ReqLLM.Model.from!("mistral:mistral-large-latest")
      prompt = "Hello world"
      opts = [temperature: 0.7, max_tokens: 100]

      {:ok, request} = Mistral.prepare_request(:chat, model, prompt, opts)

      assert %Req.Request{} = request
      assert String.ends_with?(request.url.path, "/chat/completions")
      assert request.method == :post
    end

    test "attach configures authentication and pipeline" do
      model = ReqLLM.Model.from!("mistral:mistral-large-latest")
      opts = [temperature: 0.5, max_tokens: 50]

      request = Req.new() |> Mistral.attach(model, opts)

      # Verify authentication
      auth_header = Enum.find(request.headers, fn {name, _} -> name == "authorization" end)
      assert auth_header != nil
      {_, [auth_value]} = auth_header
      assert String.starts_with?(auth_value, "Bearer ")

      # Verify pipeline steps
      request_steps = Keyword.keys(request.request_steps)
      response_steps = Keyword.keys(request.response_steps)

      assert :llm_encode_body in request_steps
      assert :llm_decode_response in response_steps
    end

    test "error handling for invalid configurations" do
      model = ReqLLM.Model.from!("mistral:mistral-large-latest")
      prompt = "Hello world"

      # Unsupported operation
      {:error, error} = Mistral.prepare_request(:unsupported, model, prompt, [])
      assert %ReqLLM.Error.Invalid.Parameter{} = error

      # Provider mismatch
      wrong_model = ReqLLM.Model.from!("openai:gpt-4")

      assert_raise ReqLLM.Error.Invalid.Provider, fn ->
        Req.new() |> Mistral.attach(wrong_model, [])
      end
    end
  end

  describe "body encoding & context translation" do
    test "encode_body without tools" do
      model = ReqLLM.Model.from!("mistral:mistral-large-latest")
      context = context_fixture()

      # Create a mock request with the expected structure
      mock_request = %Req.Request{
        options: [
          context: context,
          model: model.model,
          temperature: 0.7,
          max_tokens: 100
        ],
        private: %{}
      }

      # Encode the body using Mistral provider defaults
      encoded_request = Mistral.encode_body(mock_request)

      assert %Req.Request{} = encoded_request
      assert is_binary(encoded_request.body)

      # Verify JSON structure
      body = Jason.decode!(encoded_request.body)
      assert body["model"] == model.model
      assert body["temperature"] == 0.7
      assert body["max_tokens"] == 100
      assert is_list(body["messages"])
    end

    test "encode_body with tools" do
      model = ReqLLM.Model.from!("mistral:mistral-large-latest")
      context = context_fixture()

      tool = ReqLLM.tool(
        name: "get_weather",
        description: "Get weather info",
        parameter_schema: [location: [type: :string, required: true]]
      )

      mock_request = %Req.Request{
        options: [
          context: context,
          model: model.model,
          temperature: 0.7,
          max_tokens: 100,
          tools: [tool]
        ],
        private: %{}
      }

      encoded_request = Mistral.encode_body(mock_request)
      body = Jason.decode!(encoded_request.body)

      assert is_list(body["tools"])
      assert length(body["tools"]) == 1
      assert hd(body["tools"])["function"]["name"] == "get_weather"
    end
  end

  describe "response decoding" do
    test "decode_response handles standard chat response" do
      mock_response_body = %{
        "id" => "chatcmpl-test",
        "object" => "chat.completion",
        "model" => "mistral-large-latest",
        "choices" => [
          %{
            "index" => 0,
            "message" => %{
              "role" => "assistant",
              "content" => "Hello! How can I help you today?"
            },
            "finish_reason" => "stop"
          }
        ],
        "usage" => %{
          "prompt_tokens" => 10,
          "completion_tokens" => 9,
          "total_tokens" => 19
        }
      }

      mock_response = %Req.Response{
        status: 200,
        body: Jason.encode!(mock_response_body),
        headers: [{"content-type", "application/json"}]
      }

      {:ok, decoded_response} = Mistral.decode_response(mock_response)

      assert %ReqLLM.Response{} = decoded_response
      assert decoded_response.message.content == "Hello! How can I help you today?"
      assert decoded_response.usage.input_tokens == 10
      assert decoded_response.usage.output_tokens == 9
    end

    test "decode_response handles error response" do
      error_body = %{
        "error" => %{
          "type" => "invalid_request_error",
          "message" => "Invalid API key"
        }
      }

      mock_response = %Req.Response{
        status: 401,
        body: Jason.encode!(error_body),
        headers: [{"content-type", "application/json"}]
      }

      {:error, error} = Mistral.decode_response(mock_response)
      assert %ReqLLM.Error.HTTP{} = error
      assert error.status == 401
    end
  end
end