defmodule ReqLLM.Coverage.Mistral.CoreTest do
  @moduledoc """
  Core Mistral API feature coverage tests using simple fixtures.

  Run with LIVE=true to test against live API and record fixtures.
  Otherwise uses fixtures for fast, reliable testing.
  """

  use ReqLLM.ProviderTest.Core,
    provider: :mistral,
    model: "mistral:mistral-large-latest"

  # Mistral-specific tests
  test "reasoning model generation" do
    {:ok, response} =
      ReqLLM.generate_text(
        "mistral:magistral-medium-latest",
        "What is 2+2? Think step by step.",
        temperature: 0.0,
        max_tokens: 50,
        fixture: "reasoning_generation"
      )

    assert %ReqLLM.Response{} = response
    text = ReqLLM.Response.text(response)
    assert is_binary(text)
    assert String.length(text) > 0
    assert response.id != nil
    assert text =~ "4"
  end

  test "code generation with Codestral" do
    {:ok, response} =
      ReqLLM.generate_text(
        "mistral:codestral-latest",
        "Write a Python function to add two numbers",
        temperature: 0.1,
        max_tokens: 100,
        fixture: "code_generation"
      )

    assert %ReqLLM.Response{} = response
    text = ReqLLM.Response.text(response)
    assert is_binary(text)
    assert String.length(text) > 0
    assert response.id != nil
    assert text =~ "def"
  end

  test "small model efficiency" do
    {:ok, response} =
      ReqLLM.generate_text(
        "mistral:ministral-3b-latest",
        "Hello world!",
        temperature: 0.0,
        max_tokens: 10,
        fixture: "small_model"
      )

    assert %ReqLLM.Response{} = response
    text = ReqLLM.Response.text(response)
    assert is_binary(text)
    assert String.length(text) > 0
    assert response.id != nil
  end
end