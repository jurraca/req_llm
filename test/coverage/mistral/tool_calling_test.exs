defmodule ReqLLM.Coverage.Mistral.ToolCallingTest do
  @moduledoc """
  Mistral tool calling capability tests using fixtures.

  Tests function calling with various Mistral models.
  """

  use ReqLLM.ProviderTest.ToolCalling,
    provider: :mistral,
    model: "mistral:mistral-large-latest"
end