defmodule ReqLLM.Coverage.Mistral.ObjectGenerationTest do
  @moduledoc """
  Mistral object generation capability tests using fixtures.

  Tests structured output generation with Mistral models.
  """

  use ReqLLM.ProviderTest.ObjectGeneration,
    provider: :mistral,
    model: "mistral:mistral-large-latest"
end