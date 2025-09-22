defmodule ReqLLM.Coverage.Mistral.StreamingTest do
  @moduledoc """
  Mistral streaming capability tests using fixtures.

  Tests streaming text generation with various Mistral models.
  """

  use ReqLLM.ProviderTest.Streaming,
    provider: :mistral,
    model: "mistral:mistral-large-latest"
end