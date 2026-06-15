include(joinpath(@__DIR__, "LLMUtils.jl"))

# ---------------------------------------------------------------------------
# API credentials.
#
# Secrets are read from environment variables so that no keys are committed to
# the repository. Set the ones you need before running, e.g. (bash):
#   export OPENROUTER_API_KEY="sk-or-v1-..."
#   export AZURE_OPENAI_KEY="..."
#   export AZURE_OPENAI_ENDPOINT="my-azure-resource"
#
# For the experiments in this paper only OPENROUTER_API_KEY is required
# (Gemini 3.1 Pro is accessed via OpenRouter). The Azure/OpenAI and other
# providers below are kept for completeness but are not used by the paper.
# ---------------------------------------------------------------------------

# ---- Azure OpenAI ---------------------------------------------------------
key = get(ENV, "AZURE_OPENAI_KEY", "")
version_new = "2025-01-01-preview"
version_new2 = "2024-12-01-preview"
version = "2024-10-01-preview"
version_old = "2024-05-01-preview"
endpoint = get(ENV, "AZURE_OPENAI_ENDPOINT", "")
base_url = "https://$(endpoint).openai.azure.com/openai/deployments/"

gpt4o = LLMAccessAzureOpenAI(key, "gpt-4o", version, endpoint)

gpt4_1 = LLMAccessAzureOpenAI(key, "gpt-4.1", version_new2, endpoint)
gpt4_1_mini = LLMAccessAzureOpenAI(key, "gpt-4.1-mini", version_new2, endpoint)
gpt4_1_nano = LLMAccessAzureOpenAI(key, "gpt-4.1-nano", version_new2, endpoint)

gpt4_1_batch = LLMAccessAzureOpenAI(key, "gpt-4.1", version_new2, endpoint, "gpt-4.1-batch")

gpt5 = LLMAccessAzureOpenAI(key, "gpt-5", version_new2, endpoint)
gpt5_mini = LLMAccessAzureOpenAI(key, "gpt-5-mini", version_new2, endpoint)
gpt5_nano = LLMAccessAzureOpenAI(key, "gpt-5-nano", version_new2, endpoint)

gpt5_batch = LLMAccessAzureOpenAI(key, "gpt-5", version_new2, endpoint, "gpt-5-batch")

# ---- OpenRouter (used for the Gemini 3.1 Pro runs in this paper) -----------
key_openrouter = get(ENV, "OPENROUTER_API_KEY", "")

gemini_2_5_flash_lite = LLMAccessOpenRouter(key_openrouter, "google/gemini-2.5-flash-lite", ["google-vertex"])
gemini_2_5_flash = LLMAccessOpenRouter(key_openrouter, "google/gemini-2.5-flash", ["google-vertex/global"])
gemini_2_5 = LLMAccessOpenRouter(key_openrouter, "google/gemini-2.5-pro", ["google-vertex/global"])
gemini_3_1_pro = LLMAccessOpenRouter(key_openrouter, "google/gemini-3.1-pro-preview", ["google-vertex/global"])
gemini_3_flash = LLMAccessOpenRouter(key_openrouter, "google/gemini-3-flash-preview", ["google-vertex/global"])
gemini_3_1_flash_lite = LLMAccessOpenRouter(key_openrouter, "google/gemini-3.1-flash-lite-preview", ["google-vertex/global"])
gemini_3_5_flash = LLMAccessOpenRouter(key_openrouter, "google/gemini-3.5-flash", ["google-vertex/global"])

gemini_3_1_pro_flex = LLMAccessOpenRouter(key_openrouter, "google/gemini-3.1-pro-preview", ["google-vertex/global"], "flex")
gemini_3_flash_flex = LLMAccessOpenRouter(key_openrouter, "google/gemini-3-flash-preview", ["google-vertex/global"], "flex")
gemini_3_1_flash_lite_flex = LLMAccessOpenRouter(key_openrouter, "google/gemini-3.1-flash-lite-preview", ["google-vertex/global"], "flex")
gemini_3_5_flash_flex = LLMAccessOpenRouter(key_openrouter, "google/gemini-3.5-flash", ["google-vertex/global"], "flex")

# NOTE: OpenRouter has no `google/gemini-3.1-flash-preview` (only the lite variant at 3.1).
# Use `gemini_3_flash` (slug above) as the non-lite Flash from the 3.x family.

claude_opus_4_5 = LLMAccessOpenRouter(key_openrouter, "anthropic/claude-opus-4.5", ["amazon-bedrock"])
claude_sonnet_4_5 = LLMAccessOpenRouter(key_openrouter, "anthropic/claude-sonnet-4.5", ["google-vertex"])
claude_haiku_4_5 = LLMAccessOpenRouter(key_openrouter, "anthropic/claude-haiku-4.5", ["amazon-bedrock"])

claude_opus_4_5_flex = LLMAccessOpenRouter(key_openrouter, "anthropic/claude-opus-4.5", ["amazon-bedrock"], "flex")
claude_sonnet_4_5_flex = LLMAccessOpenRouter(key_openrouter, "anthropic/claude-sonnet-4.5", ["google-vertex"], "flex")
claude_haiku_4_5_flex = LLMAccessOpenRouter(key_openrouter, "anthropic/claude-haiku-4.5", ["amazon-bedrock"], "flex")

key_openrouter_copilot = get(ENV, "OPENROUTER_API_KEY_COPILOT", "")
