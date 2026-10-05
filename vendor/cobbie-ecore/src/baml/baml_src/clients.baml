// Learn more about clients at https://docs.boundaryml.com/docs/snippets/clients/overview

// Z.AI GLM 4.7 (with Coding Plan API key)
client<llm> GLM_4_7{
  provider openai
  options {
    base_url "https://api.z.ai/api/coding/paas/v4"
    model "glm-4.7"
    api_key env.Z_AI_API_KEY
    temperature 0.1
    max_tokens 8192
    http {
      connect_timeout_ms 30000
      request_timeout_ms 180000
    }
  }
}

// Z.AI GLM 4.7 Flash (lightweight variant, free)
client<llm> GLM_4_7_Flash {
  provider openai
  retry_policy Exponential
  options {
    base_url "https://api.z.ai/api/coding/paas/v4"
    model "glm-4.7-flash"
    api_key env.Z_AI_API_KEY
    temperature 0.1
    max_tokens 8192
    http {
      connect_timeout_ms 10000
      request_timeout_ms 60000
    }
  }
}

// Z.AI GLM 4.5 (with Coding Plan API key)
client<llm> GLM_4_5_air{
  provider openai
  retry_policy Exponential
  options {
    base_url "https://api.z.ai/api/coding/paas/v4"
    model "glm-4.5-air"
    api_key env.Z_AI_API_KEY
    temperature 0.1
    max_tokens 8192
    http {
      connect_timeout_ms 10000
      request_timeout_ms 60000
    }
  }
}

client<llm> Devstral {
  provider "openai-generic"
  retry_policy Constant
  options {
    base_url "http://localhost:11434/v1"
    model "devstral-small-2:latest"
    http {
      connect_timeout_ms 10000
      request_timeout_ms 300000
    }
  }
}

client<llm> Gemini_2_5_Flash_Lite {
  provider google-ai
  retry_policy Exponential
  options {
    model "gemini-2.5-flash-lite"
    api_key env.GEMINI_API_KEY
    generationConfig {
      temperature 0.1
    }
    http {
      connect_timeout_ms 10000
      request_timeout_ms 60000
    }
  }
}

// OpenAI GPT-4.1
client<llm> OpenAI_GPT_4_1 {
  provider openai
  retry_policy Exponential
  options {
    model "gpt-4.1"
    api_key env.OPENAI_API_KEY
    temperature 0.1
    max_tokens 8192
    http {
      connect_timeout_ms 30000
      request_timeout_ms 180000
    }
  }
}

// Frozen GPT-4.1 snapshot used by the current ToG planner and closure runtime.
// It remains distinct from the moving GPT-4.1 alias for source lineage.
client<llm> OpenAI_GPT_4_1_2025_04_14 {
  provider openai
  retry_policy Exponential
  options {
    model "gpt-4.1-2025-04-14"
    api_key env.OPENAI_API_KEY
    temperature 0
    max_tokens 8192
    http {
      connect_timeout_ms 30000
      request_timeout_ms 180000
    }
  }
}

// Frozen GPT-4.1 client for the retrieval-closure causal experiment.  The
// application intentionally does not impose an output-token ceiling; provider
// context limits still apply.
client<llm> OpenAI_GPT_4_1_2025_04_14_NoCap {
  provider openai
  retry_policy Exponential
  options {
    model "gpt-4.1-2025-04-14"
    api_key env.OPENAI_API_KEY
    temperature 0
    http {
      connect_timeout_ms 30000
      request_timeout_ms 180000
    }
  }
}

// OpenAI GPT-5.5 system reasoning client.
// Responses API parameters intentionally omit temperature and legacy max_tokens.
client<llm> OpenAI_GPT_5_5_SYSTEM {
  provider openai-responses
  retry_policy Exponential
  options {
    model "gpt-5.5-2026-04-23"
    api_key env.OPENAI_API_KEY
    reasoning {
      effort "medium"
    }
    max_output_tokens 8192
    http {
      connect_timeout_ms 30000
      request_timeout_ms 180000
    }
  }
}

// OpenAI GPT-5.5 bounded COBBIE judge client.
// A smaller output ceiling keeps five-criterion rejudging predictable.
client<llm> OpenAI_GPT_5_5_JUDGE {
  provider openai-responses
  retry_policy Exponential
  options {
    model "gpt-5.5-2026-04-23"
    api_key env.OPENAI_API_KEY
    reasoning {
      effort "medium"
    }
    max_output_tokens 2048
    http {
      connect_timeout_ms 30000
      request_timeout_ms 180000
    }
  }
}

// Gemini 3 Pro Preview - for second LLM judge evaluation
client<llm> Gemini_3_Pro {
  provider google-ai
  retry_policy Exponential
  options {
    model "gemini-3-pro-preview"
    api_key env.GEMINI_API_KEY
    http {
      connect_timeout_ms 10000
      request_timeout_ms 60000
    }
  }
}

// MiniMax M2.7 (OpenAI-compatible API)
client<llm> MiniMax_M2_7 {
  provider openai
  options {
    base_url "https://api.minimax.io/v1"
    model "MiniMax-M2.7"
    api_key env.MINIMAX_API_KEY
    temperature 0.1
    http {
      connect_timeout_ms 30000
      request_timeout_ms 300000
    }
  }
}

// Anthropic Claude Sonnet 4.6
client<llm> Claude_Sonnet_4_6 {
  provider anthropic
  retry_policy Exponential
  options {
    model "claude-sonnet-4-6"
    api_key env.ANTHROPIC_API_KEY
    max_tokens 8192
    temperature 0.1
    http {
      connect_timeout_ms 30000
      request_timeout_ms 180000
    }
  }
}

// https://docs.boundaryml.com/docs/snippets/clients/retry
retry_policy Constant {
  max_retries 0
}

retry_policy Exponential {
  max_retries 0
}
