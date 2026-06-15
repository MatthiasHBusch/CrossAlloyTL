# Quick sanity check: can we call gemini_3_1_pro at all?
include(joinpath(@__DIR__, "julia_lib", "LLMs.jl"))

chat = LLMChat(gemini_3_1_pro)
chat.reasoning_effort = "minimal"
chat.max_tokens = 200
chat.retries = 1

push!(chat.conversation, ("system", "You are a chemist."))
push!(chat.conversation, ("user", "Reply with the single number 42."))

println("Calling gemini_3_1_pro ...")
try
    response = ask_gpt(chat)
    println("Response: ", response)
catch e
    println("ERROR: ", e)
end
