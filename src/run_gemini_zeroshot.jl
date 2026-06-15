#=
Zero-shot Gemini 3.1 Pro contamination check.

For each of the 3 target alloys (AZ31, AZ91, WE43), ask Gemini 3.1 Pro to
predict the inhibition efficiency for each of the 75 prepared molecules
WITHOUT giving it ANY in-context examples (no training set, no transfer
pool). Each molecule is asked 3 times to mirror the standard N_ITER.

Goal: if zero-shot performance is meaningfully above chance for any alloy,
that is strong evidence Gemini 3.1 Pro has already seen the molecule/IE
pairs (or the underlying ExCorr dataset) during pretraining.

The system prompt is the same as for the standard runs (so the framing /
units are identical), but the user message contains only the question for
the single test molecule plus its experimental conditions — no training
examples whatsoever.

Usage (from tl_crossalloy root):
    julia src/run_gemini_zeroshot.jl
=#

include(joinpath(@__DIR__, "julia_lib", "LLMs.jl"))
include(joinpath(@__DIR__, "julia_lib", "Corrosion_Prompts_efficient.jl"))

using DataFrames, CSV, Random, JSON3

const TARGET_ALLOYS = ["AZ31", "AZ91", "WE43"]
const N_ITER = 3
const SEED = 42
const NUM_THREADS = 50

const SCRIPT_DIR = @__DIR__
const DATA_FILE = joinpath(SCRIPT_DIR, "..", "data", "ExCorrDatasetClean.csv")
const RESULTS_DIR = joinpath(SCRIPT_DIR, "..", "results", "gemini_zeroshot")
mkpath(RESULTS_DIR)

function prepare_alloy_data(full_data::DataFrame, alloy::String)
    alloy_data = filter(row -> row.BaseMaterial == "Mg" && row.Alloy == alloy, full_data)
    grouped = groupby(alloy_data,
        [:BaseMaterial, :Alloy, :Method, :AggressiveComponent,
         :Operating_Concentration_mM])
    sizes = [nrow(g) for g in grouped]
    largest = grouped[findfirst(==(maximum(sizes)), sizes)]
    counts = Dict{Any,Int}()
    for idx in largest.index
        counts[idx] = get(counts, idx, 0) + 1
    end
    dups = [k for (k, v) in counts if v > 1]
    out = filter(row -> !(row.index in dups), largest)
    return out[1:75, :]
end

# Build a sample_data dict for ONE row, suitable for fill_prompt.
function sample_dict(row)
    bms2bm = Dict("Al" => "Aluminum", "Fe" => "Iron", "Mg" => "Magnesium",
                  "Zn" => "Zinc", "Cu" => "Copper", "Ni" => "Nickel",
                  "Ti" => "Titanium")
    Dict(
        "molecule_name" => string(row.IUPAC),
        "smiles_string" => string(row.isomeric_SMILES),
        "base_material_symbol" => string(row.BaseMaterial),
        "base_material" => bms2bm[string(row.BaseMaterial)],
        "alloy_symbol" => string(row.Alloy),
        "method" => string(row.Method),
        "aggressive_component" => string(row.AggressiveComponent),
        "operating_concentration_mM" => string(row.Operating_Concentration_mM),
    )
end

# Zero-shot user prompt — same fields the standard "prediction" prompt uses,
# but WITHOUT any training-set lines.
const ZERO_SHOT_USER_TEMPLATE = """
Based on your knowledge, predict the corrosion inhibition efficiency for the following molecule:
- Corrosion modulator molecule IUPAC name: <molecule_name>
- Corrosion modulator molecule SMILES: <smiles_string>
The experimental conditions of the sample are:
- Operating concentration of inhibitor in mM: <operating_concentration_mM>
- Base material of corrosion sample: <base_material>
- Alloy of corrosion sample: <alloy_symbol>
- Method used for measuring the corrosion: <method>
- Aggressive Component: <aggressive_component>
The last sentence of your response must contain your prediction.
"""

const SYSTEM_PROMPT_TEMPLATE = prompts["names_only"]["with_preanalysis"]["system"]

function build_chat(row)
    sd = sample_dict(row)
    sys_msg = fill_prompt(SYSTEM_PROMPT_TEMPLATE, sd)
    user_msg = fill_prompt(ZERO_SHOT_USER_TEMPLATE, sd)

    c = LLMChat(gemini_3_1_pro)
    c.reasoning_effort = "minimal"
    c.max_tokens = 4000
    c.retries = 5
    push!(c.conversation, ("system", sys_msg))
    push!(c.conversation, ("user", user_msg))
    return c
end

println("Loading data: $DATA_FILE")
full_data = DataFrame(CSV.File(DATA_FILE))

for alloy in TARGET_ALLOYS
    out_file = joinpath(RESULTS_DIR, "Gemini3_1Pro_$(alloy)_zeroshot.json")
    if isfile(out_file)
        println("[skip] $alloy (already done)")
        continue
    end

    println("\n=== ALLOY: $alloy ===")
    alloy_data = prepare_alloy_data(full_data, alloy)
    n = nrow(alloy_data)
    println("  $n molecules × $N_ITER iterations = $(n*N_ITER) calls")

    # Build (mol_index, iter) pairs and a flat chat list
    chats = LLMChat[]
    mol_iter = Tuple{Int,Int}[]
    for it in 1:N_ITER
        for i in 1:n
            push!(chats, build_chat(alloy_data[i, :]))
            push!(mol_iter, (i, it))
        end
    end

    println("  Submitting $(length(chats)) parallel calls (threads=$NUM_THREADS) ...")
    answers = ask_gpt_threaded(chats; num_threads=NUM_THREADS)

    # Aggregate per molecule
    preds = Dict{String,Vector{Union{Float64,Nothing}}}()
    truth = Dict{String,Float64}()
    for i in 1:n
        name = string(alloy_data[i, :IUPAC])
        preds[name] = Vector{Union{Float64,Nothing}}(undef, N_ITER)
        truth[name] = float(alloy_data[i, :IE])
    end

    for (k, ans) in enumerate(answers)
        i, it = mol_iter[k]
        name = string(alloy_data[i, :IUPAC])
        if ans == ""
            preds[name][it] = nothing
        else
            v = search_for_last_number_in_string(String(ans))
            preds[name][it] = isnan(v) ? nothing : v
        end
    end

    # Save in a flat-ish JSON keyed like the other Gemini files but
    # without the (input_type/approach/model/ne/nc/nf) nesting since
    # those are degenerate here.
    open(out_file, "w") do io
        JSON3.pretty(io, Dict(
            "alloy" => alloy,
            "model" => "google/gemini-3.1-pro-preview",
            "n_iter" => N_ITER,
            "predictions" => preds,
            "ground_truth" => truth,
        ))
    end
    println("  Saved -> $out_file")
end

println("\nDone. Outputs in $RESULTS_DIR")
