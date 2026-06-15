#=
Cross-alloy TL paper: Gemini 3.1 Pro runner.

Adds Gemini 3.1 Pro as a 6th model, evaluated on the same 5 transfer
settings (exact, close_unfilt, close_filt, far_unfilt, far_filt) for
the 3 target alloys (AZ31, AZ91, WE43) with 5-fold CV.

Per-fold molecule-disjoint filtering is handled inside the upstream
`create_experiment_states` via the `filter_test_molecules=true` kwarg.
This means a SINGLE call per (alloy, setting) covers all 5 folds.

Output JSON: results/gemini/Gemini3_1Pro_<alloy>_<setting>.json
in the same nested ChemProp-style structure used elsewhere.

To keep API costs bounded, we run only N_ITER = 3 iterations.

Usage (from tl_crossalloy root):
    julia src/run_gemini.jl
=#

include(joinpath(@__DIR__, "julia_lib", "LLMs.jl"))
include(joinpath(@__DIR__, "julia_lib", "Corrosion_Prompts_efficient.jl"))

using DataFrames, CSV, Random, JSON3
using Statistics

# ---------- Configuration -------------------------------------------------
const TARGET_ALLOYS = ["AZ31", "AZ91", "WE43"]
const SETTINGS = ["exact", "close_unfilt", "close_filt", "far_unfilt", "far_filt"]
const K_FOLD = 5
const N_ITER = 3
const SEED = 42

const SCRIPT_DIR = @__DIR__
const DATA_FILE = joinpath(SCRIPT_DIR, "..", "data", "ExCorrDatasetClean.csv")
const RESULTS_DIR = joinpath(SCRIPT_DIR, "..", "results", "gemini")
mkpath(RESULTS_DIR)

# ---------- LLM ------------------------------------------------------------
chat = LLMChat(gemini_3_1_pro)
chat.reasoning_effort = "minimal"  # gemini-3.1-pro requires reasoning
chat.max_tokens = 4000
chat.retries = 5
const LLM_LIST = LLMChat[chat]

const APPROACH = "with_preanalysis"
const INPUT_TYPE = "names_only"

# ---------- Data helpers --------------------------------------------------

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

function get_close_pool(full_data::DataFrame)
    filter(row -> row.BaseMaterial == "Mg" &&
           !(row.Alloy in TARGET_ALLOYS), full_data)
end

function get_far_pool(full_data::DataFrame)
    filter(row -> row.BaseMaterial != "Mg", full_data)
end

# Map setting name -> (close_df, far_df, filter_flag)
function setting_inputs(setting::String, close_pool::DataFrame, far_pool::DataFrame)
    if setting == "exact"
        return (DataFrame(), DataFrame(), false)
    elseif setting == "close_unfilt"
        return (close_pool, DataFrame(), false)
    elseif setting == "close_filt"
        return (close_pool, DataFrame(), true)
    elseif setting == "far_unfilt"
        return (close_pool, far_pool, false)
    elseif setting == "far_filt"
        return (close_pool, far_pool, true)
    else
        error("Unknown setting: $setting")
    end
end

# ---------- Main loop -----------------------------------------------------

println("Loading data: $DATA_FILE")
full_data = DataFrame(CSV.File(DATA_FILE))
println("Total records: $(nrow(full_data))")

close_pool = get_close_pool(full_data)
far_pool = get_far_pool(full_data)
println("Close pool: $(nrow(close_pool)) records")
println("Far pool:   $(nrow(far_pool)) records")

println("\nGemini 3.1 Pro cross-alloy TL runner")
println("Settings: $SETTINGS")
println("Alloys:   $TARGET_ALLOYS")
println("Folds:    $K_FOLD,  iterations per fold: $N_ITER")
println("Per-fold molecule-disjoint filtering: enabled for *_filt settings")
println()

for alloy in TARGET_ALLOYS
    println("\n=== ALLOY: $alloy ===")
    alloy_data = prepare_alloy_data(full_data, alloy)
    for setting in SETTINGS
        out_file = joinpath(RESULTS_DIR,
            "Gemini3_1Pro_$(alloy)_$(setting).json")
        chat_file = joinpath(RESULTS_DIR,
            "Gemini3_1Pro_$(alloy)_$(setting)_chat.json")
        if isfile(out_file)
            println("  [skip] $alloy / $setting (already done)")
            continue
        end
        close_df, far_df, do_filter = setting_inputs(setting, close_pool, far_pool)
        println("  Running $alloy / $setting " *
                "(close=$(nrow(close_df)), far=$(nrow(far_df)), filter=$do_filter)")

        full_run_extended(out_file, alloy_data, LLM_LIST, [INPUT_TYPE],
                          [APPROACH], K_FOLD, [60],
                          close_df, far_df, N_ITER;
                          save_chat_to_file_name=chat_file,
                          random_split_seed=SEED,
                          mock=false,
                          filter_test_molecules=do_filter)
    end
end

println("\nDone. Raw outputs in $RESULTS_DIR")
