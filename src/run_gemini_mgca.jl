#=
Cross-alloy TL paper: Gemini 3.1 Pro (flex) runner --- MgCa CONTRAST.

Complementary to `run_gemini.jl` (the main 3-alloy in-context experiment),
this script adds Gemini 3.1 Pro as a 6th model to the MgCa contrast
experiment, so that the MgCa panel of the contrast figure can show the
in-context LLM alongside the five fitted base methods (RF, GBR, MLP,
kNN-Tanimoto, ChemProp) produced by `mgca_contrast.py`.

It mirrors the Python base-method protocol in `mgca_contrast.py`:
  * target alloy  : MgCa
  * records       : ALL MgCa records (~56, ~49 unique molecules);
                    NO largest-group selection and NO 75-record cap
                    (unlike the main 3 alloys) --- exactly as the base
                    methods do for MgCa.
  * folds         : 4-fold CV (56/4 = 14 test per fold)
  * target train  : the full complement of each fold (~42 records)
  * close pool    : all OTHER Mg alloys, INCLUDING AZ31/AZ91/WE43
                    (MgCa itself is excluded). This matches
                    `get_mgca_pools` in mgca_contrast.py and differs from
                    the main runner, where AZ31/AZ91/WE43 are the targets
                    and hence excluded from the close pool.
  * far pool      : all non-Mg base materials (Al, Fe, Cu, Zn)
  * settings      : exact, close_unfilt, close_filt, far_unfilt, far_filt
  * filtering     : per-fold molecule-disjoint filtering for the *_filt
                    settings (handled inside `create_experiment_states`
                    via filter_test_molecules=true).

Split convention: as in the main Gemini runner, `create_experiment_states`
applies a single deterministic seed-42 shuffle before forming the
contiguous k-folds. This matches the base-method protocol at the level
the main experiment used (same records, same #folds, same pools/filters);
per-fold membership is not byte-identical to the unshuffled Python folds,
exactly as in the main experiment.

To keep API costs bounded, we run only N_ITER = 3 iterations, on the
flex service tier (gemini_3_1_pro_flex).

Output JSON: results/gemini_mgca/Gemini3_1Pro_MgCa_<setting>.json
in the same nested ChemProp-style structure used elsewhere.

Usage (from tl_crossalloy root):
    julia src/run_gemini_mgca.jl
=#

include(joinpath(@__DIR__, "julia_lib", "LLMs.jl"))
include(joinpath(@__DIR__, "julia_lib", "Corrosion_Prompts_efficient.jl"))

using DataFrames, CSV, Random, JSON3
using Statistics

# ---------- Configuration -------------------------------------------------
const TARGET_ALLOY = "MgCa"
const SETTINGS = ["exact", "close_unfilt", "close_filt", "far_unfilt", "far_filt"]
const K_FOLD = 4          # 56 / 4 = 14 test per fold (matches mgca_contrast.py)
const N_ITER = 3
const SEED = 42

const SCRIPT_DIR = @__DIR__
const DATA_FILE = joinpath(SCRIPT_DIR, "..", "data", "ExCorrDatasetClean.csv")
const RESULTS_DIR = joinpath(SCRIPT_DIR, "..", "results", "gemini_mgca")
mkpath(RESULTS_DIR)

# ---------- LLM ------------------------------------------------------------
chat = LLMChat(gemini_3_1_pro_flex)
chat.reasoning_effort = "minimal"  # gemini-3.1-pro requires reasoning
chat.max_tokens = 4000
chat.retries = 5
const LLM_LIST = LLMChat[chat]

const APPROACH = "with_preanalysis"
const INPUT_TYPE = "names_only"

# ---------- Data helpers --------------------------------------------------

# All MgCa records, in dataset (CSV) order. Mirrors `prepare_mgca` in
# mgca_contrast.py: no largest-group selection, no dedup, no 75-cap.
function prepare_mgca_data(full_data::DataFrame)
    filter(row -> row.BaseMaterial == "Mg" && row.Alloy == TARGET_ALLOY, full_data)
end

# Close pool for the MgCa contrast: every OTHER Mg alloy (AZ31/AZ91/WE43
# included), MgCa excluded. Matches get_mgca_pools in mgca_contrast.py.
function get_close_pool(full_data::DataFrame)
    filter(row -> row.BaseMaterial == "Mg" &&
           row.Alloy != TARGET_ALLOY, full_data)
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

mgca_data = prepare_mgca_data(full_data)
n_mgca = nrow(mgca_data)
# Full per-fold complement: for a contiguous k-fold over n records the
# training set is n - div(n, k) records. Use that as the target-train size
# so the whole complement is kept (matches the base-method runner).
const N_TRAIN = n_mgca - div(n_mgca, K_FOLD)

close_pool = get_close_pool(full_data)
far_pool = get_far_pool(full_data)

println("\nGemini 3.1 Pro (flex) MgCa-contrast runner")
println("Target:   $TARGET_ALLOY  ($n_mgca records, $(length(unique(mgca_data.isomeric_SMILES))) unique SMILES)")
println("Settings: $SETTINGS")
println("Folds:    $K_FOLD,  target-train per fold: $N_TRAIN,  iterations per fold: $N_ITER")
println("Close pool: $(nrow(close_pool)) records   Far pool: $(nrow(far_pool)) records")
println("Per-fold molecule-disjoint filtering: enabled for *_filt settings")
println()

for setting in SETTINGS
    out_file = joinpath(RESULTS_DIR,
        "Gemini3_1Pro_$(TARGET_ALLOY)_$(setting).json")
    chat_file = joinpath(RESULTS_DIR,
        "Gemini3_1Pro_$(TARGET_ALLOY)_$(setting)_chat.json")
    if isfile(out_file)
        println("  [skip] $TARGET_ALLOY / $setting (already done)")
        continue
    end
    close_df, far_df, do_filter = setting_inputs(setting, close_pool, far_pool)
    println("  Running $TARGET_ALLOY / $setting " *
            "(close=$(nrow(close_df)), far=$(nrow(far_df)), filter=$do_filter)")

    full_run_extended(out_file, mgca_data, LLM_LIST, [INPUT_TYPE],
                      [APPROACH], K_FOLD, [N_TRAIN],
                      close_df, far_df, N_ITER;
                      save_chat_to_file_name=chat_file,
                      random_split_seed=SEED,
                      mock=false,
                      filter_test_molecules=do_filter)
end

println("\nDone. Raw outputs in $RESULTS_DIR")
