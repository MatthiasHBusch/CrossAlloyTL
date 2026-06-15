#=
Mock-run sanity check for the molecule-disjoint filter.

We reuse the upstream `create_experiment_states` directly (no API calls)
and verify, for each of the 5 transfer settings:

 - close_unfilt / far_unfilt: at least one test-fold SMILES MUST appear
   inside the constructed training-data prompt (= positive control:
   leakage is present).
 - close_filt / far_filt: ZERO test-fold SMILES may appear inside the
   training-data prompt (= negative control: filter works).
 - exact: training-data prompt is empty (no transfer pool).

Exits with code 1 if any check fails.
=#

include(joinpath(@__DIR__, "julia_lib", "LLMs.jl"))
include(joinpath(@__DIR__, "julia_lib", "Corrosion_Prompts_efficient.jl"))

using DataFrames, CSV, Random

const TARGET_ALLOYS = ["AZ31", "AZ91", "WE43"]
const SETTINGS = ["exact", "close_unfilt", "close_filt",
                  "far_unfilt", "far_filt"]
const K_FOLD = 5
const SEED = 42

const SCRIPT_DIR = @__DIR__
const DATA_FILE = joinpath(SCRIPT_DIR, "..", "data", "ExCorrDatasetClean.csv")

# --- Helpers (mirroring run_gemini.jl) ------------------------------------
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
get_close_pool(d) = filter(r -> r.BaseMaterial == "Mg" && !(r.Alloy in TARGET_ALLOYS), d)
get_far_pool(d) = filter(r -> r.BaseMaterial != "Mg", d)

function setting_inputs(setting::String, close_pool::DataFrame, far_pool::DataFrame)
    if setting == "exact";        return (DataFrame(), DataFrame(), false)
    elseif setting == "close_unfilt"; return (close_pool, DataFrame(), false)
    elseif setting == "close_filt";   return (close_pool, DataFrame(), true)
    elseif setting == "far_unfilt";   return (close_pool, far_pool,   false)
    elseif setting == "far_filt";     return (close_pool, far_pool,   true)
    end
end

# A dummy LLMChat (not called, just needed to construct states)
chat = LLMChat(gemini_3_1_pro)
chat.reasoning_effort = "minimal"
chat.max_tokens = 100
const DUMMY_LLM = LLMChat[chat]

# --- Per-setting filter check ---------------------------------------------
println("Loading data: $DATA_FILE")
full_data = DataFrame(CSV.File(DATA_FILE))
close_pool = get_close_pool(full_data)
far_pool = get_far_pool(full_data)
println("Close pool: $(nrow(close_pool))  Far pool: $(nrow(far_pool))")

global n_failures = 0

for alloy in TARGET_ALLOYS
    println("\n=== ALLOY: $alloy ===")
    alloy_data = prepare_alloy_data(full_data, alloy)
    rng = MersenneTwister(SEED)
    shuffled_idx = shuffle(rng, 1:nrow(alloy_data))
    shuffled = alloy_data[shuffled_idx, :]

    # Per-fold test SMILES sets (matches what create_experiment_states uses
    # internally because we pass the same SEED).
    fold_size = div(nrow(shuffled), K_FOLD)
    test_smiles_per_fold = [Set{String}(shuffled[((j-1)*fold_size+1):(j*fold_size),
                                                 :isomeric_SMILES])
                            for j in 1:K_FOLD]

    for setting in SETTINGS
        close_df, far_df, do_filter = setting_inputs(setting, close_pool, far_pool)

        states = create_experiment_states(
            alloy_data, vcat(close_df, far_df), DUMMY_LLM,
            ["names_only"], ["with_preanalysis"],
            K_FOLD, [60], nrow(close_df), nrow(far_df), 1, SEED;
            filter_test_molecules=do_filter)

        # Build a "training prompt" string from each state's conversation.
        # We look at the extended-training-data message (cache-tagged) and
        # check whether ANY of THIS fold's OTHER test SMILES (i.e. excluding
        # the one being predicted in this state) appear as a labelled
        # training example.
        # A SMILES that only appears in the question prompt for the current
        # molecule does NOT count as leakage.
        leaks_per_fold = zeros(Int, K_FOLD)
        for state in states
            test_smi = state.data["smiles_string"]
            fold_id = 0
            for (j, s) in enumerate(test_smiles_per_fold)
                if test_smi in s
                    fold_id = j
                    break
                end
            end
            fold_id == 0 && continue
            # Find the extended-training-data message
            for turn in state.llmchat.conversation
                if length(turn) >= 2 && occursin("Here is additional data",
                                                  String(turn[2]))
                    train_text = String(turn[2])
                    # Count test SMILES (other than the current one) that
                    # leak into the training section.
                    # Use precise matching: SMILES appears as a complete
                    # value on its own line ("<label>: <SMILES>\n"), not a
                    # substring of a longer SMILES.
                    train_lines = split(train_text, "\n")
                    train_smi_set = Set{String}()
                    for ln in train_lines
                        # Lines look like "Smiles: <SMI>" or "Structure: <SMI>"
                        m = match(r"^[A-Za-z ]+:\s*(\S.*\S|\S)\s*$", ln)
                        m === nothing && continue
                        push!(train_smi_set, String(m.captures[1]))
                    end
                    for ts in test_smiles_per_fold[fold_id]
                        ts == test_smi && continue
                        if ts in train_smi_set
                            leaks_per_fold[fold_id] += 1
                        end
                    end
                    break
                end
            end
        end

        total_leaks = sum(leaks_per_fold)

        # Decide pass/fail
        if setting == "exact"
            ok = total_leaks == 0
            status = ok ? "OK (no transfer pool)" : "FAIL — exact should have no train data"
        elseif endswith(setting, "_unfilt")
            ok = total_leaks > 0  # leakage present = expected
            status = ok ? "OK — leakage detected as expected ($total_leaks SMILES occurrences)" :
                          "FAIL — expected leakage but found NONE"
        else  # _filt
            ok = total_leaks == 0  # no leakage = expected
            status = ok ? "OK — no leakage as expected" :
                          "FAIL — found $total_leaks leaked SMILES occurrences"
        end

        println("  $setting: leaks/fold = $leaks_per_fold  → $status")
        if !ok
            global n_failures += 1
        end
    end
end

println("\n" * "="^60)
if n_failures == 0
    println("✓ ALL CHECKS PASSED")
    exit(0)
else
    println("✗ $n_failures CHECK(S) FAILED")
    exit(1)
end
