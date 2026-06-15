include(joinpath(@__DIR__, "LLMUtils.jl"))
include(joinpath(@__DIR__, "FileWritingHelpers.jl"))
using CSV
using DataFrames
using JSON
using Dates
using PrettyTables
using Statistics
using Printf
using Random


# --- STRUCTS & TYPES ---

mutable struct ExperimentState
    id::String
    input_data::String
    approach::String
    llmchat::LLMChat
    num_exact_samples::Int           # Exact samples (target alloy)
    num_close_samples::Int           # Close transfer samples
    num_far_samples::Int             # Far transfer samples
    data::Dict # Contains specific training/test data for this sample
    step_index::Int
    done::Bool
    prediction::Any
    history::Vector{String} # To track which prompts were used
    run_id::Int # To track which iteration this state belongs to (for batch splitting)
end

# Lock for file writing to avoid race conditions in parallel execution
const FILE_WRITE_LOCK = ReentrantLock()

# --- EXPERIMENT LOGIC ---

function get_prompt_sequence(input_data::String, approach::String)
    # Define the sequence of keys in `prompts` for each approach
    tasks = get_possible_tasks(input_data, approach)

    # Standard ordering logic
    analyses = sort(collect(filter(x -> contains(x, "analysis"), tasks)))
    consecutive = sort(collect(filter(x -> contains(x, "summary"), tasks)))
    prediction = collect(filter(x -> contains(x, "prediction"), tasks))

    return [analyses; consecutive; prediction]
end

function create_experiment_states(
    data_frame_test::DataFrame,
    data_frame_train::DataFrame,
    llmchats::Vector{LLMChat},
    input_types::Vector{String},
    approaches::Vector{String},
    k_folds::Int,
    exact_training_sizes::Vector{Int},
    num_close_samples::Int,
    num_far_samples::Int,
    num_runs::Int,
    random_split_seed::Union{Int,Missing};
    filter_test_molecules::Bool=false
)
    states = Vector{ExperimentState}()
    rng = MersenneTwister(random_split_seed)

    # Column names for corrosion data (default)
    name_column = "IUPAC"
    smiles_column = "isomeric_SMILES"
    target_column = "IE"
    descriptors_columns = ["P_VSA_MR_5", "LUMO / eV", "E1p", "CATS3D_02_AP", "Mor04m"]

    # Check if descriptors exist in the data
    has_descriptors = all(col -> col in names(data_frame_test), descriptors_columns)

    # Shuffle data once
    data_shuffled = data_frame_test[shuffle(rng, 1:size(data_frame_test, 1)), :]
    n_rows = size(data_shuffled, 1)

    # Create states
    count = 0
    for num_exact_samples in exact_training_sizes
        num_test_samples = div(n_rows, k_folds)
        test_sets = [(num_test_samples*(i-1)+1):(num_test_samples*i) for i in 1:k_folds]
        training_sets = [setdiff(1:n_rows, ts)[1:num_exact_samples] for ts in test_sets]

        for run in 1:num_runs
            for j in 1:k_folds
                training_indices = training_sets[j]
                test_indices = Vector{Int}(test_sets[j])

                # Get the data dictionary for this specific fold
                fold_data = get_data_dict(data_shuffled, training_indices, test_indices)

                # Extract test samples
                test_molecule_names = fold_data["molecule_names_test_single"]
                test_smiles = fold_data["smiles_strings_test_single"]

                # Optional: filter the transfer pool by SMILES of this fold's
                # test molecules to remove molecule-level label leakage.
                if filter_test_molecules && size(data_frame_train, 1) > 0
                    test_smiles_set = Set{String}(test_smiles)
                    data_frame_train_local = filter(
                        row -> !(row.isomeric_SMILES in test_smiles_set),
                        data_frame_train)
                else
                    data_frame_train_local = data_frame_train
                end

                for (idx, molecule_name) in enumerate(test_molecule_names)
                    # Create specific data dict for this sample
                    sample_data = copy(fold_data)
                    sample_data["molecule_name"] = molecule_name
                    sample_data["smiles_string"] = test_smiles[idx]

                    # Add descriptors if present
                    if has_descriptors && haskey(fold_data, "descriptors_test_single")
                        sample_data["descriptors"] = fold_data["descriptors_test_single"][idx]
                    end

                    for input_data in input_types
                        for approach in approaches
                            # Determine which columns to use based on approach
                            target_column_approach = target_column
                            smiles_column_approach = smiles_column
                            base_material_column = "BaseMaterial"
                            alloy_column = "Alloy"
                            # Define labels based on blinding level (defaulting to Level 1)
                            question_prefix = "Predict the inhibition efficiency of the following setting"
                            name_label = "IUPAC"
                            structure_label = "SMILES"
                            material_label_prefix = "Corrosion sample"
                            conc_label = "Concentration inhibitor mM"
                            aggressive_label = "Aggressive environmental component"
                            target_label = "inhibition efficiency"
                            show_aggressive = true

                            if approach in ["input_output_prompting", "with_preanalysis", "gpt_generated_prompts", "refined_prompting"]
                                # Level 1 Defaults apply
                            elseif approach == "wp_corrosion_blind"
                                # Level 2
                                target_column_approach = "transformed_IE"
                                target_label = "molecular property related to corrosion modulation"
                                question_prefix = "Predict the molecular property related to corrosion modulation of the following setting"
                            elseif approach == "wp_molproperty_clear"
                                # Level 3
                                question_prefix = "Predict the molecular property of the following setting"
                                name_label = "IUPAC"
                                structure_label = "SMILES"
                                material_label_prefix = "Material setting"
                                conc_label = "Operating concentration in mM"
                                target_label = "molecular property"
                                show_aggressive = false
                            elseif approach == "wp_molproperty_blind"
                                # Level 4
                                target_column_approach = "transformed_IE"
                                question_prefix = "Predict the molecular property of the following setting"
                                name_label = "IUPAC"
                                structure_label = "SMILES"
                                material_label_prefix = "Material setting"
                                conc_label = "Operating concentration in mM"
                                target_label = "molecular property"
                                show_aggressive = false
                            elseif approach == "wp_sampleproperty_clear"
                                # Level 5
                                question_prefix = "Predict the sample property of the following setting"
                                name_label = "" # Or omitted if not in prompt, but we'll keep consistent structure
                                smiles_column_approach = "transformed_smiles"
                                structure_label = "Sample structure string"
                                material_label_prefix = "Parameter"
                                base_material_column = "transformed_base_material"
                                alloy_column = "transformed_alloy"
                                conc_label = "Parameter 1"
                                target_label = "sample property"
                                show_aggressive = false
                            elseif approach == "wp_sampleproperty_blind"
                                # Level 6
                                question_prefix = "Predict the sample property of the following setting"
                                name_label = ""
                                target_column_approach = "transformed_IE" # transformed IE
                                smiles_column_approach = "transformed_smiles"
                                structure_label = "Sample structure string"
                                material_label_prefix = "Parameter"
                                base_material_column = "transformed_base_material"
                                alloy_column = "transformed_alloy"
                                conc_label = "Parameter 1"
                                target_label = "sample property"
                                show_aggressive = false
                            else
                                @warn "Unknown approach: $approach, using default columns"
                            end

                            # Generate extended training data prompt matching finetuning dataset format.
                            # Use the (optionally filtered) per-fold training set.
                            prompt_extended_training_data = ""
                            if size(data_frame_train_local, 1) > 0
                                prompt_extended_training_data = "Here is additional data from previous experiments that might help you:\n"
                                for i in axes(data_frame_train_local, 1)
                                    # Match finetuning dataset format
                                    prompt_extended_training_data *= "Question: $(question_prefix):\n"

                                    if name_label != ""
                                        prompt_extended_training_data *= "$(name_label): $(data_frame_train_local[i, name_column])\n"
                                    end
                                    prompt_extended_training_data *= "$(structure_label): $(data_frame_train_local[i, smiles_column_approach])\n"

                                    if base_material_column in names(data_frame_train_local) && alloy_column in names(data_frame_train_local)
                                        base_mat_value = data_frame_train_local[i, base_material_column]
                                        alloy_value = data_frame_train_local[i, alloy_column]

                                        if approach in ["wp_sampleproperty_clear", "wp_sampleproperty_blind"]
                                            prompt_extended_training_data *= "Parameter 2: $(base_mat_value)\n"
                                            prompt_extended_training_data *= "Parameter 3: $(alloy_value)\n"
                                        else
                                            prompt_extended_training_data *= "$(material_label_prefix): $(base_mat_value), $(alloy_value)\n"
                                        end
                                    end

                                    if "Operating_Concentration_mM" in names(data_frame_train_local)
                                        prompt_extended_training_data *= "$(conc_label): $(data_frame_train_local[i, :Operating_Concentration_mM])\n"
                                    end

                                    if show_aggressive && "AggressiveComponent" in names(data_frame_train_local)
                                        prompt_extended_training_data *= "$(aggressive_label): $(data_frame_train_local[i, :AggressiveComponent])\n"
                                    end

                                    prompt_extended_training_data *= "Answer: Predicted $(target_label): $(data_frame_train_local[i, target_column_approach])%.\n\n"
                                end
                            end
                            for llmchat in llmchats
                                count += 1
                                state = ExperimentState(
                                    string(count),
                                    input_data,
                                    approach,
                                    deepcopy(llmchat),
                                    num_exact_samples,
                                    num_close_samples,
                                    num_far_samples,
                                    sample_data,
                                    1,    # Step 1
                                    false, # Not done
                                    nothing,
                                    [],
                                    run
                                )

                                # Initialize System Prompt
                                sys_prompt = prompts[input_data][approach]["system"]
                                sys_msg = ("system", fill_prompt(sys_prompt, sample_data))

                                # Handle o1 models that don't support system prompts
                                if contains(llmchat.llmaccess.model, "o1")
                                    sys_msg = ("user", sys_msg[2])
                                end

                                push!(state.llmchat.conversation, sys_msg)

                                # Add extended training data if provided (approach-specific)
                                if prompt_extended_training_data != ""
                                    # Check if llmchat supports caching (Gemini models)
                                    push!(state.llmchat.conversation, ("user", prompt_extended_training_data, "ephemeral"))
                                end

                                push!(states, state)
                            end
                        end
                    end
                end
            end
        end
    end

    return states
end

function global_batch_execution(
    save_file::String,
    data_frame_test::DataFrame,
    data_frame_train::DataFrame,
    llmchats::Vector{LLMChat},
    input_types::Vector{String},
    approaches::Vector{String},
    k_folds::Int,
    exact_training_sizes::Vector{Int},
    num_close_samples::Int,
    num_far_samples::Int,
    num_runs::Int;
    random_split_seed::Union{Int,Missing}=42,
    save_chat_to_file_name::String="",
    mock::Bool=false,
    filter_test_molecules::Bool=false
)
    # 1. Create all states
    println("Generating experiment states...")
    all_states = create_experiment_states(
        data_frame_test, data_frame_train, llmchats, input_types, approaches,
        k_folds, exact_training_sizes, num_close_samples, num_far_samples,
        num_runs, random_split_seed;
        filter_test_molecules=filter_test_molecules
    )
    println("Total tasks to process: ", length(all_states))

    # Helper to detect batch mode
    function is_batch_llm(llmaccess::LLMAccess)
        d = hasproperty(llmaccess, :deployment) ? llmaccess.deployment : ""
        return contains(lowercase(d), "batch")
    end

    # Group states by unique LLM identifier
    llm_groups = Dict{String,Vector{ExperimentState}}()
    for state in all_states
        key = string(state.llmchat.llmaccess)
        if !ismissing(state.llmchat.reasoning_effort)
            key = string(state.llmchat.llmaccess) * "(" * state.llmchat.reasoning_effort * ")"
        end
        if !haskey(llm_groups, key)
            llm_groups[key] = []
        end
        push!(llm_groups[key], state)
    end

    for (llm_key, states) in llm_groups
        println("Processing group: $llm_key with $(length(states)) tasks")

        sample_llmaccess = states[1].llmchat.llmaccess
        use_batch = is_batch_llm(sample_llmaccess)
        println("  Mode: ", use_batch ? "BATCH API" : "PARALLEL ASYNC")

        # Split states by run_id for parallel execution
        run_groups = Dict{Int,Vector{ExperimentState}}()
        for state in states
            if !haskey(run_groups, state.run_id)
                run_groups[state.run_id] = []
            end
            push!(run_groups[state.run_id], state)
        end

        println("  Split into $(length(run_groups)) run groups for parallel execution.")

        @sync begin
            for (run_id, run_states) in run_groups
                #@async begin # async begin: comment out for threaded use, use for batch processing
                println("  [Run $run_id] Starting execution for $(length(run_states)) tasks...")

                # Execution loop for this group
                while true
                    # 1. Collect Active States (not done)
                    active_states = filter(s -> !s.done, run_states)
                    if isempty(active_states)
                        break
                    end

                    # 2. Prepare Inputs
                    llmchats_batch = LLMChat[]
                    states_in_batch = []

                    for state in active_states
                        # Get sequence of tasks
                        seq = get_prompt_sequence(state.input_data, state.approach)

                        if state.step_index > length(seq)
                            state.done = true
                            continue
                        end

                        task_key = seq[state.step_index]
                        raw_prompt = prompts[state.input_data][state.approach][task_key]
                        filled_prompt = fill_prompt(raw_prompt, state.data)

                        if state.llmchat.conversation[end][2] != filled_prompt
                            # Add cache hint for second-to-last step (prediction prompt)
                            if state.step_index == length(seq) - 1
                                push!(state.llmchat.conversation, ("user", filled_prompt, "ephemeral"))
                            else
                                push!(state.llmchat.conversation, ("user", filled_prompt))
                            end
                        end

                        # Deduplicate conversations
                        if !(state.llmchat.conversation in [chat.conversation for chat in llmchats_batch])
                            push!(llmchats_batch, state.llmchat)
                            push!(states_in_batch, state)
                        end
                    end

                    if isempty(llmchats_batch)
                        break
                    end

                    println("  [Run $run_id] Step batch size: $(length(llmchats_batch))")

                    # 3. MOCK or REAL EXECUTION
                    answers = []
                    if mock
                        answers = ["Mock response: " * chat.conversation[end][2] * "\n [42.5]" for chat in llmchats_batch]
                    else
                        if use_batch
                            answers = ask_gpt_batch(llmchats_batch; use_last_file_with_id=missing)
                        else
                            answers = ask_gpt_threaded(llmchats_batch; num_threads=50)
                        end
                    end

                    # 4. Update States
                    current_step_index = -2
                    for (i, ans) in enumerate(answers)
                        if ans == ""
                            continue
                        end
                        state = states_in_batch[i]

                        seq = get_prompt_sequence(state.input_data, state.approach)
                        task_key = seq[state.step_index]

                        push!(state.llmchat.conversation, ("assistant", ans))

                        # Check for prediction
                        if contains(task_key, "prediction")
                            val = search_for_last_number_in_string(ans)
                            state.prediction = val
                        end

                        state.step_index += 1
                        current_step_index = state.step_index
                        if state.step_index > length(seq)
                            state.done = true
                        end
                    end

                    # Update states that share conversations
                    for state in active_states
                        if state.llmchat.conversation[end][1] == "assistant"
                            # has already been updated
                            continue
                        end

                        ans = ""
                        for state_with_answer in states_in_batch
                            if state_with_answer.llmchat.conversation[1:end-1] == state.llmchat.conversation
                                ans = state_with_answer.llmchat.conversation[end][2]
                                push!(state.llmchat.conversation, ("assistant", ans))

                                if ans == ""
                                    @warn("Empty answer for state $(state.id)")
                                else
                                    if state.approach != state_with_answer.approach
                                        @warn("Approach mismatch for state $(state.id)")
                                    end
                                    if state.input_data != state_with_answer.input_data
                                        @warn("Input data mismatch for state $(state.id)")
                                    end
                                    break
                                end
                            end
                        end

                        if ans == ""
                            @warn("No answer found for state $(state.id)")
                            continue
                        end

                        seq = get_prompt_sequence(state.input_data, state.approach)
                        task_key = seq[state.step_index]

                        if contains(task_key, "prediction")
                            val = search_for_last_number_in_string(ans)
                            state.prediction = val
                        end

                        state.step_index += 1
                        if state.step_index > length(seq)
                            state.done = true
                        end
                    end
                end

                # 5. Save Results
                println("  [Run $run_id] Finished. Saving results...")

                finished_states = filter(s -> s.done, run_states)

                # Three-layer hierarchy for extended tracking
                keys_list = [
                    [s.input_data, s.approach, string(s.llmchat.llmaccess),
                        string(s.num_exact_samples), string(s.num_close_samples), string(s.num_far_samples),
                        s.data["molecule_name"]]
                    for s in finished_states
                ]
                vals = [s.prediction for s in finished_states]

                if !isempty(keys_list)
                    lock(FILE_WRITE_LOCK) do
                        try
                            append_values_to_json(save_file, keys_list, vals)

                            if save_chat_to_file_name != ""
                                chat_vals = [s.llmchat.conversation[end][2] for s in finished_states]
                                append_values_to_json(save_chat_to_file_name, keys_list, chat_vals)
                            end
                        catch e
                            @warn "Error saving results for Run $run_id: $e"
                        end
                    end
                end
                #end # async end
            end
        end
    end

    println("Done.")
end

function full_run_extended(
    save_file::String,
    data_frame::DataFrame,
    llmchats::Vector{LLMChat},
    input_types::Vector{String},
    approaches::Vector{String},
    k_folds::Int,
    exact_training_sizes::Vector{Int},
    data_frame_close_transfer::DataFrame,
    data_frame_far_transfer::DataFrame,
    num_runs::Int;
    save_chat_to_file_name::String="",
    random_split_seed::Union{Int,Missing}=42,
    mock::Bool=false,
    filter_test_molecules::Bool=false
)
    # Combine transfer learning DataFrames
    data_frame_train = vcat(data_frame_close_transfer, data_frame_far_transfer)

    # Get counts for tracking
    num_close_samples = nrow(data_frame_close_transfer)
    num_far_samples = nrow(data_frame_far_transfer)

    global_batch_execution(
        save_file, data_frame, data_frame_train, llmchats, input_types, approaches,
        k_folds, exact_training_sizes, num_close_samples, num_far_samples, num_runs;
        save_chat_to_file_name=save_chat_to_file_name,
        random_split_seed=random_split_seed,
        mock=mock,
        filter_test_molecules=filter_test_molecules
    )
end

function full_run(
    save_file::String,
    data_frame::DataFrame,
    llms::Vector{LLMChat},
    input_data_types::Vector{String},
    approaches::Vector{String},
    k_fold_cross_validation_k::Int,
    training_set_sizes::Vector{Int},
    num_runs::Int;
    save_chat_to_file_name::String="",
    random_split_seed::Union{Int,Missing}=42,
    mock::Bool=false
)
    # Wrapper for backward compatibility - uses empty DataFrames for no transfer learning
    full_run_extended(
        save_file, data_frame, llms, input_data_types, approaches,
        k_fold_cross_validation_k, training_set_sizes, DataFrame(), DataFrame(), num_runs;
        save_chat_to_file_name=save_chat_to_file_name,
        random_split_seed=random_split_seed,
        mock=mock
    )
end

# --- PROMPT INFRASTRUCTURE ---

prompts = Dict()

prompts["names_only"] = Dict()
prompts["names_and_descriptors"] = Dict()

prompts["names_only"]["input_output_prompting"] = Dict()
prompts["names_and_descriptors"]["input_output_prompting"] = Dict()

prompts["names_only"]["gpt_generated_prompts"] = Dict()
prompts["names_and_descriptors"]["gpt_generated_prompts"] = Dict()

prompts["names_only"]["refined_prompting"] = Dict()
prompts["names_and_descriptors"]["refined_prompting"] = Dict()

prompts["names_only"]["with_preanalysis"] = Dict()
prompts["names_and_descriptors"]["with_preanalysis"] = Dict()

# Blinding variants
prompts["names_only"]["wp_corrosion_blind"] = Dict()
prompts["names_only"]["wp_molproperty_clear"] = Dict()
prompts["names_only"]["wp_molproperty_blind"] = Dict()
prompts["names_only"]["wp_sampleproperty_clear"] = Dict()
prompts["names_only"]["wp_sampleproperty_blind"] = Dict()

# Helper functions

function fill_prompt(prompt::String, data::Dict)
    if !occursin("<", prompt) || !occursin(">", prompt)
        return prompt
    end
    if length(keys(data)) == 0
        @warn "No data provided to fill the prompt."
        return prompt
    end
    for key in keys(data)
        prompt = replace(prompt, "<" * key * ">" => string(data[key]))
    end
    if occursin("<", prompt) && occursin(">", prompt)
        missing_placeholders = collect(eachmatch(r"<[^>]+>", prompt))
        println("Warning: Not all placeholders were replaced in the prompt. Missing: ", [m.match for m in missing_placeholders])
    end
    return prompt
end

function fill_prompt(prompt::Vector{String}, data::Dict)
    filled_prompts = Vector{String}()
    for p in prompt
        push!(filled_prompts, fill_prompt(p, data))
    end
    return filled_prompts
end

function get_possible_tasks(input_data::String, approach::String)
    return keys(prompts[input_data][approach])
end

function search_for_last_number_in_string(str::String)
    # Regex handles: hyphen-minus (-), en-dash (–), and Unicode minus sign (−, U+2212)
    regex = r"(?:(?<=^)|(?<=[^0-9.]))([-–−]?(?:\d+(?:\.\d+)?|\.\d+))(?=[^0-9]|$)"
    matches = collect(eachmatch(regex, str))
    if !isempty(matches)
        m = matches[end].match
        try
            return parse(Float64, m)
        catch
        end
        try
            return parse(Float64, m * ".0")
        catch
        end
        # Handle en-dash
        try
            return parse(Float64, replace(m, "–" => "-"))
        catch
        end
        # Handle Unicode minus sign (U+2212)
        try
            return parse(Float64, replace(m, "−" => "-"))
        catch
        end
    end
    return NaN
end

function get_data_dict(data::DataFrame, training_set::Vector{Int}, test_set::Vector{Int})
    # Column names for corrosion data
    name_column = "IUPAC"
    smiles_column = "isomeric_SMILES"
    target_column = "IE"
    descriptors_columns = ["P_VSA_MR_5", "LUMO / eV", "E1p", "CATS3D_02_AP", "Mor04m"]

    data_dict = Dict()

    # Training set strings
    data_dict["molecule_names_training"] = "[" * join(data[training_set, name_column], ", ") * "]"
    data_dict["smiles_strings_training"] = "[" * join(data[training_set, smiles_column], ", ") * "]"
    data_dict["inhibition_efficiencies_training"] = "[" * join(data[training_set, target_column], ", ") * "]"

    # Test set strings
    data_dict["molecule_names_test"] = "[" * join(data[test_set, name_column], ", ") * "]"
    data_dict["smiles_strings_test"] = "[" * join(data[test_set, smiles_column], ", ") * "]"

    # Singles for prediction loop
    data_dict["molecule_names_test_single"] = data[test_set, name_column]
    data_dict["smiles_strings_test_single"] = data[test_set, smiles_column]

    # Add transformed columns for blinding experiments (if they exist)
    if "transformed_smiles" in names(data)
        data_dict["structure_strings_training"] = "[" * join(data[training_set, "transformed_smiles"], ", ") * "]"
        data_dict["structure_strings_test"] = "[" * join(data[test_set, "transformed_smiles"], ", ") * "]"
        data_dict["structure_strings_test_single"] = data[test_set, "transformed_smiles"]
    end

    if "transformed_IE" in names(data)
        data_dict["transformed_inhibition_efficiencies_training"] = "[" * join(data[training_set, "transformed_IE"], ", ") * "]"
    end

    # Experimental conditions (same for all samples in a batch)
    if !isempty(test_set)
        data_dict["base_material_symbol"] = data[test_set[1], "BaseMaterial"]
        data_dict["alloy_symbol"] = data[test_set[1], "Alloy"]
        data_dict["method"] = data[test_set[1], "Method"]
        data_dict["aggressive_component"] = data[test_set[1], "AggressiveComponent"]
        data_dict["operating_concentration_mM"] = data[test_set[1], "Operating_Concentration_mM"]

        # Base material name mapping
        bms2bm = Dict("Al" => "Aluminum", "Fe" => "Iron", "Mg" => "Magnesium", "Zn" => "Zinc", "Cu" => "Copper", "Ni" => "Nickel", "Ti" => "Titanium")
        data_dict["base_material"] = bms2bm[data_dict["base_material_symbol"]]

        # Add transformed materials for blinding experiments (if they exist)
        if "transformed_base_material" in names(data)
            data_dict["transformed_base_material"] = data[test_set[1], "transformed_base_material"]
        end
        if "transformed_alloy" in names(data)
            data_dict["transformed_alloy"] = data[test_set[1], "transformed_alloy"]
        end
    end

    # Descriptors (conditional on presence in DataFrame)
    has_descriptors = all(col -> col in names(data), descriptors_columns)
    if has_descriptors
        data_dict["descriptors_training"] = ""
        for column in descriptors_columns
            data_dict["descriptors_training"] *= column * ": [" * join(data[training_set, column], ", ") * "], "
        end

        data_dict["descriptors_test"] = ""
        for column in descriptors_columns
            data_dict["descriptors_test"] *= column * ": [" * join(data[test_set, column], ", ") * "], "
        end

        # Singles descriptors
        data_dict["descriptors_test_single"] = []
        for i in test_set
            descriptor_string = ""
            for column in descriptors_columns
                descriptor_string *= column * ": " * string(data[i, column]) * ", "
            end
            push!(data_dict["descriptors_test_single"], descriptor_string)
        end
    end

    return data_dict
end

# --- PROMPTS DEFINITIONS (from Corrosion_Prompts.jl) ---

script_dir = @__DIR__

include(joinpath(script_dir, "Corrosion_Prompts_Prompts_Only.jl"))
