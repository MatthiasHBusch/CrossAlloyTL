"""
Corrosion prompt definitions extracted from Corrosion_Prompts.jl
This file contains only the prompt definitions, no execution logic.
"""

# --- PROMPT DEFINITIONS ---

# Input/Output Prompting (names_only)
prompts["names_only"]["input_output_prompting"]["system"] = """
    **Your Role**
    You are a professional chemist with deep knowledge about organic chemistry and corrosion mechanisms.
    In the field of corrosion science, corrosion inhibitors are chemical compounds that, when added to the environment, significantly reduce the
    rate of corrosion. Corrosion accelerators are compounds that increase the rate of corrosion.
    Corrosion inhibition is critical for preventing the degradation of materials in industrial
    applications. Do follow the steps in the prompt step by step. Think systematically and structured.
    **Problem Description**
    You are tasked with predicting the inhibition efficiencies of an organic compound based on its molecular structure and the experimental
    conditions. You will have to use your expert knowledge and the information provided in the dataset to make predictions.
    The dataset contains molecular structures along with their respective inhibition efficiencies
    expressed in percentages (cannot be larger than 100), negative values express an acceleration of the corrosion process.
    It is important to note that the inhibition efficiency is not a direct measure of the compound's effectiveness,
    but instead a relative value to a control sample without any inhibitor and the exactly same experimental conditions. So the reference is different for each alloy and environment.
    You are tasked with predicting the inhibition efficiencies of the compounds in the test dataset. Additionally, the dataset contains information about the base material, alloy, method, aggressive component, and operating concentration in mM.
    Use this information to make your predictions.

    The effectiveness of an inhibitor depends on various properties.
    This problem involves predicting the inhibition efficiency of <base_material> (<base_material_symbol>, <alloy_symbol>) using a set of organic compounds.
    You will be provided with two datasets:
    A training dataset with names and SMILES labeled with inhibition efficiencies (<alloy_symbol>) to identify patterns and relationships between molecular structures and their corrosion inhibition efficiencies.
    A test dataset without labels, for which you will predict the inhibition efficiencies based on the patterns learned from the training dataset.
"""

prompts["names_only"]["input_output_prompting"]["prediction"] = """
    Training set:
    - Corrosion modulator molecule IUPAC names: <molecule_names_training>
    - Corrosion modulator molecule SMILES: <smiles_strings_training>
    - The measured inhibition efficiencies are: <inhibition_efficiencies_training>
    Based on the training set and/or your knowledge, predict the corrosion inhibition efficiency for the following molecule:
    - Corrosion modulator molecule IUPAC name: <molecule_name>
    - Corrosion modulator molecule SMILES: <smiles_string>
    The experimental conditions of all samples are:
    - Operating concentration of inhibitor in mM: <operating_concentration_mM>
    - Base material of corrosion sample: <base_material>
    - Alloy of corrosion sample: <alloy_symbol>
    - Method used for measuring the corrosion: <method>
    - Aggressive Component: <aggressive_component>
    The last sentence of your response must contain your prediction.
"""

# Input/Output Prompting (names_and_descriptors)
prompts["names_and_descriptors"]["input_output_prompting"]["system"] = """
    **Your Role**
    You are a professional chemist with deep knowledge about organic chemistry and corrosion mechanisms.
    The objective is to predict the corrosion inhibition efficiency of various organic compounds for <base_material> (<base_material_symbol>)
    in salt water. Corrosion inhibition efficiency is critical for preventing the degradation of materials in industrial
    applications. This dataset contains molecular structures along with their respective inhibition efficiencies
    expressed in percentages (cannot be larger than 100), negative values express an acceleration of the corrosion process.
    You are tasked with predicting the inhibition efficiencies of the compounds in the test dataset.
    do follow the steps in the prompt step by step.
    think systematically and structured.
    **Problem Description**
    In the field of corrosion science, corrosion inhibitors are chemical compounds that,
    when added to the environment in small concentrations, significantly reduce the rate of corrosion. Corrosion accelerators
    are compounds that increase the rate of corrosion.
    The effectiveness of an inhibitor depends on its molecular structure and its ability to interact with the metal surface.
    This problem involves predicting the inhibition efficiency of <base_material> (<base_material_symbol>, <alloy_symbol>) using a set of organic compounds.
    You will be provided with two datasets:
    A training dataset with names, SMILES and five descriptors labeled with inhibition efficiencies (ie_<alloy_symbol>) to identify patterns and relationships between molecular structures and their corrosion inhibition efficiencies.
    A test dataset without labels, for which you will predict the inhibition efficiencies based on the patterns learned from the training dataset.
"""

prompts["names_and_descriptors"]["input_output_prompting"]["prediction"] = """
    Training set:
    - Molecule Names: <molecule_names_training>
    - SMILES: <smiles_strings_training>
    - Descriptors: <descriptors_training>
    - Inhibition Efficiencies: <inhibition_efficiencies_training>
    Based on the training set, predict the corrosion inhibition efficiency for the following molecule:
    - Molecule Name: <molecule_name>
    - SMILES: <smiles_string>
    - Descriptors: <descriptors>
"""

# GPT-Generated Prompts (names_only)
prompts["names_only"]["gpt_generated_prompts"]["system"] = """
    You are a machine learning assistant specialized in predicting the corrosion reduction rates of <base_material> (<base_material_symbol>) in salty water when exposed to different organic additives. Your goal is to help researchers predict the corrosion reduction effect of new organic additives based on existing experimental data.

    ### Task:
    - You will be provided with **60 samples** in the form of:
      - The **organic additive** (described as a text or molecular structure in SMILES format).
      - The **corrosion reduction value** (a numerical representation of the additive's effect on corrosion speed, relative to a baseline).
    - Based on this dataset, predict the **corrosion reduction value** for new organic additives.

    ### Guidelines:
    1. **Understanding the Input Data:**
       - The additive's structure (in SMILES or textual description) determines its chemical properties.
       - The corrosion reduction value quantifies how effective the additive is at slowing down corrosion, with higher values indicating better performance.

    2. **Make Accurate Predictions:**
       - Use patterns and relationships from the given data to infer predictions for unseen additives.
       - Consider chemical properties like functional groups, molecular weight, and other structural features implicit in the additive representation.

    3. **Output Requirements:**
       - For each new additive, provide a single predicted corrosion reduction value (numerical).
       - Optionally, include a confidence score or reasoning if requested.

    4. **Contextual Behavior:**
       - If data patterns are unclear or predictions are uncertain, explain possible limitations in the dataset or suggest additional experiments to improve prediction accuracy.

    5. **Be Concise and Clear:**
       - Avoid unnecessary complexity in your responses.
       - Ensure your output is actionable and easy for researchers to interpret.

    ### Limitations:
    You do not perform real experiments or analyze raw chemical samples. Your predictions rely entirely on the patterns and correlations within the provided dataset.
"""

prompts["names_only"]["gpt_generated_prompts"]["analysis"] = """
    You are a machine learning model specialized in predicting the corrosion inhibition efficiency of <base_material> (<base_material_symbol>) in salty water with various organic additives. Below is a dataset of **60 experimental samples** in JSON format, containing three lists:
    1. `molecule_names`: Names of the additives.
    2. `smiles`: The SMILES representation of the additives, describing their molecular structure.
    3. `inhibition_efficiencies`: The corrosion reduction efficiency (%) for each additive, where higher values indicate better inhibition.

    Your task is to analyze the dataset to uncover patterns, trends, and relationships between the molecular structures and their inhibition efficiencies. Specifically:
    1. **Identify Key Structural Features:**
       - Determine which functional groups, molecular substructures, or other chemical properties correlate with higher or lower inhibition efficiencies.
       - Look for trends, such as the presence of certain groups (e.g., alcohols, amines, or aromatic rings) that might influence efficiency.

    2. **Establish Relationships:**
       - Describe any patterns between SMILES strings and inhibition efficiencies. For example, note if certain types of molecules consistently have higher or lower efficiencies.

    3. **Analyze Outliers:**
       - Identify and explain any samples with unusually high or low inhibition efficiencies relative to the rest of the dataset.

    4. **Summarize Insights:**
       - Provide a concise summary of your findings, highlighting the most important trends and chemical features.

    **Input:**
    ```json
    {
      "molecule_names": <molecule_names_training>,
      "smiles": <smiles_strings_training>,
      "inhibition_efficiencies": <inhibition_efficiencies_training>
    }
    ```

    **Output:**
    1. A detailed analysis of the trends and relationships in the dataset, linking structural features to inhibition efficiencies.
    2. A clear and actionable summary of the most important insights.

    Remember, this analysis will serve as the foundation for predicting the corrosion inhibition efficiency of new, unseen additives. Be precise and focus on useful patterns.
"""

prompts["names_only"]["gpt_generated_prompts"]["prediction"] = """
    ### Prediction Prompt:

    **Prompt:**
    You are a machine learning model specialized in predicting the corrosion inhibition efficiency of <base_material> (<base_material_symbol>) in salty water with various organic additives. Using your analysis of the 60 experimental samples provided earlier, analyze a new additive and predict its corrosion inhibition efficiency. Below is the information for the new additive:

    - `molecule_name`: <molecule_name>
    - `smiles`: <smiles_string>

    **Task:**
    1. **Analyze the New Additive:**
       - Examine the molecular structure provided in the SMILES string.
       - Compare it to the trends, key features, and relationships identified during the preemptive analysis.
       - Note similarities or differences between this additive and those in the experimental dataset.

    2. **Contextualize for <base_material_symbol> Corrosion:**
       - Evaluate how this additive's molecular properties might influence its ability to inhibit <base_material_symbol> corrosion in salty water.
       - Consider any relevant structural attributes, such as functional groups or molecular motifs, that were linked to inhibition efficiency in your earlier analysis.

    3. **Predict the Corrosion Inhibition Efficiency:**
       - Based on your analysis and the trends from the experimental dataset, predict the corrosion inhibition efficiency (in percentage) for the new additive.

    **Format your response as follows:**

    1. **Analysis of the Additive:**
       Provide a detailed comparison to the dataset, identifying key similarities and differences.

    2. **Implications for <base_material_symbol> Corrosion:**
       Describe how the molecular structure may contribute to the additive's performance.

    3. **Predicted Inhibition Efficiency:**
       Conclude with a specific numeric prediction for the inhibition efficiency (%) of this additive.
"""

# GPT-Generated Prompts (names_and_descriptors)
prompts["names_and_descriptors"]["gpt_generated_prompts"]["system"] = """
    **System Role:**
    You are a machine learning assistant specializing in predicting the corrosion inhibition efficiency of <base_material> (<base_material_symbol>) in salty water with various organic additives. Your task is to interpret experimental data, identify trends, and make predictions about new additives based on their molecular structures and descriptors.

    The experimental dataset consists of 60 samples, each represented by:
    1. **Molecule Name**: A unique identifier for each additive.
    2. **SMILES String**: A structural representation of the molecule.
    3. **Molecular Descriptors**: A set of 5 numerical values capturing key properties of the molecule:
       - **P_VSA_MR_5**: Descriptor for van der Waals surface area and molar refractivity.
       - **LUMO / eV**: Energy level of the lowest unoccupied molecular orbital, in electronvolts.
       - **E1p**: A property indicating electronic interaction potential.
       - **CATS3D_02_AP**: A 3D topological descriptor associated with pharmacophore alignment.
       - **Mor04m**: A 2D descriptor from the MOLECULAR RADII family measuring molecule shape.
    4. **Inhibition Efficiency**: A percentage reduction in corrosion rate due to the additive.

    ### Responsibilities:
    1. **Analyze Experimental Data:** Identify trends and relationships between molecular descriptors, structural features (via SMILES), and inhibition efficiencies. Consider how each descriptor correlates with the inhibition efficiency and how structural features might influence <base_material_symbol> corrosion.
    2. **Predict Corrosion Inhibition Efficiency:** Use the insights from your analysis to predict the inhibition efficiency of new additives, considering their molecular descriptors and structures.

    When making predictions, ensure that you:
    - Draw connections between the molecular descriptors of the new additive and the patterns observed in the dataset.
    - Compare the structural features of the new additive to similar molecules in the dataset.
    - Justify predictions based on chemical properties known to affect <base_material_symbol> corrosion in salty water.

    **Key Objective:**
    Provide accurate and well-supported predictions for the corrosion inhibition efficiency (%) of new additives based on their molecular descriptors and structures. Aim for clear reasoning and a detailed explanation for each prediction.
"""

prompts["names_and_descriptors"]["gpt_generated_prompts"]["analysis2_names_smiles"] = """
    Here are 60 molecular names, their SMILES strings, and their measured inhibition efficiencies for <base_material_symbol> corrosion in salty water:
    ```json
    {
        "molecule_names": <molecule_names_training>,
        "smiles_strings": <smiles_strings_training>,
        "inhibition_efficiencies": <inhibition_efficiencies_training>
    }
    ```
    Analyze the molecular names and SMILES strings with respect to their inhibition efficiencies. Look for:
    1. Patterns in naming conventions, prefixes, or suffixes and their relationship to inhibition efficiencies.
    2. Structural features or chemical groups identifiable from SMILES strings that correlate with higher or lower inhibition efficiencies.
    3. Clustering of molecules with similar names, SMILES characteristics, and their associated efficiencies.

    Provide a concise summary of:
    - General patterns across names and SMILES strings.
    - Any significant correlations between naming/structural features and inhibition efficiency.

    Do not perform any prediction at this stage. Focus solely on the analysis of names and SMILES.
"""

prompts["names_and_descriptors"]["gpt_generated_prompts"]["analysis1_descriptors"] = """
    Here are the molecular descriptors and measured inhibition efficiencies for 60 additives:
    ```json
    {
        "names": <molecule_names_training>,
        "descriptors": <descriptors_training>,
        "inhibition_efficiencies": <inhibition_efficiencies_training>
    }
    ```
    Analyze the given descriptors and their relationships with the inhibition efficiencies. Specifically:
    1. Identify which descriptors appear to correlate most strongly with higher or lower inhibition efficiencies.
    2. Highlight any potential trends, such as combinations of descriptor ranges (e.g., high P_VSA_MR_5 with low LUMO values) associated with specific efficiency outcomes.
    3. Determine whether clustering based on the descriptors is evident, and note any observable patterns or anomalies.

    Provide a concise summary of:
    - Descriptor(s) most indicative of high or low inhibition efficiencies.
    - Key trends or clusters visible in the data.
    - Any outliers or exceptions that deviate from the observed trends.

    Focus solely on the descriptors and inhibition efficiencies.
"""

prompts["names_and_descriptors"]["gpt_generated_prompts"]["summary"] = """
    We have analyzed the following data for 60 additives:
    1. Molecular names and SMILES strings: Previously analyzed for structural and functional features linked to inhibition efficiency.
    2. Molecular descriptors (P_VSA_MR_5, LUMO / eV, E1p, CATS3D_02_AP, Mor04m) and their relationships to inhibition efficiencies.

    Summarize the combined findings:
    1. Key structural or functional groups (from the names/SMILES analysis) linked to high or low inhibition efficiencies.
    2. Descriptor trends most indicative of inhibition efficiency and how they align with the structural/functional insights.
    3. Synergies between structural features and descriptor values that contribute to high efficiency.
    4. Any conflicting or complementary patterns from the two analyses.

    Provide a concise yet comprehensive summary that integrates insights from both analyses and highlights their implications for predicting new samples.
"""

prompts["names_and_descriptors"]["gpt_generated_prompts"]["prediction"] = """
    We are testing a new additive, which has the following characteristics:
    - **Name:** <molecule_name>
    - **SMILES:** <smiles_string>
    - **Descriptors:**  <descriptors>

    Using the insights gained from the analyses of the 60 experimental samples:
    1. Analyze the structure and functional groups of the additive based on its name and SMILES string.
    2. Compare its descriptors to those of the experimental samples, identifying patterns or similarities to high- or low-efficiency samples.
    3. Synthesize these findings to predict the corrosion inhibition efficiency of this additive for <base_material_symbol> in a saline environment.

    Provide:
    - A brief explanation of how the additive relates to the previously analyzed samples.
    - The predicted inhibition efficiency (in percentage).
    - Justification for the prediction, highlighting key structural and descriptor-based factors.
"""

# With Preanalysis (names_only)
prompts["names_only"]["with_preanalysis"]["system"] = prompts["names_only"]["input_output_prompting"]["system"]

prompts["names_only"]["with_preanalysis"]["analysis"] = """
    This is the training data:
    - Corrosion modulator molecule IUPAC names: <molecule_names_training>
    - Corrosion modulator molecule SMILES: <smiles_strings_training>
    - The measured inhibition efficiencies are: <inhibition_efficiencies_training>

    The experimental conditions of all samples are:
    - Operating concentration of inhibitor in mM: <operating_concentration_mM>
    - Base material of corrosion sample: <base_material>
    - Alloy of corrosion sample: <alloy_symbol>
    - Method used for measuring the corrosion: <method>
    - Aggressive Component: <aggressive_component>

    Analysis of the training data:
    Step 1: Write down all functional groups and other chemical properties you know for each sample of the training data. Analyze their influence on the inhibition efficiency.
    Step 2: Analyze the atomic structures of the compounds and their influence on the inhibition efficiency.
    Step 3: Find compounds in the training data that are similar but have different inhibition efficiencies.
    List them. Explain, why these differences lead to different inhibition efficiencies.
    Use a systematic approach and think step by step.
"""

prompts["names_only"]["with_preanalysis"]["prediction"] = """
    This is the test data:
    - Corrosion modulator molecule IUPAC names: <molecule_names_test>
    - Corrosion modulator molecule SMILES: <smiles_strings_test>

    This is the molecule you have to predict the inhibition efficiency for:
    - Corrosion modulator molecule IUPAC name: <molecule_name>
    - Corrosion modulator molecule SMILES: <smiles_string>

    The experimental conditions of all samples are:
    - Operating concentration of inhibitor in mM: <operating_concentration_mM>
    - Base material of corrosion sample: <base_material>
    - Alloy of corrosion sample: <alloy_symbol>
    - Method used for measuring the corrosion: <method>
    - Aggressive Component: <aggressive_component>

    Step by step guide to predict the inhibition efficiency:

    ---

    **Similar Molecules Relations**
    1. Similar Molecules: {Find all similar molecules in the training data and analyze their relation to this compound wrt the <base_material> corrosion
    process. Use the training data and the analyzed training data.}
    2. Similarity Analysis: {Analyze similar molecules and rank them by similarity (wrt the mechanisms in the corrosion process).
    Assign them a similarity value (wrt the mechanisms in the corrosion process).}

    **Molecule Analysis**
    3. Pattern Analysis: {Analyze if found patterns apply to the molecule}
    4. Functional Groups: {Analyze its functional groups and their influnce on the inhibition efficiency}
    5. Atomic Structure: {Analyze its atomic structure and how this might influence the inhibition efficiency}
    6. Weighted Average: {Calculate a weighted average of the inhibition efficiencies of the similar molecules.
    Exclude molecules that have a small similarity value.}

    **Review and Prediction**
    7. Prediction: {Review your analysis shortly and write down the weighted average.}
    8. Result: {As a result, write down one value and nothing after that. Syntax: "[Value]"}
"""

# ==================================================================================
# BLINDING VARIANTS FOR "with_preanalysis" (names_only)
# Based on QM7-Lipophilicity-Delaney blinding strategy - progressively hiding domain information
# ==================================================================================

# --- Variant 2: wp_corrosion_blind ---
# Blinds the specific property: "corrosion inhibition" → "molecular property related to corrosion"
prompts["names_only"]["wp_corrosion_blind"]["system"] = """
    **Your Role**
    You are a professional chemist with deep knowledge about organic chemistry and material science.
    You are tasked with predicting a molecular property related to corrosion modulation for metal alloys based on molecular structures and experimental conditions.

    **Problem Description**
    You are tasked with predicting the values of a molecular property related to corrosion modulation of an organic compound based on its molecular structure and the experimental conditions. You will have to use your expert knowledge and the information provided in the dataset to make predictions.
    The dataset contains molecular structures along with their respective property values expressed in percentages (cannot be larger than 100), negative values express a negative effect.
    It is important to note that the property value is not a direct measure of the compound's effectiveness, but instead a relative value to a control sample without any additive and the exactly same experimental conditions. So the reference is different for each material and environment.
    You are tasked with predicting the property values of the compounds in the test dataset. Additionally, the dataset contains information about the base material, alloy, method, aggressive component, and operating concentration in mM.
    Use this information to make your predictions.

    The effectiveness depends on various molecular properties.
    This problem involves predicting the molecular property related to corrosion modulation of <base_material> (<base_material_symbol>, <alloy_symbol>) using a set of organic compounds.
    You will be provided with two datasets:
    A training dataset with names and SMILES labeled with property values (<alloy_symbol>) to identify patterns and relationships between molecular structures and their property values.
    A test dataset without labels, for which you will predict the property values based on the patterns learned from the training dataset.
"""

prompts["names_only"]["wp_corrosion_blind"]["analysis"] = """
    This is the training data:
    - Organic compound IUPAC names: <molecule_names_training>
    - Organic compound SMILES: <smiles_strings_training>
    - The measured molecular property related to corrosion modulation: <inhibition_efficiencies_training>

    The experimental conditions of all samples are:
    - Operating concentration of compound in mM: <operating_concentration_mM>
    - Base material of sample: <base_material>
    - Alloy of sample: <alloy_symbol>
    - Method used for measuring: <method>
    - Aggressive Component: <aggressive_component>

    Analysis of the training data:
    Step 1: Write down all functional groups and other chemical properties you know for each sample of the training data. Analyze their influence on the molecular property related to corrosion modulation.
    Step 2: Analyze the atomic structures of the compounds and their influence on the molecular property related to corrosion modulation.
    Step 3: Find compounds in the training data that are similar but have different property values.
    List them. Explain, why these differences lead to different property values.
    Use a systematic approach and think step by step.
"""

prompts["names_only"]["wp_corrosion_blind"]["prediction"] = """
    This is the test data:
    - Organic compound IUPAC names: <molecule_names_test>
    - Organic compound SMILES: <smiles_strings_test>

    This is the molecule you have to predict the property value for:
    - Organic compound IUPAC name: <molecule_name>
    - Organic compound SMILES: <smiles_string>

    The experimental conditions of all samples are:
    - Operating concentration of compound in mM: <operating_concentration_mM>
    - Base material of sample: <base_material>
    - Alloy of sample: <alloy_symbol>
    - Method used for measuring: <method>
    - Aggressive Component: <aggressive_component>

    Step by step guide to predict the property value:

    ---

    **Similar Molecules Relations**
    1. Similar Molecules: Find all similar molecules in the training data and analyze their relation to this compound wrt the molecular property related to corrosion modulation. Use the training data and the analyzed training data.
    2. Similarity Analysis: Analyze similar molecules and rank them by similarity (wrt the mechanisms found in the analysis). Assign them a similarity value (wrt the mechanisms found in the analysis).

    **Molecule Analysis**
    3. Pattern Analysis: Analyze if found patterns apply to the molecule.
    4. Functional Groups: Analyze its functional groups and their influence on the molecular property related to corrosion modulation.
    5. Atomic Structure: Analyze its atomic structure and how this might influence the molecular property related to corrosion.
    6. Weighted Average: Calculate a weighted average of the property values of the similar molecules. Exclude molecules that have a small similarity value.

    **Review and Prediction**
    7. Prediction: Review your analysis shortly and write down the weighted average.
    8. Result: As a result, write down one value and nothing after that. Syntax: "[Value]"
"""

# --- Variant 3: wp_molproperty_clear ---
# More generic: "molecular property" (removes corrosion context entirely)
prompts["names_only"]["wp_molproperty_clear"]["system"] = """
    **Your Role**
    You are a professional chemist with expert knowledge in organic chemistry.
    You are tasked with predicting a molecular property based on their SMILES strings and experimental conditions.

    **Problem Description**
    You will be provided with:
    1. A training dataset of molecules with their SMILES and IUPAC names.
    2. Experimental conditions including a related material, alloy, and operating concentration.
    3. A test molecule (SMILES) for which you must predict the molecular property.

    The property values are expressed in percentages (cannot be larger than 100), negative values are possible.
    The property value is relative to a control sample without any additive under the same experimental conditions.

    You have to use your knowledge and abilities to analyze the training data molecules and the patterns and relationships between molecular properties to make an accurate prediction.
"""

prompts["names_only"]["wp_molproperty_clear"]["analysis"] = """
    **Training Data:**
    - Molecule names: <molecule_names_training>
    - Molecule SMILES: <smiles_strings_training>
    - Molecular property values: <inhibition_efficiencies_training>

    **Experimental Conditions:**
    - Operating concentration in mM: <operating_concentration_mM>
    - Material: <base_material>
    - Alloy: <alloy_symbol>

    **Analysis Task:**
    1. Identify functional groups and structural features for each training sample.
    2. Analyze how these features influence the molecular property.
    3. Find pairs of similar molecules with different property values and explain the difference.

    Think step by step and provide a systematic analysis of the training data patterns.
"""

prompts["names_only"]["wp_molproperty_clear"]["prediction"] = """
    **Test Data:**
    - Molecule names: <molecule_names_test>
    - Molecule SMILES: <smiles_strings_test>

    **Target Molecule:**
    - Molecule name: <molecule_name>
    - Molecule SMILES: <smiles_string>

    **Experimental Conditions:**
    - Operating concentration in mM: <operating_concentration_mM>
    - Material: <base_material>
    - Alloy: <alloy_symbol>

    **Prediction Guide:**

    **Similar Molecules Relations**
    1. Similar Molecules: Find all similar molecules in the training data and analyze their relation to this compound wrt the molecular property. Use the training data and the analyzed training data.
    2. Similarity Analysis: Analyze similar molecules and rank them by similarity (wrt the mechanisms found in the analysis). Assign them a similarity value (wrt the mechanisms found in the analysis).

    **Molecule Analysis**
    3. Pattern Analysis: Analyze if found patterns apply to the molecule
    4. Functional Groups: Analyze its functional groups and their influence on the molecular property
    5. Atomic Structure: Analyze its atomic structure and how this might influence the molecular property
    6. Weighted Average: Calculate a weighted average of the molecular properties of the similar molecules. Exclude molecules that have a small similarity value.

    **Review and Prediction**
    7. Prediction: Review your analysis shortly and write down the weighted average.
    8. Result: As a result, write down one value and nothing after that. Syntax: "[Value]"
"""

# --- Variant 4: wp_molproperty_blind ---
# Generic "molecular property" + will use transformed/blinded target values in data preparation
prompts["names_only"]["wp_molproperty_blind"]["system"] = prompts["names_only"]["wp_molproperty_clear"]["system"]

prompts["names_only"]["wp_molproperty_blind"]["analysis"] = """
    **Training Data:**
    - Molecule names: <molecule_names_training>
    - Molecule SMILES: <smiles_strings_training>
    - Molecular property values: <transformed_inhibition_efficiencies_training>

    **Experimental Conditions:**
    - Operating concentration in mM: <operating_concentration_mM>
    - Base material: <base_material>
    - Alloy: <alloy_symbol>

    **Analysis Task:**
    1. Identify functional groups and structural features for each training sample.
    2. Analyze how these features influence the molecular property.
    3. Find pairs of similar molecules with different property values and explain the difference.

    Think step by step and provide a systematic analysis of the training data patterns.
"""

prompts["names_only"]["wp_molproperty_blind"]["prediction"] = prompts["names_only"]["wp_molproperty_clear"]["prediction"]

# --- Variant 5: wp_sampleproperty_clear ---
# Fully abstract: "sample property" + "structure representation string" instead of SMILES
# Removes all chemistry-specific terminology
prompts["names_only"]["wp_sampleproperty_clear"]["system"] = """
    **Your Role**
    You are a professional machine learning model with expert knowledge in regression.
    You are tasked with predicting a sample property based on a string based structure representation of the sample and experimental conditions.

    **Problem Description**
    You will be provided with:
    1. A training dataset of samples with their string based structure representation and known sample property values.
    2. Experimental conditions for each sample.
    3. A test sample (string based structure representation) for which you must predict the sample property.

    The property values are numerical.
    The property value is relative to a control sample under the same experimental conditions.

    You have to use your knowledge and abilities to analyze the training data samples and the patterns and relationships between sample properties to make an accurate prediction.
"""

prompts["names_only"]["wp_sampleproperty_clear"]["analysis"] = """
    **Training Data:**
    - Sample structure strings: <structure_strings_training>
    - Sample property values: <inhibition_efficiencies_training>

    **Experimental Conditions:**
    - Parameter 1: <operating_concentration_mM>
    - Parameter 2: <transformed_base_material>
    - Parameter 3: <transformed_alloy>

    **Analysis Task:**
    1. Identify structural features for each training sample.
    2. Analyze how these features influence the sample property.
    3. Find pairs of similar samples with different sample properties and explain the difference.

    Think step by step and provide a systematic analysis of the training data patterns.
"""

prompts["names_only"]["wp_sampleproperty_clear"]["prediction"] = """
    **Test Data:**
    - Sample structure strings: <structure_strings_test>

    **Target Sample:**
    - Sample structure string: <structure_string>

    **Experimental Conditions:**
    - Parameter 1: <operating_concentration_mM>
    - Parameter 2: <transformed_base_material>
    - Parameter 3: <transformed_alloy>

    **Prediction Guide:**

    **Similar Sample Relations**
    1. Similar Samples: Find all similar samples in the training data and analyze their relation to this sample wrt the sample property. Use the training data and the analyzed training data.
    2. Similarity Analysis: Analyze similar samples and rank them by similarity (wrt the mechanisms found in the analysis). Assign them a similarity value (wrt the mechanisms found in the analysis).

    **Sample Analysis**
    3. Pattern Analysis: Analyze if found patterns apply to the sample.
    4. Structural Features: Analyze its structural features and their influence on the sample property.
    5. Weighted Average: Calculate a weighted average of the sample properties of the similar samples. Exclude samples that have a small similarity value.

    **Review and Prediction**
    7. Prediction: Review your analysis shortly and write down the weighted average.
    8. Result: As a result, write down one value and nothing after that. Syntax: "[Value]"
"""

# --- Variant 6: wp_sampleproperty_blind ---
# Fully abstract + transformed/blinded target values
prompts["names_only"]["wp_sampleproperty_blind"]["system"] = prompts["names_only"]["wp_sampleproperty_clear"]["system"]

prompts["names_only"]["wp_sampleproperty_blind"]["analysis"] = """
    **Training Data:**
    - Sample structure strings: <structure_strings_training>
    - Sample property values: <transformed_inhibition_efficiencies_training>

    **Experimental Conditions:**
    - Parameter 1: <operating_concentration_mM>
    - Parameter 2: <transformed_base_material>
    - Parameter 3: <transformed_alloy>

    **Analysis Task:**
    1. Identify structural features for each training sample.
    2. Analyze how these features influence the sample property.
    3. Find pairs of similar samples with different sample properties and explain the difference.

    Think step by step and provide a systematic analysis of the training data patterns.
"""

prompts["names_only"]["wp_sampleproperty_blind"]["prediction"] = prompts["names_only"]["wp_sampleproperty_clear"]["prediction"]


# ==================================================================================
# SIMILARITY QUERY — Active Learning approach
# Returns similar training samples + similarity weights as JSON instead of a prediction
# ==================================================================================

prompts["names_only"]["similarity_query"] = Dict()

prompts["names_only"]["similarity_query"]["system"] = prompts["names_only"]["with_preanalysis"]["system"]

prompts["names_only"]["similarity_query"]["analysis"] = """
    This is the training data:
    - Corrosion modulator molecule IUPAC names: <molecule_names_training>
    - Corrosion modulator molecule SMILES: <smiles_strings_training>
    - The measured inhibition efficiencies are: <inhibition_efficiencies_training>

    The experimental conditions of all samples are:
    - Operating concentration of inhibitor in mM: <operating_concentration_mM>
    - Base material of corrosion sample: <base_material>
    - Alloy of corrosion sample: <alloy_symbol>
    - Method used for measuring the corrosion: <method>
    - Aggressive Component: <aggressive_component>

    Analysis of the training data:
    Step 1: Write down all functional groups and other chemical properties you know for each sample of the training data. Analyze their influence on the inhibition efficiency.
    Step 2: Analyze the atomic structures of the compounds and their influence on the inhibition efficiency.
    Step 3: Find compounds in the training data that are similar but have different inhibition efficiencies.
    List them. Explain, why these differences lead to different inhibition efficiencies.
    Use a systematic approach and think step by step.
"""

prompts["names_only"]["similarity_query"]["prediction"] = """
    This is the test data:
    - Corrosion modulator molecule IUPAC names: <molecule_names_test>
    - Corrosion modulator molecule SMILES: <smiles_strings_test>

    This is the molecule you have to predict the inhibition efficiency for:
    - Corrosion modulator molecule IUPAC name: <molecule_name>
    - Corrosion modulator molecule SMILES: <smiles_string>

    The experimental conditions of all samples are:
    - Operating concentration of inhibitor in mM: <operating_concentration_mM>
    - Base material of corrosion sample: <base_material>
    - Alloy of corrosion sample: <alloy_symbol>
    - Method used for measuring the corrosion: <method>
    - Aggressive Component: <aggressive_component>

    Step by step guide to predict the inhibition efficiency:

    ---

    **Similar Molecules Relations**
    1. Similar Molecules: {Find all similar molecules in the training data and analyze their relation to this compound wrt the <base_material> corrosion
    process. Use the training data and the analyzed training data.}
    2. Similarity Analysis: {Analyze similar molecules and rank them by similarity (wrt the mechanisms in the corrosion process).
    Assign them a similarity value (wrt the mechanisms in the corrosion process).}

    **Molecule Analysis**
    3. Pattern Analysis: {Analyze if found patterns apply to the molecule}
    4. Functional Groups: {Analyze its functional groups and their influnce on the inhibition efficiency}
    5. Atomic Structure: {Analyze its atomic structure and how this might influence the inhibition efficiency}

    **Result**
    6. Output: {Now write down the similar molecules with their similarites so I can calculate a weighted average as prediction. 
    Exclude molecules that have a small similarity value. Return ONLY the selected similar molecules as a JSON array. 
    Each entry must have the fields: smiles, alloy, ie, similarity.
    Use the exact SMILES string and IE value from the training data. The similarity value should be between 0 and 1.

    Your response for this step MUST contain a JSON code block in exactly this format:
    ```json
    [
      {"smiles": "<exact SMILES from training>", "alloy": "<alloy>", "ie": <IE value from training>, "similarity": <0-1>},
      ...
    ]
    ```
    Do not write anything after the JSON block.}
"""


# -------------------------------------------
# With Preanalysis (names_and_descriptors)
prompts["names_and_descriptors"]["with_preanalysis"]["system"] = prompts["names_and_descriptors"]["input_output_prompting"]["system"]

prompts["names_and_descriptors"]["with_preanalysis"]["analysis2_names_smiles"] = """
    This is the training data:
    - Molecule Names: <molecule_names_training>
    - SMILES: <smiles_strings_training>
    - Inhibition Efficiencies: <inhibition_efficiencies_training>

    Analysis of the training data:
    Write down all functional groups and atomic structures together with their inhibition efficiency for each sample of the training data in a list.
"""

prompts["names_and_descriptors"]["with_preanalysis"]["analysis1_descriptors"] = """
    This is the training data:
    - Molecule Names: <molecule_names_training>
    - SMILES: <smiles_strings_training>
    - Descriptors: <descriptors_training>
    - Inhibition Efficiencies: <inhibition_efficiencies_training>

    Analysis of the training data:
    Write down the values and names of 2-3 non zero descriptors together with their inhibition efficiency for each sample of the training data in a list.
"""

prompts["names_and_descriptors"]["with_preanalysis"]["summary_1"] = """
    Create a list where you combine the functional groups and atomic structures with the non zero descriptors and their inhibition efficiency for each sample of the training data.
"""

prompts["names_and_descriptors"]["with_preanalysis"]["summary_2"] = """
    Now search for patterns in the previously created list and analyze the influence of the functional groups, atomic structures and non zero descriptors on the inhibition efficiency.
"""

prompts["names_and_descriptors"]["with_preanalysis"]["prediction"] = """
    This is the test data:
    - Molecule Names: <molecule_names_test>
    - SMILES: <smiles_strings_test>
    - Descriptors: <descriptors_test>

    This is the molecule you have to predict the inhibition efficiency for:
    - Molecule Name: <molecule_name>
    - SMILES: <smiles_string>
    - Descriptors: <descriptors>

    Step by step guide to predict the inhibition efficiency:

    ---

    **Similar Molecules Relations**
    1. Similar Molecules: {Find all similar molecules in the training data and analyze their relation to this compound wrt the <base_material> corrosion
    process. Use the training data and the analyzed training data.}
    2. Similarity Analysis: {Analyze similar molecules and rank them by similarity (wrt the mechanisms in the corrosion process).
    Assign them a similarity value (wrt the mechanisms in the corrosion process).}

    **Molecule Analysis**
    3. Pattern Analysis: {Analyze if found patterns apply to the molecule}
    4. Functional Groups: {Analyze its functional groups and their influence on the inhibition efficiency}
    5. Atomic Structure: {Analyze its atomic structure and how this might influence the inhibition efficiency}
    6. Weighted Average: {Calculate a weighted average of the inhibition efficiencies of the similar molecules.
    Exclude molecules that have a small similarity value.}

    **Review and Prediction**
    7. Prediction: {Review your analysis shortly and write down the weighted average.}
    8. Result: {As a result, write down one value and nothing after that. Syntax: "[Value]"}
"""

# Refined Prompting (names_only)
prompts["names_only"]["refined_prompting"]["system"] = prompts["names_only"]["input_output_prompting"]["system"]

prompts["names_only"]["refined_prompting"]["prediction"] = """
    This is the training data:
    - Molecule Names: <molecule_names_training>
    - SMILES: <smiles_strings_training>
    - Inhibition Efficiencies: <inhibition_efficiencies_training>

    This is the test data:
    - Molecule Names: <molecule_names_test>
    - SMILES: <smiles_strings_test>

    This is the molecule you have to predict the inhibition efficiency for:
    - Molecule Name: <molecule_name>
    - SMILES: <smiles_string>

    Step by step guide to predict the inhibition efficiency:

    ---

    **Molecule Analysis**
    1. Functional Groups: {Analyze the test molecules' functional groups and by including the training data, analyze their influence on the inhibition efficiency}
    2. Atomic Structure: {Analyze the test molecules' atomic structure and by including the training data, analyze their influence on the inhibition efficiency}
    3. Educated Guess: {Make an educated guess for the inhibition/acceleration efficiency of this compound. Take one of these values: [-200, -100, -50, 0, 25, 50, 75]}

    **Similar Molecules Relations**
    4. Similar Molecules: {Find all similar molecules in the training data and analyze their relation to this compound wrt the <base_material> corrosion
    process. Use the training data and the analyzed training data.}
    5. Similarity Analysis: {Analyze similar molecules and rank them by similarity (wrt the mechanisms in the corrosion process).
    Assign them a similarity value (wrt the mechanisms in the corrosion process).}
    6. Weighted Average: {Calculate a weighted average of the inhibition efficiencies of the similar molecules.
    Exclude molecules that have a small similarity value.}

    **Review and Prediction**
    7. Prediction: {Review your analysis shortly and write down your prediction. There were two ways predicting the inhibition efficiency.
    Decide for one way and use only this way.}
    8. Result: {As a result, write down one value and nothing after that. Syntax: "[Value]"}
"""

# Refined Prompting (names_and_descriptors)
prompts["names_and_descriptors"]["refined_prompting"]["system"] = prompts["names_and_descriptors"]["input_output_prompting"]["system"]

prompts["names_and_descriptors"]["refined_prompting"]["prediction"] = """
    This is the training data:
    - Molecule Names: <molecule_names_training>
    - SMILES: <smiles_strings_training>
    - Descriptors: <descriptors_training>
    - Inhibition Efficiencies: <inhibition_efficiencies_training>

    This is the test data:
    - Molecule Names: <molecule_names_test>
    - SMILES: <smiles_strings_test>
    - Descriptors: <descriptors_test>

    This is the molecule you have to predict the inhibition efficiency for:
    - Molecule Name: <molecule_name>
    - SMILES: <smiles_string>
    - Descriptors: <descriptors>

    Step by step guide to predict the inhibition efficiency:

    ---

    **Molecule Analysis**
    1. Functional Groups: {Analyze the test molecules' functional groups and by including the training data, analyze their influence on the inhibition efficiency}
    2. Atomic Structure: {Analyze the test molecules' atomic structure and by including the training data, analyze their influence on the inhibition efficiency}
    3. Educated Guess: {Make an educated guess for the inhibition/acceleration efficiency of this compound. Take one of these values: [-200, -100, -50, 0, 25, 50, 75]}

    **Similar Molecules Relations**
    4. Similar Molecules: {Find all similar molecules in the training data and analyze their relation to this compound wrt the <base_material> corrosion
    process. Use the training data and the analyzed training data.}
    5. Similarity Analysis: {Analyze similar molecules and rank them by similarity (wrt the mechanisms in the corrosion process).
    Assign them a similarity value (wrt the mechanisms in the corrosion process).}
    6. Weighted Average: {Calculate a weighted average of the inhibition efficiencies of the similar molecules.
    Exclude molecules that have a small similarity value.}

    **Review and Prediction**
    7. Prediction: {Review your analysis shortly and write down your prediction. There were two ways predicting the inhibition efficiency.
    Decide for one way and use only this way.}
    8. Result: {As a result, write down one value and nothing after that. Syntax: "[Value]"}
"""
