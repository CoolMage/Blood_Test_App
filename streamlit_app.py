import streamlit as st
import fitz
import pandas as pd
import io
import plotly.graph_objects as go
from scipy import stats
import glob
import os




# Streamlit app title
st.title("PDF Data Extractor and Blood Plot Comparison")

# Function to load and combine data from multiple CSV files
def load_data(files):
    combined_data = pd.DataFrame()
    for file in files:
        df = pd.read_csv(file)
        combined_data = pd.concat([combined_data, df], ignore_index=True)
    return combined_data

# Function to prepare data for a specific test
def prepare_test_data(data_group1, data_group2, test_name):
    pattern = r"[<> ]"
    group1_values = data_group1[data_group1['Test'] == test_name]['Výsledek']
    group2_values = data_group2[data_group2['Test'] == test_name]['Výsledek']

    # Convert to string first to avoid issues with non-string values
    group1_values = group1_values.astype(str)
    group2_values = group2_values.astype(str)

    # Replace commas with dots and convert to numeric
    group1_values = pd.to_numeric(group1_values.str.replace(",", ".", regex=False), errors='coerce').dropna()
    group2_values = pd.to_numeric(group2_values.str.replace(",", ".", regex=False), errors='coerce').dropna()

    # Remove unwanted characters
    group1_values = pd.to_numeric(group1_values.astype(str).str.replace(pattern, "", regex=True), errors='coerce').dropna()
    group2_values = pd.to_numeric(group2_values.astype(str).str.replace(pattern, "", regex=True), errors='coerce').dropna()

    
    return group1_values, group2_values

# Function to perform statistical test
def perform_statistical_test(values1, values2):
    if len(values1) > 0 and len(values2) > 0:
        _, p_value = stats.mannwhitneyu(values1, values2, alternative='two-sided')
        return p_value
    return None

# Function to generate a box plot
def generate_box_plot(test_name, values1, values2, unit, p_value):
    fig = go.Figure()

    # Add box plots for both groups
    fig.add_trace(go.Box(y=values1, name="Group 1", boxpoints='all', jitter=0.3, pointpos=-1.8))
    fig.add_trace(go.Box(y=values2, name="Group 2", boxpoints='all', jitter=0.3, pointpos=-1.8))

    # Add statistical test result to the title
    p_value_text = f"p = {p_value:.4f}" if p_value is not None else "No sufficient data for statistical test"
    fig.update_layout(
        title=f"Comparison of {test_name}<br><sup>{p_value_text}</sup>",
        yaxis_title=f"{unit or 'Unit'}",
        boxmode="group"
    )

    return fig

def find_unit(line, previous_test, previous_result, current_category,selected_categories):
            if line not in [previous_test, previous_result] and "_" not in line and "|" not in line and line not in selected_categories:
                if current_category in ["Moč chemicky", "Močový sediment"] and len(line) > 7:
                    return None
                if current_category == "Močové parametry" and len(line) > 10:
                    return None
                return line
            return None

def find_ref_range(lines, current_index, previous_test_index, previous_test, previous_result,selected_categories):
    for i in range(current_index - 1, previous_test_index, -1):
        line = lines[i].strip()
        if ("-" in line and any(char.isdigit() for char in line) and
                "_" not in line and "|" not in line and line not in [previous_test, previous_result,selected_categories] and not any(char.isalpha() for char in line)):
            return line
    return None

def find_hodnoceni(lines, current_index):
    for i in range(current_index - 1, max(current_index - 4, -1), -1):
        line = lines[i].strip()
        if all(char in "| *" for char in line):
            return line
    return None

def unit_conversion(test, value, from_unit, to_unit, molecular_weight=None, unique_id=None):
    """Perform unit conversion based on predefined rules."""
    conversion_map = {
        ("µmol/l", "mg/dl"): lambda x, mw: x * mw / 10000,
        ("mg/dl", "µmol/l"): lambda x, mw: x * 10000 / mw,
        ("mg/dl", "mmol/l"): lambda x, mw: x * 10 / mw,
        ("10³/μl", "10^9/l"): lambda x: x * 1,
        ("10⁶/μl", "10^12/l"): lambda x: x * 1,
        ("fl(μm³)", "fl"): lambda x: x * 1,
        ("g/dl", "mg/dl"): lambda x: x * 100,
        ("g/dl", "g/l"): lambda x: x * 10,
        ("%", "l/l"): lambda x: x / 100,
        ("u/l", "µkat/l"): lambda x: x / 60,
        ("mg/dl", "g/dl"): lambda x: x / 100,
    }

    if (from_unit, to_unit) in conversion_map:
        conversion_func = conversion_map[(from_unit, to_unit)]
        if "mw" in conversion_func.__code__.co_varnames and molecular_weight is None:
            st.warning(f"Conversion from {from_unit} to {to_unit} requires molecular weight!")
            # Add a unique_id argument to the key
            molecular_weight = st.number_input(
                f"Enter molecular weight for {test} (required for unit conversion):",
                min_value=0.0,
                step=0.1,
                format="%.2f",
                key=f"{test}_{from_unit}_{to_unit}_mw_{unique_id}",
            )
            if not molecular_weight:
                return value  # Return original value if molecular weight is not provided
        return conversion_func(value, molecular_weight) if molecular_weight else conversion_func(value)
    else:
        st.warning(f"No conversion defined for {from_unit} to {to_unit}. Returning original value.")
        return value




# Tabbed interface for different functionalities
tab1, tab2 = st.tabs(["Extract Data from PDFs", "Generate Box Plots"])

# Tab 1: Extract Data from PDFs
with tab1:

    default_categories = [
        "Krevní obraz",
        "Základní biochemie",
        "Diabetologie",
        "Lipidový metabolismus",
        "Metabolismus železa",
        "Stav séra",
        "Moč chemicky",
        "Močový sediment",
        "Močové parametry"
    ]
    
    if "categories" not in st.session_state:
            st.session_state.categories = default_categories

    st.write("Select categories to include in the analysis:")
    selected_categories = st.multiselect(
        "Categories:",
        options=st.session_state.categories,
        default=st.session_state.categories
    )

    # Input to add a new category
    new_category = st.text_input("Add a new category:")
    if st.button("Add Category"):
        if new_category and new_category not in st.session_state.categories:
            st.session_state.categories.append(new_category)
            st.success(f"Category '{new_category}' added!")
        elif new_category in st.session_state.categories:
            st.warning(f"Category '{new_category}' already exists!")
        else:
            st.error("Please enter a valid category name.")
 
    st.write("")
    st.divider()
    st.write("Please try uploading the file using the 'Browse files' button.")
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:

            with fitz.open(stream=uploaded_file.read(), filetype="pdf") as pdf:
                full_text = ""
                for page in pdf:
                    full_text += page.get_text()

            lines = full_text.splitlines()

            data = []
            current_category = None

            previous_test_index = -1
            previous_test = ""
            previous_result = ""

            for i, line in enumerate(lines):
                line = line.strip()

                # Process only the selected categories
                if line in selected_categories:
                    current_category = line
                    continue

                if current_category:
                    if "_" in line:
                        if len(line) > 40:
                            continue
                        test_name = line
                        result_line = lines[i + 1].strip() if i + 1 < len(lines) else "None"

                        unit_line = find_unit(
                            lines[i - 1].strip() if i - 1 >= 0 else None,
                            previous_test,
                            previous_result,
                            current_category,
                            selected_categories
                        )
                        ref_range = find_ref_range(
                            lines, i, previous_test_index, previous_test, previous_result,selected_categories
                        )
                        hodnoceni = find_hodnoceni(lines, i)

                        previous_test_index = i
                        previous_test = test_name
                        previous_result = result_line

                        data.append({
                            "Category": current_category,
                            "Test": test_name,
                            "Výsledek": result_line,
                            "Jednotka": unit_line,
                            "Ref. meze": str(ref_range) if ref_range else None,
                            "Hodnocení": hodnoceni
                        })

            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Store the extracted data in session state
            if "extracted_data" not in st.session_state:
                st.session_state.extracted_data = []
            st.session_state.extracted_data.append(df)
      #      if df.empty:
       #         st.error("No data was extracted from the PDF. Please check the file content.")
        #    else:
         #       st.write("Extracted data preview:", df.head())
            
            # Create a downloadable CSV
            csv_buffer = io.BytesIO()
            df.to_csv(csv_buffer, index=False, encoding="utf-8")
            csv_buffer.seek(0)

            #st.write(f"CSV buffer size: {len(csv_buffer.getvalue())} bytes")
            
            # Streamlit download button
            file_name = f"{uploaded_file.name.split('.')[0]}_extracted_data.csv"
            
            st.download_button(
                label=f"Download Extracted Data from {uploaded_file.name}",
                data=csv_buffer,
                file_name=f"{file_name}",
                mime="text/csv"
            )
            st.write(f"Extracted Data from {file_name}:")
            st.dataframe(df)

# Tab 2: Generate Box Plots
with tab2:

    # Input fields for age and sex
    st.write("Enter demographic information for reference data retrieval:")
    age = st.text_input("Enter Age:", "")
    sex_mapping = {
    "Male": "_male",
    "Female": "_female"
    }
    selected_display_sex = st.selectbox("Select Sex:", options=list(sex_mapping.keys()))
    # Get the underlying value based on the selected display value
    sex = sex_mapping[selected_display_sex]

    # Load reference files
    ref_dir = "/workspaces/Blood_Test_App/CSV_KEY/combined_unique_values.csv"
    age_sex_dir = "/workspaces/Blood_Test_App/CSV_REF"

    # Load the reference key table
    ref_test_key_table = pd.read_csv(ref_dir)

    # Dynamically load age and sex-specific reference files
    referens_vel_files = glob.glob(os.path.join(age_sex_dir, f"*{age}*{sex}*.csv"))
    if referens_vel_files:
        df_ref_vel = pd.concat([pd.read_csv(f) for f in referens_vel_files], ignore_index=True)
    else:
        st.warning("No reference data available for the specified age and sex.")
        df_ref_vel = pd.DataFrame()

 
    # Load extracted data from Tab 1
    extracted_data = (
        pd.concat(st.session_state.extracted_data, ignore_index=True)
        if "extracted_data" in st.session_state and st.session_state.extracted_data
        else pd.DataFrame()
    )

    st.write("Upload CSV files for both groups to generate box plots for comparison.")
    col1, col2 = st.columns(2)

    with col1:
        group1_files = st.file_uploader("Upload Group 1 CSV files", type="csv", accept_multiple_files=True, key="group1")
    with col2:
        group2_files = st.file_uploader("Upload Group 2 CSV files", type="csv", accept_multiple_files=True, key="group2")

    # Combine manual and extracted data
    manual_group1 = load_data(group1_files) if group1_files else pd.DataFrame()
    manual_group2 = load_data(group2_files) if group2_files else pd.DataFrame()

    combined_group1 = pd.concat([manual_group1, extracted_data], ignore_index=True) if not manual_group1.empty else extracted_data
    combined_group2 = pd.concat([manual_group2, extracted_data], ignore_index=True) if not manual_group2.empty else extracted_data

    # Ensure the "Test" column exists
    if "Test" not in combined_group1.columns or "Test" not in combined_group2.columns:
        st.error("No 'Test' column found in the uploaded or extracted data. Ensure the correct format.")
    else:
        # Allow user to select tests
        st.write("Assign tests to groups:")
        # Select a single test for both groups
            
        available_tests = list(set(combined_group1["Test"].unique()).intersection(set(combined_group2["Test"].unique())))
        if not available_tests:
            st.error("No common tests found between Group 1 and Group 2.")
        else:
            selected_test = st.selectbox("Select a Test for Comparison:", options=available_tests)

            # Filter data based on the selected test
            filtered_group1 = combined_group1[combined_group1["Test"] == selected_test]
            filtered_group2 = combined_group2[combined_group2["Test"] == selected_test]

            st.write("Filtered Data for Group 1:")
            st.dataframe(filtered_group1)

            st.write("Filtered Data for Group 2:")
            st.dataframe(filtered_group2)

            # Generate box plots and reference ranges
            if not filtered_group1.empty and not filtered_group2.empty:
                group1_values, group2_values = prepare_test_data(filtered_group1, filtered_group2, selected_test)
                group1_unit = filtered_group1["Jednotka"].iloc[0]
                group2_unit = filtered_group2["Jednotka"].iloc[0]

                # Perform statistical test
                p_value = perform_statistical_test(group1_values, group2_values)

                # Generate box plot
                fig = generate_box_plot(selected_test, group1_values, group2_values, group1_unit, p_value)
        
                # Add reference range to the plot as a shaded region or horizontal lines
                # Add reference range to the plot as a shaded region or horizontal lines
                if not df_ref_vel.empty and selected_test in ref_test_key_table["Unique_Values"].values:
                    key_name = ref_test_key_table[ref_test_key_table["Unique_Values"] == selected_test]["Unique_Values_Ref"].iloc[0]
                    ref_row = df_ref_vel[df_ref_vel["TEST"] == key_name]
                    if not ref_row.empty:
                        ref_range = ref_row["RANGE"].iloc[0]
                        ref_min, ref_max = map(float, ref_range.replace(" ", "").split("-"))
                        ref_unit = ref_row["UNIT"].iloc[0]

                        molecular_weight = None
                        if ref_unit != group1_unit:
                            st.warning(f"Conversion from {ref_unit} to {group1_unit} requires molecular weight!")
                            molecular_weight = st.number_input(
                                f"Enter molecular weight for {selected_test} (required for unit conversion):",
                                min_value=0.0,
                                step=0.1,
                                format="%.2f",
                                key=f"{selected_test}_molecular_weight",
                            )
                            
                        # Perform the conversion only if molecular_weight is provided
                        if molecular_weight and ref_unit != group1_unit:
                            try:
                                ref_min = unit_conversion(selected_test, ref_min, ref_unit.lower(), group1_unit.lower(), molecular_weight=molecular_weight)
                                ref_max = unit_conversion(selected_test, ref_max, ref_unit.lower(), group1_unit.lower(), molecular_weight=molecular_weight)
                            except Exception as e:
                                st.error(f"Error in conversion: {e}")
                                ref_min, ref_max = None, None

                        # Only add shapes if ref_min and ref_max are valid
                        if ref_min is not None and ref_max is not None:
                            # Add a shaded region for the reference range
                            fig.add_shape(
                                type="rect",
                                x0=-0.5,  # Extend the reference range across the x-axis
                                x1=1.5,   # Assuming there are two groups (Group 1 and Group 2)
                                y0=ref_min,
                                y1=ref_max,
                                fillcolor="rgba(0, 128, 0, 0.2)",  # Semi-transparent green
                                line=dict(width=0),
                            )

                            # Optionally, add lines for the reference range
                            fig.add_shape(
                                type="line",
                                x0=-0.5,
                                x1=1.5,
                                y0=ref_min,
                                y1=ref_min,
                                line=dict(color="green", dash="dash"),
                            )
                            fig.add_shape(
                                type="line",
                                x0=-0.5,
                                x1=1.5,
                                y0=ref_max,
                                y1=ref_max,
                                line=dict(color="green", dash="dash"),
                            )
                        else:
                            st.warning("Reference range could not be plotted due to missing or invalid data.")

                # Display the updated plot
                st.plotly_chart(fig, use_container_width=True)




