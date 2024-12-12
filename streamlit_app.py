import streamlit as st
import fitz
import pandas as pd
import io
import plotly.graph_objects as go
from scipy import stats


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

# Tab 2: Generate Box Plots
with tab2:
    st.write("Upload CSV files for both groups to generate box plots for comparison.")
    
    group1_files = st.file_uploader("Upload Group 1 CSV files", type="csv", accept_multiple_files=True, key="group1")
    group2_files = st.file_uploader("Upload Group 2 CSV files", type="csv", accept_multiple_files=True, key="group2")

    if group1_files and group2_files:
        # Load data for both groups
        data_group1 = load_data(group1_files)
        data_group2 = load_data(group2_files)

        # Get list of unique tests
        all_tests = set(data_group1['Test']).union(set(data_group2['Test']))
        selected_test = st.selectbox("Select Test to Compare", options=list(all_tests))

        if selected_test:
            group1_values, group2_values = prepare_test_data(data_group1, data_group2, selected_test)

            # Get units
            group1_unit = data_group1[data_group1['Test'] == selected_test]['Jednotka'].unique()
            group2_unit = data_group2[data_group2['Test'] == selected_test]['Jednotka'].unique()
            unit = group1_unit[0] if len(group1_unit) > 0 else (group2_unit[0] if len(group2_unit) > 0 else None)

            # Perform statistical test
            p_value = perform_statistical_test(group1_values, group2_values)

            # Generate and display the box plot
            fig = generate_box_plot(selected_test, group1_values, group2_values, unit, p_value)
            st.plotly_chart(fig, use_container_width=True)
