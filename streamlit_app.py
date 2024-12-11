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
    group1_values = data_group1[data_group1['Test'] == test_name]['Výsledek']
    group2_values = data_group2[data_group2['Test'] == test_name]['Výsledek']

    # Replace commas with dots and convert to numeric
    group1_values = pd.to_numeric(group1_values.str.replace(",", ".", regex=False), errors='coerce').dropna()
    group2_values = pd.to_numeric(group2_values.str.replace(",", ".", regex=False), errors='coerce').dropna()

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

# Tabbed interface for different functionalities
tab1, tab2 = st.tabs(["Extract Data from PDFs", "Generate Box Plots"])

# Tab 1: Extract Data from PDFs
with tab1:
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        categories = [
            "Krevní obraz",
            "Základní biochemie",
            "Diabetologie",
            "Lipidový metabolismus",
            "Metabolismus železa",
            "Moč chemicky",
            "Močový sediment",
            "Močové parametry"
        ]

        for uploaded_file in uploaded_files:
            with fitz.open(stream=uploaded_file.read(), filetype="pdf") as pdf:
                full_text = ""
                for page in pdf:
                    full_text += page.get_text()

            lines = full_text.splitlines()

            data = []
            current_category = None

            for line in lines:
                line = line.strip()
                if line in categories:
                    current_category = line
                    continue

                if current_category and "_" in line:
                    test_name = line
                    result_line = lines[lines.index(line) + 1].strip() if lines.index(line) + 1 < len(lines) else "None"

                    data.append({
                        "Category": current_category,
                        "Test": test_name,
                        "Výsledek": result_line
                    })

            df = pd.DataFrame(data)
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
            csv_buffer.seek(0)

            st.download_button(
                label=f"Download Extracted Data from {uploaded_file.name}",
                data=csv_buffer.getvalue(),
                file_name=f"{uploaded_file.name.split('.')[0]}.csv",
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
