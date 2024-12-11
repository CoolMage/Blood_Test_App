import streamlit as st
import fitz
import pandas as pd
import io

# Streamlit app title
st.title("PDF Data Extractor")

# Upload PDFs
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

    # Process each uploaded file
    for uploaded_file in uploaded_files:
        # Read PDF content
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as pdf:
            full_text = ""
            for page in pdf:
                full_text += page.get_text()

        # Split text by lines
        lines = full_text.splitlines()

        data = []
        current_category = None

        def find_unit(line, previous_test, previous_result, current_category):
            if line not in [previous_test, previous_result] and "_" not in line and "|" not in line:
                if current_category in ["Moč chemicky", "Močový sediment"] and len(line) > 7:
                    return None
                if current_category == "Močové parametry" and len(line) > 10:
                    return None
                return line
            return None

        def find_ref_range(lines, current_index, previous_test_index, previous_test, previous_result):
            for i in range(current_index - 1, previous_test_index, -1):
                line = lines[i].strip()
                if ("-" in line and any(char.isdigit() for char in line) and
                        "_" not in line and "|" not in line and line not in [previous_test, previous_result] and not any(char.isalpha() for char in line)):
                    return line
            return None

        def find_hodnoceni(lines, current_index):
            for i in range(current_index - 1, max(current_index - 4, -1), -1):
                line = lines[i].strip()
                if all(char in "| *" for char in line):
                    return line
            return None

        previous_test_index = -1
        previous_test = ""
        previous_result = ""

        for i, line in enumerate(lines):
            line = line.strip()

            if line in categories:
                current_category = line
                continue

            if current_category:
                if "_" in line:
                    if len(line) > 40:
                        continue
                    test_name = line
                    result_line = lines[i + 1].strip() if i + 1 < len(lines) else "None"

                    unit_line = find_unit(lines[i - 1].strip() if i - 1 >= 0 else None, previous_test, previous_result, current_category)
                    ref_range = find_ref_range(lines, i, previous_test_index, previous_test, previous_result)
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

        # Create a downloadable CSV
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
        csv_buffer.seek(0)

        # Streamlit download button
        st.download_button(
            label=f"Download Extracted Data from {uploaded_file.name}",
            data=csv_buffer.getvalue(),
            file_name=f"{uploaded_file.name.split('.')[0]}.csv",
            mime="text/csv"
        )
