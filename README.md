
# PDF Data Extractor and Blood Plot Comparison

## Overview

This project is a **Streamlit-based application** designed to:
1. Extract data from uploaded PDF reports.
2. Perform statistical analyses and generate insightful visualizations, such as box plots, to compare blood test results across groups.
3. Provide tools for unit conversions and incorporate reference ranges dynamically.

---

## Features

### 1. **Data Extraction from PDFs**
- Select specific categories to analyze (e.g., `Krevní obraz`, `Základní biochemie`).
- Add custom categories dynamically.
- Upload PDFs in SYNLAB format, extract tabular data, and preview results in an interactive table.
- Download extracted data as CSV files.

### 2. **Generate Comparative Box Plots**
- Compare blood test results for two groups.
- Perform statistical testing using the **Mann-Whitney U test**.
- Overlay reference ranges with shaded regions or dashed lines.
- Handle some unit conversions.

### 3. **Laboratory Reference Data Integration**
- Retrieve reference data dynamically for analysis of lab test results.
- Use sex and age of test subjects (e.g., rats) to select appropriate reference ranges.

### 4. **Unit Conversion**
- Convert between units (e.g., `mg/dl` to `µmol/l`) with optional molecular weight input for certain conversions.

---

## Reference Data

Reference data is sourced from the following document and may be supplemented in the future:
[Wistar Han Clinical Lab Parameters](https://www.criver.com/sites/default/files/resources/rm_rm_r_Wistar_Han_clin_lab_parameters_08.pdf)

---

## Installation

### Prerequisites

- **Python 3.8+**
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### Dependencies

The application requires the following key libraries:
- **Streamlit**: For the web-based interface.
- **Pandas**: For data manipulation.
- **Plotly**: For generating interactive visualizations.
- **PyMuPDF (fitz)**: For extracting text from PDFs.
- **Scipy**: For statistical analysis.

---

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/pdf-data-extractor.git
   cd pdf-data-extractor
   ```

2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

3. Navigate to the app in your browser at `http://localhost:8501`.

### Tabs

#### Tab 1: Extract Data from PDFs
- Upload one or multiple PDF files in SYNLAB format.
- Select categories for analysis.
- Ensure all relevant categories from your files are added for correct analysis.
- Preview extracted data and download it as a CSV.

#### Tab 2: Generate Box Plots
- Upload CSV files for Group 1 and Group 2.
- Select a specific test (e.g., `Glucose`).
- View the box plot comparison, including:
  - Statistical test results.
  - Reference ranges overlaid on the plot.

---

## Folder Structure

```
project_root/
│
├── app.py                    # Main application script
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation (this file)
├── /workspaces               # Workspace for CSV data
│   ├── CSV_KEY               # Key files for reference mapping
│   └── CSV_REF               # Reference data by age and sex of subjects
└── ...
```

---

## Key Functions

### PDF Data Extraction
- **`find_unit()`**: Extract units from text lines.
- **`find_ref_range()`**: Identify reference ranges for a test.
- **`find_hodnoceni()`**: Extract evaluation comments from the report.

### Data Analysis
- **`prepare_test_data()`**: Clean and prepare test results for analysis.
- **`perform_statistical_test()`**: Perform the Mann-Whitney U test.

### Visualization
- **`generate_box_plot()`**: Create box plots with Plotly, overlaying statistical results and reference ranges.

### Unit Conversion
- **`unit_conversion()`**: Convert values between units using predefined conversion maps.

---

## Examples

### 1. Extract Data
- Upload a PDF file containing blood test results in SYNLAB format.
- Select categories like `Základní biochemie`.
- Preview extracted data and download it as a CSV.

### 2. Generate Box Plots
- Upload CSV files for Group 1 and Group 2.
- Select a specific test (e.g., `Glucose`).
- View the box plot comparison, including:
  - Statistical test results.
  - Reference ranges overlaid on the plot.

---

## Future Enhancements

- Add support for multi-language PDF parsing.
- Enhance statistical options (e.g., parametric tests, multiple testing correction).
- Expand the range of available graphs.
- Work on code adequacy...


