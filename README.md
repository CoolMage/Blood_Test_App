
# PDF Data Extractor and Blood Plot Comparison

## Overview

This project is a **Streamlit-based application** designed to:
1. Extract data from uploaded PDF reports as CSV.
2. If you want, you can immediately visualize the data in the adjacent tab as box-plots with p-value and interval of normal values for some tests.

---

## Reference Data

Data on reference values for test parameters are consistent with Crl:WI(Han) rat and were taken from the following source:
[Wistar Han Clinical Lab Parameters](https://www.criver.com/sites/default/files/resources/rm_rm_r_Wistar_Han_clin_lab_parameters_08.pdf)

- At the moment the list of reference values is limited, but there is a possibility to enter them manually in the application. The database will be updated.

---

## How to use it?

Just follow the link to the Streamlit app: `https://blank-app-41kx5t6lic7.streamlit.app/`

### Tabs

#### Tab 1: Extract Data from PDFs
- Select categories for analysis.
  - This is an important point. If you do not specify all categories as they are written in your file - the program may not identify some tests.
  - More no less. It's okay if you specify extra categories.
- Upload one or multiple PDF files in SYNLAB format.
  - To avoid problems with downloading the table, it is recommended to use the download button instead of dragging the file into the window.
- Preview extracted data and download it as a CSV.
  - The program has been tested on half a hundred files, but sometimes commits local errors. Most often they are caused by the fact that one of the categories is not specified or the format differs from the classical one. It is recommended to check the received tables.

#### Tab 2: Generate Box Plots
- Choose the age and sex of the rats.
- Upload CSV files for Group 1 and Group 2.
- Select a specific test.
- View the box plot comparison.

---

## Folder Structure

```
project_root/
│
├── streamlit_app.py                    # Main application script
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation (this file)
├── /workspaces               # Workspace for CSV data
│   ├── CSV_KEY               # Key files for reference mapping
│   └── CSV_REF               # Reference data by age and sex of subjects
└── ...
```
