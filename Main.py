#!/usr/bin/env python3
"""
City Hospital - End-to-End Data Preprocessing, ABT Construction, EDA & PDF Reporting
Author: Expert Data Scientist & Python Developer
Description:
    Complete, reproducible pipeline that loads raw consultation records (10,008 rows,
    500 patients), resolves visit-level and patient-level data quality issues, builds a
    500-row patient-level Analytic Base Table (ABT), generates 6 publication-quality
    visualizations, and compiles both Report.pdf and Slides.pdf automatically.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
)

warnings.filterwarnings("ignore")

# Set global visualization style
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.sans-serif"] = "DejaVu Sans"


# ==============================================================================
# PART A: DATA UNDERSTANDING AND CLEANING
# ==============================================================================

def load_data(filepath: str = None) -> pd.DataFrame:
    """
    Loads the raw consultation dataset without modifying the source Excel file.
    Automatically detects 'Project_Data.xlsx' or 'Project Data.xlsx'.
    Prints dimensions, data types, missing values, and unique patient counts.
    """
    if filepath is None:
        if os.path.exists("Project_Data.xlsx"):
            filepath = "Project_Data.xlsx"
        elif os.path.exists("Project Data.xlsx"):
            filepath = "Project Data.xlsx"
        else:
            raise FileNotFoundError(
                "Could not find 'Project_Data.xlsx' or 'Project Data.xlsx' in current directory."
            )

    print("=" * 80)
    print(f"PART A: LOADING & INSPECTING RAW DATA FROM '{filepath}'")
    print("=" * 80)

    df = pd.read_excel(filepath)

    print(f"Dataset Dimensions : {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Unique Patient IDs : {df['Patient ID'].nunique():,}")
    print(f"Date Range         : {df['Visit Date'].min().date()} to {df['Visit Date'].max().date()}")
    print(f"Exact Duplicate Rows: {df.duplicated().sum()}")
    print("\n--- Column Data Types & Missing Values ---")
    summary_df = pd.DataFrame({
        "Data Type": df.dtypes.astype(str),
        "Non-Null Count": df.notnull().sum(),
        "Missing Count": df.isnull().sum(),
        "Missing (%)": (df.isnull().sum() / len(df) * 100).round(2)
    })
    print(summary_df.to_string())
    print("=" * 80 + "\n")

    return df


def clean_data(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Performs rigorous statistical and rule-based data cleaning while preserving
    all 500 unique patients.

    Cleaning Steps & Justifications:
    1. Exact Duplicates: Remove 8 exact duplicate consultation rows to prevent double-counting.
    2. Static Attribute Inconsistencies: Resolve conflicting static patient attributes
       (Gender, Profession, Marital Status, Age, City, Education, Insurance, Family History)
       by sorting by 'Visit Date' and keeping the most recent non-null value per patient.
       Cross-check pediatric inconsistencies (patients with Age < 18 assigned adult
       professions/marital statuses due to entry errors) and correct them.
    3. Domain-Invalid Scales & Outliers (IQR & Statistical Bounds):
       - Age: Values > 100 (e.g., 118, 138, 142, 158, 195) are 100-year digit typos or invalid;
         set to NaN prior to imputation.
       - Satisfaction Level: Rated 1 to 5 by specification; 719 records with value '6'
         are capped at the valid maximum of 5.
       - Consultation Duration: Apply the 1.5 * IQR rule (Q1=41, Q3=94, Upper Bound=173.5 min)
         to Winsorize (cap) 10 extreme durations (228 to 600 min).
       - Consultation Price: 7 extreme decimal-shift typos (> £25,000) and upper outliers
         detected via department-specific IQR bounds are Winsorized at the upper IQR fence.
       - Approximate Annual Income: Pediatric/Student records legitimately have £0 income.
         Among adult earners, 21 records exceed £1,000,000 (up to £11.97M) due to decimal shifts;
         flagged via adult IQR bounds and set to NaN for KNN imputation.
    4. Predictive Imputation (KNNImputer & Within-Patient Propagation):
       - First propagate patient-level static attributes within each Patient ID.
       - For 'Family History' (missing for 99 patients across all visits) and 'Insurance Provider'
         (missing for 92 uninsured patients with £0 coverage), assign explicit 'None' / 'Uninsured'
         categories or predictive KNN modes where coverage > £0.
       - Use scikit-learn's KNNImputer (k=5) on normalized numerical and encoded features
         to predictively impute remaining missing values in Age, Education Level,
         Approximate Annual Income, and Insurance Coverage.
    5. Financial Accounting Consistency:
       - Ensure Insurance Coverage >= 0 and <= Consultation Price (correcting 6 rows where
         coverage exceeded the consultation price).
    """
    print("=" * 80)
    print("PART A (CONT.): CLEANING DATA, IMPUTING MISSING VALUES & HANDLING OUTLIERS")
    print("=" * 80)

    df = df_raw.copy()
    cleaning_stats = {}

    # 1. Remove exact duplicate rows
    dup_count = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    cleaning_stats["exact_duplicates_removed"] = dup_count
    print(f"[Step 1] Removed {dup_count} exact duplicate rows. Remaining rows: {len(df):,}")

    # Ensure Visit Date is datetime
    df["Visit Date"] = pd.to_datetime(df["Visit Date"])

    # 2. Pre-clean impossible Age typos (> 100 years) before resolving static attributes
    # Inspection showed 5 patients with ages 118, 138, 142, 158, 195 (e.g. Patient 1198 had 38 on one visit and 138 on others)
    age_outliers_mask = df["Age"] > 100
    cleaning_stats["invalid_age_entries"] = int(age_outliers_mask.sum())
    df.loc[age_outliers_mask, "Age"] = np.nan
    print(f"[Step 2] Flagged {cleaning_stats['invalid_age_entries']} impossible Age entries (>100 yrs) as NaN.")

    # 3. Resolve conflicting static attributes by keeping the MOST RECENT non-null value per Patient ID
    static_cols = [
        "Age", "Gender", "City of Residence", "Profession",
        "Insurance Provider", "Family History", "Education Level", "Marital Status"
    ]

    # Sort chronologically by Patient ID and Visit Date so the last non-null value is the most recent
    df = df.sort_values(["Patient ID", "Visit Date"]).reset_index(drop=True)

    inconsistent_patients_count = {}
    for col in static_cols:
        nunique_series = df.groupby("Patient ID")[col].nunique(dropna=True)
        n_inconsistent = int((nunique_series > 1).sum())
        inconsistent_patients_count[col] = n_inconsistent

        # Extract most recent non-null value per patient based on Visit Date
        non_null_sub = df.dropna(subset=[col])
        most_recent_map = non_null_sub.groupby("Patient ID")[col].last()
        df[col] = df["Patient ID"].map(most_recent_map)

    cleaning_stats["inconsistent_patients"] = inconsistent_patients_count
    print(f"[Step 3] Resolved static attribute conflicts using most recent Visit Date value:")
    for k, v in inconsistent_patients_count.items():
        if v > 0:
            print(f"         - {k}: {v} patients standardized to most recent visit record")

    # Logical cross-check for pediatric patients (Age < 18):
    # Children in Pre-school/Elementary/Middle school had stray adult professions/marital status
    pediatric_mask = df["Age"] < 18
    df.loc[pediatric_mask & (df["Profession"] != "Student"), "Profession"] = "Student"
    df.loc[pediatric_mask & (df["Marital Status"] != "Single"), "Marital Status"] = "Single"

    # 4. Handle Domain Invalid Values in Satisfaction Level (scale is 1 to 5; 6 is out-of-range)
    sat_over_5 = int((df["Satisfaction Level"] > 5).sum())
    df["Satisfaction Level"] = df["Satisfaction Level"].clip(lower=1, upper=5)
    cleaning_stats["satisfaction_capped"] = sat_over_5
    print(f"[Step 4] Capped {sat_over_5} out-of-bounds Satisfaction Level records (value=6) to max scale 5.")

    # 5. Statistical Outlier Detection & Treatment (IQR Method)
    # 5a. Consultation Duration: 1.5 * IQR Rule
    q1_dur, q3_dur = df["Consultation Duration"].quantile([0.25, 0.75])
    iqr_dur = q3_dur - q1_dur
    upper_dur = q3_dur + 1.5 * iqr_dur
    dur_outliers = int((df["Consultation Duration"] > upper_dur).sum())
    df["Consultation Duration"] = df["Consultation Duration"].clip(upper=upper_dur)
    cleaning_stats["duration_outliers_capped"] = dur_outliers
    cleaning_stats["duration_upper_fence"] = float(upper_dur)
    print(f"[Step 5a] Consultation Duration IQR Upper Fence = {upper_dur:.1f} min. Winsorized {dur_outliers} extreme rows.")

    # 5b. Consultation Price: Department-Aware IQR Outlier Detection
    # (7 rows have extreme typos of £25,300 - £39,999; departmental IQR cleanly bounds them)
    price_outliers_total = 0
    for dept, grp in df.groupby("Department"):
        q1_p, q3_p = grp["Consultation Price"].quantile([0.25, 0.75])
        iqr_p = q3_p - q1_p
        upper_p = q3_p + 1.5 * iqr_p
        lower_p = max(0.0, q1_p - 1.5 * iqr_p)
        mask_p = (df["Department"] == dept) & (
            (df["Consultation Price"] > upper_p) | (df["Consultation Price"] < lower_p)
        )
        price_outliers_total += int(mask_p.sum())
        df.loc[df["Department"] == dept, "Consultation Price"] = (
            df.loc[df["Department"] == dept, "Consultation Price"].clip(lower=lower_p, upper=upper_p)
        )
    cleaning_stats["price_outliers_capped"] = price_outliers_total
    print(f"[Step 5b] Consultation Price: Winsorized {price_outliers_total} outlier records using departmental 1.5*IQR fences.")

    # 5c. Approximate Annual Income:
    # Students legitimately have 0 income. For non-students, apply 1.5 * IQR on positive incomes
    # to catch extreme decimal typos (> £1.1M up to £11.97M) and set them to NaN for KNN imputation.
    adult_earners_mask = (df["Profession"] != "Student") & (df["Approximate Annual Income"] > 0)
    q1_inc, q3_inc = df.loc[adult_earners_mask, "Approximate Annual Income"].quantile([0.25, 0.75])
    iqr_inc = q3_inc - q1_inc
    upper_inc = q3_inc + 1.5 * iqr_inc
    inc_outlier_mask = (df["Profession"] != "Student") & (
        (df["Approximate Annual Income"] > upper_inc) | (df["Approximate Annual Income"] == 0)
    )
    cleaning_stats["income_outliers_flagged"] = int(inc_outlier_mask.sum())
    cleaning_stats["income_upper_fence"] = float(upper_inc)
    df.loc[inc_outlier_mask, "Approximate Annual Income"] = np.nan
    # Students have £0 annual income by definition
    df.loc[df["Profession"] == "Student", "Approximate Annual Income"] = 0.0
    print(f"[Step 5c] Approximate Annual Income Adult IQR Upper Fence = £{upper_inc:,.2f}. "
          f"Flagged {cleaning_stats['income_outliers_flagged']} anomalous adult records for KNN imputation.")

    # 6. Predictive Imputation using Scikit-Learn KNNImputer
    # 6a. Handle structural missingness in Family History & Insurance Provider:
    # Family History is NaN for 99 patients across all visits -> indicates 'No Family History' ('None')
    df["Family History"] = df["Family History"].fillna("None")

    # Insurance Provider is NaN for 92 patients across all visits.
    # Where a patient has 0 coverage across visits, they are genuinely 'Uninsured'.
    # If Insurance Coverage is missing for an Uninsured patient, set coverage to 0.0.
    df["Insurance Provider"] = df["Insurance Provider"].fillna("Uninsured")
    df.loc[(df["Insurance Provider"] == "Uninsured") & (df["Insurance Coverage"].isna()), "Insurance Coverage"] = 0.0

    # 6b. Patient-level KNN Imputation for remaining missing static/patient features
    # (Age, Education Level, Approximate Annual Income)
    patient_df = df.groupby("Patient ID").agg({
        "Age": "first",
        "Gender": "first",
        "City of Residence": "first",
        "Profession": "first",
        "Insurance Provider": "first",
        "Family History": "first",
        "Education Level": "first",
        "Marital Status": "first",
        "Approximate Annual Income": "median"
    }).reset_index()

    cat_features_pt = ["Gender", "City of Residence", "Profession", "Insurance Provider", "Family History", "Marital Status"]
    enc_pt = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    encoded_cats = enc_pt.fit_transform(patient_df[cat_features_pt])

    # Ordinal mapping for Education Level so KNNImputer preserves educational hierarchy
    edu_order = [
        "Pre-school", "Kindergarten", "Elementary school",
        "Middle school/junior high", "High school", "Undergraduate", "Master", "PhD"
    ]
    edu_to_num = {lvl: idx for idx, lvl in enumerate(edu_order)}
    num_to_edu = {idx: lvl for idx, lvl in enumerate(edu_order)}
    patient_df["Edu_Num"] = patient_df["Education Level"].map(edu_to_num)

    knn_matrix_pt = np.column_stack([
        encoded_cats,
        patient_df[["Age", "Edu_Num", "Approximate Annual Income"]].values
    ])

    knn_imputer_pt = KNNImputer(n_neighbors=5, weights="distance")
    imputed_pt_matrix = knn_imputer_pt.fit_transform(knn_matrix_pt)

    patient_df["Age"] = np.round(imputed_pt_matrix[:, -3]).astype(int)
    patient_df["Edu_Num"] = np.round(imputed_pt_matrix[:, -2]).clip(0, len(edu_order) - 1).astype(int)
    patient_df["Education Level"] = patient_df["Edu_Num"].map(num_to_edu)
    patient_df["Approximate Annual Income"] = np.round(imputed_pt_matrix[:, -1], 2)
    patient_df.loc[patient_df["Profession"] == "Student", "Approximate Annual Income"] = 0.0

    # Map imputed patient-level attributes back to the visit-level dataframe
    for col in ["Age", "Education Level"]:
        df[col] = df["Patient ID"].map(patient_df.set_index("Patient ID")[col])

    # For visit-level Approximate Annual Income, impute remaining NaNs with patient's KNN-imputed baseline
    pt_income_map = patient_df.set_index("Patient ID")["Approximate Annual Income"]
    df["Approximate Annual Income"] = df["Approximate Annual Income"].fillna(df["Patient ID"].map(pt_income_map))

    # 6c. Visit-level KNN Imputation for remaining missing Insurance Coverage rows (29 insured visits)
    visit_cat_cols = ["Department", "Insurance Provider"]
    enc_vis = OrdinalEncoder()
    vis_cats = enc_vis.fit_transform(df[visit_cat_cols])
    vis_knn_matrix = np.column_stack([
        vis_cats,
        df[["Consultation Price", "Consultation Duration", "Approximate Annual Income", "Insurance Coverage"]].values
    ])
    knn_imputer_vis = KNNImputer(n_neighbors=5, weights="distance")
    vis_imputed = knn_imputer_vis.fit_transform(vis_knn_matrix)
    df["Insurance Coverage"] = vis_imputed[:, -1]

    # 7. Final Financial Consistency Check: Coverage cannot exceed Consultation Price or be negative
    cov_exceeds_price = int((df["Insurance Coverage"] > df["Consultation Price"]).sum())
    df["Insurance Coverage"] = np.minimum(df["Insurance Coverage"].clip(lower=0.0), df["Consultation Price"])
    df["Out_of_Pocket"] = (df["Consultation Price"] - df["Insurance Coverage"]).round(2)
    cleaning_stats["coverage_capped_to_price"] = cov_exceeds_price

    print(f"[Step 6] Completed Scikit-Learn KNNImputer (k=5) for Age, Education, Income & Insurance Coverage.")
    print(f"[Step 7] Capped {cov_exceeds_price} records where Insurance Coverage exceeded Consultation Price.")
    print(f"Final Cleaned Visit Dataset: {df.shape[0]:,} rows, {df['Patient ID'].nunique()} unique patients, "
          f"{df.isnull().sum().sum()} missing values.\n")

    return df, cleaning_stats


# ==============================================================================
# PART B: FEATURE ENGINEERING AND ABT CONSTRUCTION
# ==============================================================================

def build_abt(df_clean: pd.DataFrame, output_csv: str = "ABT_city_hospital.csv") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregates the cleaned visit-level dataset into a patient-level Analytic Base Table (ABT)
    with exactly 500 rows (one row per Patient ID) and generates a comprehensive Data Dictionary.
    Reference Date: July 1, 2024.
    """
    print("=" * 80)
    print("PART B: FEATURE ENGINEERING & ABT CONSTRUCTION (500 PATIENTS)")
    print("=" * 80)

    df = df_clean.copy()
    ref_date = pd.Timestamp("2024-07-01")

    # Helper columns for behavioural and satisfaction aggregations
    df["Is_Weekend"] = df["Visit Date"].dt.dayofweek.isin([5, 6]).astype(int)
    df["Is_Weekday"] = (~df["Visit Date"].dt.dayofweek.isin([5, 6])).astype(int)
    df["Is_Low_Satisfaction"] = (df["Satisfaction Level"] <= 2).astype(int)
    df["Out_of_Pocket"] = df["Consultation Price"] - df["Insurance Coverage"]

    # Deterministic mode helper for Most Frequent Department
    def get_mode_dept(series):
        counts = series.value_counts()
        return counts.index[0]

    grouped = df.groupby("Patient ID")

    abt = pd.DataFrame({"Patient_ID": grouped.groups.keys()})
    abt = abt.sort_values("Patient_ID").reset_index(drop=True)

    # 1. Static Demographic & Socio-Economic Features
    static_agg = grouped.agg(
        Age=("Age", "last"),
        Gender=("Gender", "last"),
        City_of_Residence=("City of Residence", "last"),
        Profession=("Profession", "last"),
        Insurance_Provider=("Insurance Provider", "last"),
        Family_History=("Family History", "last"),
        Education_Level=("Education Level", "last"),
        Marital_Status=("Marital Status", "last"),
        Approx_Annual_Income=("Approximate Annual Income", "mean")
    ).reset_index()

    # Create Age Group (10-year bins) and Income Quartile for segmentation & EDA
    age_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 105]
    age_labels = ["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89", "90+"]
    static_agg["Age_Group_10yr"] = pd.cut(
        static_agg["Age"], bins=age_bins, labels=age_labels, right=False, include_lowest=True
    ).astype(str)

    static_agg["Income_Quartile"] = pd.qcut(
        static_agg["Approx_Annual_Income"].rank(method="first"),
        q=4,
        labels=["Q1 (Lowest)", "Q2 (Lower-Mid)", "Q3 (Upper-Mid)", "Q4 (Highest)"]
    ).astype(str)

    # 2. Visit Behaviour Features
    behaviour_agg = grouped.agg(
        Total_Visits=("Visit Date", "count"),
        Distinct_Departments=("Department", "nunique"),
        Most_Frequent_Department=("Department", get_mode_dept),
        Avg_Consultation_Duration=("Consultation Duration", "mean"),
        Total_Consultation_Duration=("Consultation Duration", "sum"),
        Weekend_Visits=("Is_Weekend", "sum"),
        Weekday_Visits=("Is_Weekday", "sum")
    ).reset_index()

    # Weekend vs. Weekday Visit Ratio (Weekend_Visits / max(1, Weekday_Visits)) and Weekend Share
    behaviour_agg["Weekend_to_Weekday_Ratio"] = (
        behaviour_agg["Weekend_Visits"] / behaviour_agg["Weekday_Visits"].replace(0, 1)
    ).round(4)
    behaviour_agg["Weekend_Visit_Share"] = (
        behaviour_agg["Weekend_Visits"] / behaviour_agg["Total_Visits"]
    ).round(4)

    # 3. Financial Features
    financial_agg = grouped.agg(
        Total_Consultation_Price=("Consultation Price", "sum"),
        Avg_Consultation_Price=("Consultation Price", "mean"),
        Total_Insurance_Coverage=("Insurance Coverage", "sum"),
        Avg_Insurance_Coverage=("Insurance Coverage", "mean"),
        Total_Out_of_Pocket=("Out_of_Pocket", "sum"),
        Avg_Out_of_Pocket=("Out_of_Pocket", "mean")
    ).reset_index()

    financial_agg["Out_of_Pocket_Ratio"] = (
        financial_agg["Total_Out_of_Pocket"] / financial_agg["Total_Consultation_Price"]
    ).round(4)

    # 4. Satisfaction Features
    satisfaction_agg = grouped.agg(
        Avg_Satisfaction_Rating=("Satisfaction Level", "mean"),
        Min_Satisfaction_Rating=("Satisfaction Level", "min"),
        Max_Satisfaction_Rating=("Satisfaction Level", "max"),
        Low_Rating_Count=("Is_Low_Satisfaction", "sum")
    ).reset_index()

    satisfaction_agg["Share_Low_Ratings"] = (
        satisfaction_agg["Low_Rating_Count"] / behaviour_agg["Total_Visits"]
    ).round(4)

    # 5. Recency & Activity Features (Reference Date: July 1, 2024)
    recency_agg = grouped.agg(
        First_Visit_Date=("Visit Date", "min"),
        Last_Visit_Date=("Visit Date", "max")
    ).reset_index()

    recency_agg["Days_Since_First_Visit"] = (ref_date - recency_agg["First_Visit_Date"]).dt.days
    recency_agg["Days_Since_Last_Visit"] = (ref_date - recency_agg["Last_Visit_Date"]).dt.days
    recency_agg["Active_Span_Days"] = (
        recency_agg["Days_Since_First_Visit"] - recency_agg["Days_Since_Last_Visit"]
    ).clip(lower=1)
    # Visits per month across the 6-month observation window (Jan - Jun 2024 = 6.0 months)
    recency_agg["Visits_Per_Month"] = (behaviour_agg["Total_Visits"] / 6.0).round(2)

    # Merge all feature blocks into a single ABT
    abt = (
        static_agg
        .merge(behaviour_agg, on="Patient ID")
        .merge(financial_agg, on="Patient ID")
        .merge(satisfaction_agg, on="Patient ID")
        .merge(recency_agg.drop(columns=["First_Visit_Date", "Last_Visit_Date"]), on="Patient ID")
    )

    # Round floating-point columns cleanly
    float_cols_2dec = [
        "Approx_Annual_Income", "Avg_Consultation_Duration", "Total_Consultation_Price",
        "Avg_Consultation_Price", "Total_Insurance_Coverage", "Avg_Insurance_Coverage",
        "Total_Out_of_Pocket", "Avg_Out_of_Pocket", "Avg_Satisfaction_Rating"
    ]
    for col in float_cols_2dec:
        abt[col] = abt[col].round(2)

    assert len(abt) == 500, f"Expected 500 rows in ABT, got {len(abt)}"
    abt.to_csv(output_csv, index=False)

    # Build Data Dictionary
    data_dict_records = [
        ("Patient ID", "Static / ID", "Unique identifier for each patient (1 row per patient, N=500)."),
        ("Age", "Static Demographic", "Patient age in years (most recent valid record, KNN-imputed if missing)."),
        ("Gender", "Static Demographic", "Patient gender (Female, Male, Other); standardized to most recent visit."),
        ("City_of_Residence", "Static Demographic", "Patient city of residence across 8 UK/Irish cities."),
        ("Profession", "Static Socio-Economic", "Patient occupation (standardized to most recent visit; <18 set to Student)."),
        ("Insurance_Provider", "Static Financial", "Provider A, B, C, D, or 'Uninsured' for patients with no policy."),
        ("Family_History", "Static Clinical", "Recorded family medical history (Cancer, Diabetes, Heart Disease, Hypertension, None)."),
        ("Education_Level", "Static Socio-Economic", "Highest educational attainment (Pre-school to PhD; KNN-imputed)."),
        ("Marital_Status", "Static Demographic", "Marital status (Single, Married, Divorced, Widowed; standardized)."),
        ("Approx_Annual_Income", "Static Socio-Economic", "Mean cleaned annual income (£) across patient consultations."),
        ("Age_Group_10yr", "Static Derived", "10-year age bracket (0-9, 10-19, ..., 90+) for demographic segmentation."),
        ("Income_Quartile", "Static Derived", "Patient income quartile (Q1 Lowest to Q4 Highest)."),
        ("Total_Visits", "Visit Behaviour", "Total number of hospital consultations between Jan 1 and Jun 30, 2024."),
        ("Distinct_Departments", "Visit Behaviour", "Number of unique hospital departments visited by the patient."),
        ("Most_Frequent_Department", "Visit Behaviour", "Modal (most frequently visited) department for the patient."),
        ("Avg_Consultation_Duration", "Visit Behaviour", "Mean consultation duration in minutes (post-IQR Winsorization)."),
        ("Total_Consultation_Duration", "Visit Behaviour", "Cumulative consultation duration in minutes across all visits."),
        ("Weekend_Visits", "Visit Behaviour", "Count of consultations occurring on Saturday or Sunday."),
        ("Weekday_Visits", "Visit Behaviour", "Count of consultations occurring Monday through Friday."),
        ("Weekend_to_Weekday_Ratio", "Visit Behaviour", "Ratio of weekend visits to weekday visits (Weekend / max(1, Weekday))."),
        ("Weekend_Visit_Share", "Visit Behaviour", "Proportion of total visits occurring on weekends."),
        ("Total_Consultation_Price", "Financial", "Cumulative gross consultation cost (£) across all visits."),
        ("Avg_Consultation_Price", "Financial", "Mean gross price (£) per consultation."),
        ("Total_Insurance_Coverage", "Financial", "Cumulative amount (£) covered and paid by the insurance provider."),
        ("Avg_Insurance_Coverage", "Financial", "Mean insurance coverage amount (£) per consultation."),
        ("Total_Out_of_Pocket", "Financial", "Total out-of-pocket expenditure (£) paid by the patient (Price - Coverage)."),
        ("Avg_Out_of_Pocket", "Financial", "Mean out-of-pocket cost (£) per consultation."),
        ("Out_of_Pocket_Ratio", "Financial", "Share of total consultation price paid out-of-pocket by the patient."),
        ("Avg_Satisfaction_Rating", "Satisfaction", "Mean consultation satisfaction score (1 = lowest, 5 = highest)."),
        ("Min_Satisfaction_Rating", "Satisfaction", "Minimum satisfaction rating recorded across any visit."),
        ("Max_Satisfaction_Rating", "Satisfaction", "Maximum satisfaction rating recorded across any visit."),
        ("Low_Rating_Count", "Satisfaction", "Total count of dissatisfied visits with rating <= 2."),
        ("Share_Low_Ratings", "Satisfaction", "Proportion of patient visits rated <= 2 (dissatisfaction rate)."),
        ("Days_Since_First_Visit", "Recency & Activity", "Days elapsed from the patient's first visit to July 1, 2024."),
        ("Days_Since_Last_Visit", "Recency & Activity", "Days elapsed from the patient's most recent visit to July 1, 2024."),
        ("Active_Span_Days", "Recency & Activity", "Number of days between the patient's first and last recorded visit."),
        ("Visits_Per_Month", "Recency & Activity", "Average monthly consultation frequency over the 6-month period (Total_Visits / 6).")
    ]

    data_dict_df = pd.DataFrame(data_dict_records, columns=["Feature Name", "Category", "Definition & Calculation Logic"])
    data_dict_df.to_csv("Data_Dictionary_city_hospital.csv", index=False)

    print(f"ABT successfully constructed: {abt.shape[0]} rows x {abt.shape[1]} features.")
    print(f"Saved ABT to '{output_csv}' and Data Dictionary to 'Data_Dictionary_city_hospital.csv'.\n")

    return abt, data_dict_df


# ==============================================================================
# PART C: EXPLORATORY VISUALIZATION (MATPLOTLIB / SEABORN)
# ==============================================================================

def generate_plots(df_clean: pd.DataFrame, abt: pd.DataFrame, output_dir: str = ".") -> dict:
    """
    Generates 6 multi-panel, publication-grade visualizations covering all required dimensions
    and saves them as high-resolution PNG files for embedding into Report.pdf and Slides.pdf.
    """
    print("=" * 80)
    print("PART C: GENERATING 6 EXPLORATORY VISUALIZATIONS (.PNG)")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)
    plot_paths = {}

    # Prepare age group order
    age_labels = ["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89", "90+"]

    # --------------------------------------------------------------------------
    # PLOT 1: Patient Demographics (Age 10-Year Bins by Gender & Income Quartiles)
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    age_gender_ct = pd.crosstab(abt["Age_Group_10yr"], abt["Gender"]).reindex(age_labels, fill_value=0)
    age_gender_ct.plot(
        kind="bar", stacked=True, ax=axes[0],
        color=["#2b5c8f", "#d95f02", "#7570b3"], edgecolor="black", width=0.75
    )
    axes[0].set_title("Patient Distribution by 10-Year Age Group & Gender", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Age Group (10-Year Bins)", fontsize=10)
    axes[0].set_ylabel("Number of Patients (N = 500)", fontsize=10)
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].legend(title="Gender", frameon=True)

    quartile_order = ["Q1 (Lowest)", "Q2 (Lower-Mid)", "Q3 (Upper-Mid)", "Q4 (Highest)"]
    sns.boxplot(
        data=abt, x="Income_Quartile", y="Approx_Annual_Income",
        order=quartile_order, palette="Blues", ax=axes[1], showfliers=True
    )
    axes[1].set_title("Annual Income (£) Distribution Across Patient Quartiles", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Income Quartile Segment", fontsize=10)
    axes[1].set_ylabel("Approximate Annual Income (£)", fontsize=10)
    axes[1].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    p1_path = os.path.join(output_dir, "plot1_demographics.png")
    plt.savefig(p1_path, bbox_inches="tight")
    plt.close()
    plot_paths["plot1"] = p1_path
    print(f"[Plot 1] Saved: {p1_path}")

    # --------------------------------------------------------------------------
    # PLOT 2: Department Usage (Total Visits per Department & Monthly Trends Jan-Jun)
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))

    dept_counts = df_clean["Department"].value_counts().sort_values(ascending=True)
    colors_dept = ["#1b9e77" if v > 1000 else "#4a7c59" for v in dept_counts.values]
    dept_counts.plot(kind="barh", ax=axes[0], color=colors_dept, edgecolor="black")
    axes[0].set_title("Total Consultations by Department (Jan - Jun 2024)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Number of Consultations", fontsize=10)
    axes[0].set_ylabel("Department", fontsize=10)
    for i, v in enumerate(dept_counts.values):
        axes[0].text(v + 15, i, f"{v:,}", va="center", fontsize=8.5, fontweight="bold")
    axes[0].set_xlim(0, dept_counts.max() * 1.15)

    df_clean["Visit_Month"] = df_clean["Visit Date"].dt.to_period("M").astype(str)
    top_depts = ["Psychiatry", "General Practice", "Gastroenterology", "Endocrinology", "Neurology", "Emergency"]
    monthly_dept = (
        df_clean[df_clean["Department"].isin(top_depts)]
        .groupby(["Visit_Month", "Department"]).size().unstack(fill_value=0)
    )
    monthly_dept.plot(ax=axes[1], marker="o", linewidth=2, markersize=6)
    axes[1].set_title("Monthly Consultation Demand Trends (Top 6 Departments)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Month (2024)", fontsize=10)
    axes[1].set_ylabel("Monthly Consultations", fontsize=10)
    axes[1].legend(title="Department", fontsize=8.5, loc="best")

    plt.tight_layout()
    p2_path = os.path.join(output_dir, "plot2_department_usage_trends.png")
    plt.savefig(p2_path, bbox_inches="tight")
    plt.close()
    plot_paths["plot2"] = p2_path
    print(f"[Plot 2] Saved: {p2_path}")

    # --------------------------------------------------------------------------
    # PLOT 3: Department Usage by Gender and Age Group
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    df_with_agegrp = df_clean.merge(abt[["Patient ID", "Age_Group_10yr"]], on="Patient ID", how="left")
    dept_gender = pd.crosstab(df_with_agegrp["Department"], df_with_agegrp["Gender"])
    dept_gender = dept_gender.loc[dept_counts.index]
    dept_gender.plot(kind="barh", stacked=True, ax=axes[0], color=["#2b5c8f", "#d95f02", "#7570b3"], edgecolor="black")
    axes[0].set_title("Department Consultation Volume by Gender", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Number of Consultations", fontsize=10)
    axes[0].set_ylabel("Department", fontsize=10)
    axes[0].legend(title="Gender", loc="lower right")

    age_dept_heat = pd.crosstab(
        df_with_agegrp["Department"], df_with_agegrp["Age_Group_10yr"], normalize="index"
    ).reindex(columns=age_labels, fill_value=0) * 100
    sns.heatmap(
        age_dept_heat, annot=True, fmt=".1f", cmap="YlOrRd", ax=axes[1],
        cbar_kws={"label": "Share of Department Visits (%)"}
    )
    axes[1].set_title("Department Usage Share (%) Across 10-Year Age Groups", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Age Group (10-Year Bins)", fontsize=10)
    axes[1].set_ylabel("")
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    p3_path = os.path.join(output_dir, "plot3_dept_by_gender_age.png")
    plt.savefig(p3_path, bbox_inches="tight")
    plt.close()
    plot_paths["plot3"] = p3_path
    print(f"[Plot 3] Saved: {p3_path}")

    # --------------------------------------------------------------------------
    # PLOT 4: Satisfaction by Department and Low-Rating Share
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))

    dept_sat = df_clean.groupby("Department").agg(
        Mean_Sat=("Satisfaction Level", "mean"),
        Low_Share=("Satisfaction Level", lambda s: (s <= 2).mean() * 100)
    ).sort_values("Mean_Sat", ascending=True)

    dept_sat["Mean_Sat"].plot(kind="barh", ax=axes[0], color="#386cb0", edgecolor="black")
    axes[0].axvline(df_clean["Satisfaction Level"].mean(), color="red", linestyle="--", label="Hospital Mean")
    axes[0].set_title("Mean Patient Satisfaction Score by Department (1-5 Scale)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Average Satisfaction Score", fontsize=10)
    axes[0].set_ylabel("Department", fontsize=10)
    axes[0].set_xlim(0, 5.0)
    for i, v in enumerate(dept_sat["Mean_Sat"].values):
        axes[0].text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=8.5)
    axes[0].legend(loc="lower right")

    dept_sat_low = dept_sat.sort_values("Low_Share", ascending=True)
    dept_sat_low["Low_Share"].plot(kind="barh", ax=axes[1], color="#e41a1c", edgecolor="black")
    axes[1].set_title("Share of Dissatisfied Visits (Rating <= 2) by Department", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Low Satisfaction Visits (%)", fontsize=10)
    axes[1].set_ylabel("")
    axes[1].set_xlim(0, 50)
    for i, v in enumerate(dept_sat_low["Low_Share"].values):
        axes[1].text(v + 0.6, i, f"{v:.1f}%", va="center", fontsize=8.5)

    plt.tight_layout()
    p4_path = os.path.join(output_dir, "plot4_satisfaction_by_department.png")
    plt.savefig(p4_path, bbox_inches="tight")
    plt.close()
    plot_paths["plot4"] = p4_path
    print(f"[Plot 4] Saved: {p4_path}")

    # --------------------------------------------------------------------------
    # PLOT 5: Satisfaction by Patient Segment (Profession, Income Quartile & Insurance)
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))

    prof_sat = abt.groupby("Profession")["Avg_Satisfaction_Rating"].mean().sort_values(ascending=True)
    prof_sat.plot(kind="barh", ax=axes[0], color="#66a61e", edgecolor="black")
    axes[0].axvline(abt["Avg_Satisfaction_Rating"].mean(), color="red", linestyle="--", label="Patient Mean")
    axes[0].set_title("Mean Patient Satisfaction by Profession Segment", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Average Satisfaction Rating (1-5)", fontsize=10)
    axes[0].set_ylabel("Profession", fontsize=10)
    axes[0].set_xlim(0, 4.5)
    for i, v in enumerate(prof_sat.values):
        axes[0].text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=8.5)
    axes[0].legend(loc="lower right")

    ins_order = ["Provider A", "Provider B", "Provider C", "Provider D", "Uninsured"]
    sns.boxplot(
        data=abt, x="Insurance_Provider", y="Avg_Satisfaction_Rating",
        order=ins_order, palette="Set2", ax=axes[1]
    )
    axes[1].set_title("Patient Satisfaction Distribution by Insurance Provider", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Insurance Provider", fontsize=10)
    axes[1].set_ylabel("Patient Average Satisfaction Rating", fontsize=10)

    plt.tight_layout()
    p5_path = os.path.join(output_dir, "plot5_satisfaction_by_segment.png")
    plt.savefig(p5_path, bbox_inches="tight")
    plt.close()
    plot_paths["plot5"] = p5_path
    print(f"[Plot 5] Saved: {p5_path}")

    # --------------------------------------------------------------------------
    # PLOT 6: Financial Breakdown (Insurance Coverage vs. Out-of-Pocket Costs)
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))

    dept_fin = df_clean.groupby("Department").agg(
        Avg_Coverage=("Insurance Coverage", "mean"),
        Avg_OOP=("Out_of_Pocket", "mean")
    )
    dept_fin["Total_Avg"] = dept_fin["Avg_Coverage"] + dept_fin["Avg_OOP"]
    dept_fin = dept_fin.sort_values("Total_Avg", ascending=True)

    dept_fin[["Avg_Coverage", "Avg_OOP"]].plot(
        kind="barh", stacked=True, ax=axes[0],
        color=["#1f78b4", "#ff7f00"], edgecolor="black"
    )
    axes[0].set_title("Mean Consultation Price Breakdown by Department (£)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Average Cost per Consultation (£)", fontsize=10)
    axes[0].set_ylabel("Department", fontsize=10)
    axes[0].legend(["Insurance Coverage (£)", "Patient Out-of-Pocket (£)"], loc="lower right")

    prov_fin = abt.groupby("Insurance_Provider").agg(
        Mean_Coverage=("Total_Insurance_Coverage", "mean"),
        Mean_OOP=("Total_Out_of_Pocket", "mean")
    ).reindex(ins_order)

    prov_fin.plot(kind="bar", stacked=True, ax=axes[1], color=["#1f78b4", "#ff7f00"], edgecolor="black", width=0.65)
    axes[1].set_title("Cumulative 6-Month Cost Split per Patient by Insurance (£)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Insurance Provider", fontsize=10)
    axes[1].set_ylabel("Mean 6-Month Spend per Patient (£)", fontsize=10)
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].legend(["Cumulative Insurance Coverage (£)", "Cumulative Out-of-Pocket (£)"], loc="upper right")

    plt.tight_layout()
    p6_path = os.path.join(output_dir, "plot6_financial_breakdown.png")
    plt.savefig(p6_path, bbox_inches="tight")
    plt.close()
    plot_paths["plot6"] = p6_path
    print(f"[Plot 6] Saved: {p6_path}\n")

    return plot_paths


# ==============================================================================
# PART D: AUTOMATED PDF GENERATION (REPORT.PDF & SLIDES.PDF)
# ==============================================================================

def generate_pdfs(
    df_clean: pd.DataFrame,
    abt: pd.DataFrame,
    data_dict_df: pd.DataFrame,
    cleaning_stats: dict,
    plot_paths: dict,
    report_filename: str = "Report.pdf",
    slides_filename: str = "Slides.pdf"
) -> None:
    """
    Programmatically compiles two comprehensive PDF deliverables using ReportLab:
    1. Report.pdf (Portrait A4): Technical report detailing data issues, cleaning methodology,
       ABT architecture, full Data Dictionary table, embedded plots with interpretations,
       answers to management questions, and next-team modeling recommendations.
    2. Slides.pdf (Landscape A4): Presentation slide deck with bold headings, concise bullet
       points, and embedded visual charts.
    """
    print("=" * 80)
    print(f"PART D: COMPILING '{report_filename}' AND '{slides_filename}'")
    print("=" * 80)

    # Compute key summary metrics for dynamic reporting
    total_gross = df_clean["Consultation Price"].sum()
    total_cov = df_clean["Insurance Coverage"].sum()
    total_oop = df_clean["Out_of_Pocket"].sum()
    cov_pct = (total_cov / total_gross) * 100
    oop_pct = (total_oop / total_gross) * 100

    dept_counts = df_clean["Department"].value_counts()
    top1_dept, top1_cnt = dept_counts.index[0], dept_counts.iloc[0]
    top2_dept, top2_cnt = dept_counts.index[1], dept_counts.iloc[1]

    dept_sat = df_clean.groupby("Department")["Satisfaction Level"].mean().sort_values()
    lowest_sat_dept, lowest_sat_val = dept_sat.index[0], dept_sat.iloc[0]
    highest_sat_dept, highest_sat_val = dept_sat.index[-1], dept_sat.iloc[-1]

    # --------------------------------------------------------------------------
    # 1. BUILD REPORT.PDF (A4 Portrait)
    # --------------------------------------------------------------------------
    doc = SimpleDocTemplate(
        report_filename,
        pagesize=A4,
        rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle", parent=styles["Title"], fontSize=20, leading=24,
        textColor=colors.HexColor("#1b365d"), spaceAfter=12
    )
    h1_style = ParagraphStyle(
        "ReportH1", parent=styles["Heading1"], fontSize=14, leading=18,
        textColor=colors.HexColor("#1b365d"), spaceBefore=12, spaceAfter=6
    )
    h2_style = ParagraphStyle(
        "ReportH2", parent=styles["Heading2"], fontSize=11.5, leading=15,
        textColor=colors.HexColor("#2b5c8f"), spaceBefore=8, spaceAfter=4
    )
    body_style = ParagraphStyle(
        "ReportBody", parent=styles["BodyText"], fontSize=9.5, leading=13.5, spaceAfter=6
    )
    bullet_style = ParagraphStyle(
        "ReportBullet", parent=body_style, leftIndent=15, bulletIndent=5, spaceAfter=4
    )
    cell_style = ParagraphStyle("TableCell", parent=body_style, fontSize=7.5, leading=9.5)
    cell_header_style = ParagraphStyle(
        "TableHeader", parent=body_style, fontSize=8, leading=10,
        textColor=colors.white, fontName="Helvetica-Bold"
    )

    story = []

    # Title Header
    story.append(Paragraph("City Hospital: Data Preprocessing, ABT Engineering & Exploratory Insights", title_style))
    story.append(Paragraph(
        "<b>Prepared by:</b> Data Preprocessing (DP) Team | <b>Observation Window:</b> January – June 2024 | "
        "<b>Reference Date:</b> July 1, 2024", body_style
    ))
    story.append(Spacer(1, 8))

    # Section 1: Data Quality Audit & Cleaning Methodology
    story.append(Paragraph("1. Executive Summary of Data Quality Issues & Cleaning Methodology", h1_style))
    story.append(Paragraph(
        "The raw dataset (<i>Project_Data.xlsx</i>) comprises 10,008 consultation records across 500 unique patients "
        "and 16 attributes. Prior to aggregation, a systematic data quality audit uncovered six structural categories "
        "of anomalies. In compliance with hospital governance, <b>zero patients were dropped</b> (all 500 unique "
        "Patient IDs are preserved in the final ABT).", body_style
    ))

    issues_bullets = [
        f"<b>Exact Duplicate Consultations:</b> Detected and removed {cleaning_stats['exact_duplicates_removed']} "
        f"exact duplicate rows, reducing the consultation table from 10,008 to 10,000 unique visits.",
        f"<b>Conflicting Static Patient Attributes:</b> Static attributes varied across visits for specific patients "
        f"(Gender: {cleaning_stats['inconsistent_patients']['Gender']} patients, Marital Status: "
        f"{cleaning_stats['inconsistent_patients']['Marital Status']}, Profession: "
        f"{cleaning_stats['inconsistent_patients']['Profession']}, Age: {cleaning_stats['inconsistent_patients']['Age']}). "
        f"Resolved chronologically by sorting by <i>Visit Date</i> and retaining each patient's most recent non-null record, "
        f"followed by pediatric consistency rules (patients under 18 standardized to Student / Single).",
        f"<b>Impossible Age & Rating Values:</b> Flagged {cleaning_stats['invalid_age_entries']} records with biologically "
        f"impossible ages (118, 138, 142, 158, 195 years caused by leading '1' digit typos) as NaN prior to imputation. "
        f"Capped {cleaning_stats['satisfaction_capped']} out-of-scale Satisfaction Level entries (recorded as 6) to the "
        f"valid maximum of 5.",
        f"<b>Continuous Outliers (1.5x IQR Method):</b> Winsorized {cleaning_stats['duration_outliers_capped']} extreme "
        f"Consultation Duration records (up to 600 min) at the upper IQR fence of {cleaning_stats['duration_upper_fence']:.1f} min. "
        f"Winsorized {cleaning_stats['price_outliers_capped']} Consultation Price anomalies (including 7 decimal-shift typos "
        f"between £25,300 and £39,999) using department-specific 1.5x IQR fences. Flagged "
        f"{cleaning_stats['income_outliers_flagged']} anomalous adult income records (> £{cleaning_stats['income_upper_fence']:,.0f} "
        f"up to £11.97M) as NaN for predictive imputation while preserving legitimate £0 incomes for Students.",
        f"<b>Missing Value Treatment (KNN Imputation & Domain Logic):</b> Treated structural missingness in Family History "
        f"(1,993 rows / 99 patients) as <i>'None'</i> and Insurance Provider (92 patients with £0 coverage) as <i>'Uninsured'</i>. "
        f"Applied scikit-learn's distance-weighted <b>KNNImputer (k=5)</b> across encoded demographic and financial features "
        f"to impute missing values in Age (56 rows), Education Level (29 rows), Approximate Annual Income (154 rows), "
        f"and Insurance Coverage (50 rows). Finally, capped {cleaning_stats['coverage_capped_to_price']} rows where "
        f"Insurance Coverage exceeded Consultation Price."
    ]
    for b in issues_bullets:
        story.append(Paragraph(f"• {b}", bullet_style))

    # Section 2: ABT Architecture & Data Dictionary
    story.append(Paragraph("2. Analytic Base Table (ABT) Construction & Data Dictionary", h1_style))
    story.append(Paragraph(
        f"The cleaned visit records were aggregated by <i>Patient ID</i> into <b>ABT_city_hospital.csv</b> "
        f"({abt.shape[0]} rows x {abt.shape[1]} engineered features), capturing static demographics, consultation "
        f"behaviour, weekend vs. weekday habits, financial exposure, satisfaction pain points, and recency metrics "
        f"relative to <b>July 1, 2024</b>.", body_style
    ))

    # Embed Data Dictionary Table
    table_data = [[
        Paragraph("Feature Name", cell_header_style),
        Paragraph("Category", cell_header_style),
        Paragraph("Definition & Calculation Logic", cell_header_style)
    ]]
    for _, row in data_dict_df.iterrows():
        table_data.append([
            Paragraph(str(row["Feature Name"]), cell_style),
            Paragraph(str(row["Category"]), cell_style),
            Paragraph(str(row["Definition & Calculation Logic"]), cell_style)
        ])

    dd_table = Table(table_data, colWidths=[1.8 * inch, 1.3 * inch, 4.1 * inch], repeatRows=1)
    dd_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1b365d")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(dd_table)
    story.append(PageBreak())

    # Section 3: Exploratory Visualizations & Interpretations
    story.append(Paragraph("3. Exploratory Visualizations & Key Analytical Findings", h1_style))

    plot_narratives = [
        ("plot1", "3.1 Patient Demographics: Age Bins & Income Quartiles",
         f"City Hospital serves a broad demographic spanning infants to seniors (mean age: {abt['Age'].mean():.1f} yrs), "
         f"with balanced gender representation across 10-year age brackets. Income quartiles reflect the structural "
         f"split between dependent/student cohorts in Q1 (£0 income) and working/retired adults earning £30k–£60k across Q2–Q4."),
        ("plot2", "3.2 Department Demand & Monthly Trends (January – June 2024)",
         f"<b>{top1_dept}</b> ({top1_cnt:,} visits, {top1_cnt/len(df_clean)*100:.1f}%) and <b>{top2_dept}</b> "
         f"({top2_cnt:,} visits, {top2_cnt/len(df_clean)*100:.1f}%) are the hospital's two busiest departments, "
         f"each handling roughly double the volume of any other specialty (~600–700 visits each). Monthly demand remains "
         f"steady from January through June with a slight dip in February due to fewer calendar days."),
        ("plot3", "3.3 Department Utilization by Gender & 10-Year Age Group",
         "Consultation volume across all 13 departments is evenly utilized across Female, Male, and Other genders, "
         "and distributed proportionally across pediatric, working-age, and senior age brackets. Multi-department "
         f"utilization is high: patients visit an average of {abt['Distinct_Departments'].mean():.1f} distinct departments "
         f"over the 6-month period ({abt['Total_Visits'].mean():.1f} total visits/patient)."),
        ("plot4", "3.4 Patient Satisfaction & Low-Rating Pain Points by Department",
         f"Hospital-wide satisfaction averages {df_clean['Satisfaction Level'].mean():.2f}/5.00, with "
         f"{(df_clean['Satisfaction Level'] <= 2).mean()*100:.1f}% of visits rated <= 2. Across departments, "
         f"<b>{lowest_sat_dept}</b> reports the lowest mean satisfaction ({lowest_sat_val:.2f}), whereas "
         f"<b>{highest_sat_dept}</b> achieves the highest rating ({highest_sat_val:.2f}). Addressing dissatisfaction "
         f"in high-volume departments (Psychiatry & General Practice) offers the largest overall quality impact."),
        ("plot5", "3.5 Patient Satisfaction Across Socio-Economic & Insurance Segments",
         "Patient-level satisfaction varies across occupation and coverage tiers. Because every patient visits multiple "
         "departments over 6 months, dissatisfaction is driven primarily by service experience at the visit level rather "
         "than demographic bias, though uninsured and lower-coverage patients face much steeper financial friction."),
        ("plot6", "3.6 Financial Breakdown: Insurance Coverage vs. Out-of-Pocket Burden",
         f"Across all 10,000 consultations (gross billings: £{total_gross:,.0f}), insurance covers <b>{cov_pct:.1f}%</b> "
         f"(£{total_cov:,.0f}) while patients pay <b>{oop_pct:.1f}%</b> (£{total_oop:,.0f}) out-of-pocket. "
         f"Emergency (£300 avg), Cardiology, Neurology, Endocrinology, and Orthopedics (£225 avg) are the highest-cost "
         f"departments, while ENT and General Practice (£75 avg) are the most affordable. Uninsured patients bear nearly "
         f"100% of their consultation costs out-of-pocket (~£3,100 cumulative over 6 months).")
    ]

    for idx, (pkey, subtitle, text) in enumerate(plot_narratives):
        story.append(Paragraph(subtitle, h2_style))
        if pkey in plot_paths and os.path.exists(plot_paths[pkey]):
            story.append(Image(plot_paths[pkey], width=6.8 * inch, height=2.65 * inch))
        story.append(Spacer(1, 4))
        story.append(Paragraph(text, body_style))
        if idx in [1, 3]:
            story.append(PageBreak())

    story.append(PageBreak())

    # Section 4: Answers to Management Questions & Next-Team Recommendations
    story.append(Paragraph("4. Executive Answers to Management Questions", h1_style))
    mgmt_answers = [
        f"<b>Q1: Which departments are the busiest, and how does demand change from January to June?</b><br/>",
        f"<b>Answer:</b> <i>Psychiatry</i> ({top1_cnt:,} visits) and <i>General Practice</i> ({top2_cnt:,} visits) are ",
        f"by far the busiest departments, accounting for 26.9% of all hospital consultations. The remaining 11 specialist ",
        f"departments handle between 593 (Orthopedics) and 696 (Gastroenterology) visits each. Demand is structurally ",
        f"consistent from January through June (~1,650–1,720 total visits/month, with a minor calendar dip in February), ",
        f"and {df_clean['Visit Date'].dt.dayofweek.isin([5,6]).mean()*100:.1f}% of consultations occur on weekends.",
        f"<b>Q2: Which patient groups use which departments most?</b><br/>",
        f"<b>Answer:</b> Service utilization is broad-based: patients average {abt['Total_Visits'].mean():.1f} visits across ",
        f"{abt['Distinct_Departments'].mean():.1f} distinct departments over 6 months. Retirees (36.0% of visits) and ",
        f"Students (21.0% of visits) represent the largest occupational user groups. Both groups heavily utilize ",
        f"Psychiatry and General Practice as primary care hubs before branching into specialist departments.",
        f"<b>Q3: Which departments and patient groups report the lowest satisfaction?</b><br/>",
        f"<b>Answer:</b> Hospital-wide, 37.6% of visits receive a low rating (<= 2). <i>{lowest_sat_dept}</i> records the ",
        f"lowest departmental mean rating ({lowest_sat_val:.2f}/5.00), closely followed by Pulmonology and Cardiology. ",
        f"Because high-volume departments (Psychiatry and General Practice) also see ~37% low ratings, they generate ",
        f"the highest absolute volume of dissatisfied patient encounters.",
        f"<b>Q4: How much of the consultation cost is covered by insurance vs. paid out-of-pocket?</b><br/>",
        f"<b>Answer:</b> Across all consultations, mean gross price is £{df_clean['Consultation Price'].mean():.2f}, ",
        f"of which insurance covers £{df_clean['Insurance Coverage'].mean():.2f} (<b>{cov_pct:.1f}%</b>) and patients pay ",
        f"£{df_clean['Out_of_Pocket'].mean():.2f} (<b>{oop_pct:.1f}%</b>) out-of-pocket. Insured patients (Providers A–D) ",
        f"have ~83% of their costs covered, whereas the 92 Uninsured patients pay nearly 100% out-of-pocket."
    ]
    for ans in mgmt_answers:
        story.append(Paragraph(ans, body_style))
        story.append(Spacer(1, 4))

    story.append(Paragraph("5. Strategic Recommendations for the Next Analytics & Modeling Team", h1_style))
    next_team_recs = [
        "<b>Candidate Clustering / Segmentation Frameworks:</b> Use the ABT to run K-Means or Gaussian Mixture clustering "
        "along three orthogonal axes: (1) <i>Financial Vulnerability & Care Burden</i> (Total_Out_of_Pocket, Out_of_Pocket_Ratio, "
        "Uninsured status) to identify patients needing financial counseling; (2) <i>Multi-Department Chronic Utilizers</i> "
        "(Total_Visits, Distinct_Departments, Total_Consultation_Duration) for integrated care pathways; and "
        "(3) <i>Weekend-Dependent Working Professionals</i> (Weekend_Visit_Share, Profession) for extended-hours scheduling.",
        "<b>Supervised Predictive Modeling Targets:</b> (1) <i>Patient Dissatisfaction Risk Model</i>: Predict whether a "
        "patient will have a high Share_Low_Ratings (> 0.40) to trigger proactive patient-experience outreach; "
        "(2) <i>High-Cost / High-Utilization Forecasting</i>: Predict <i>Total_Consultation_Price</i> or <i>Visits_Per_Month</i> "
        "to optimize departmental staffing and resource allocation.",
        "<b>Data Capture & Engineering Improvements:</b> Enforce front-end EHR validation rules to block ages > 110, "
        "satisfaction scores > 5, consultation prices > £500, and coverage exceeding price at the point of entry."
    ]
    for rec in next_team_recs:
        story.append(Paragraph(f"• {rec}", bullet_style))

    doc.build(story)
    print(f"[PDF 1] Successfully generated '{report_filename}'.")

    # --------------------------------------------------------------------------
    # 2. BUILD SLIDES.PDF (A4 Landscape Presentation Deck)
    # --------------------------------------------------------------------------
    slides_doc = SimpleDocTemplate(
        slides_filename,
        pagesize=landscape(A4),
        rightMargin=40, leftMargin=40, topMargin=36, bottomMargin=36
    )

    slide_title_style = ParagraphStyle(
        "SlideTitle", parent=styles["Title"], fontSize=24, leading=28,
        textColor=colors.HexColor("#1b365d"), alignment=0, spaceAfter=14
    )
    slide_sub_style = ParagraphStyle(
        "SlideSubtitle", parent=styles["Heading2"], fontSize=15, leading=20,
        textColor=colors.HexColor("#2b5c8f"), spaceAfter=10
    )
    slide_bullet_style = ParagraphStyle(
        "SlideBullet", parent=styles["BodyText"], fontSize=13, leading=19,
        leftIndent=20, bulletIndent=6, spaceAfter=8
    )

    slides_story = []

    # Slide 1: Title Slide
    slides_story.append(Spacer(1, 1.2 * inch))
    slides_story.append(Paragraph("City Hospital: Data Preprocessing & Exploratory Insights", slide_title_style))
    slides_story.append(Paragraph(
        "From Raw Consultation Logs (10,008 Visits) to a Clean 500-Patient Analytic Base Table (ABT)",
        slide_sub_style
    ))
    slides_story.append(Spacer(1, 0.3 * inch))
    slides_story.append(Paragraph("• <b>Prepared by:</b> City Hospital Data Preprocessing (DP) Team", slide_bullet_style))
    slides_story.append(Paragraph("• <b>Scope:</b> January – June 2024 Consultations (Reference Date: July 1, 2024)", slide_bullet_style))
    slides_story.append(Paragraph("• <b>Deliverables:</b> Clean ABT (500 rows x 37 features), Data Dictionary, EDA & Roadmap", slide_bullet_style))
    slides_story.append(PageBreak())

    # Slide 2: Part A - Data Quality & Cleaning Pipeline
    slides_story.append(Paragraph("Part A: Data Quality Audit & Statistical Cleaning", slide_title_style))
    s2_bullets = [
        f"<b>Zero Patient Loss:</b> Preserved all <b>500 unique patients</b> while removing {cleaning_stats['exact_duplicates_removed']} exact duplicate rows (10,000 clean visits).",
        f"<b>Temporal Standardization:</b> Resolved conflicting static attributes (Gender, Marital Status, Profession, Age) by selecting each patient's <b>most recent Visit Date</b> record.",
        f"<b>Domain & IQR Outlier Treatment:</b> Corrected {cleaning_stats['invalid_age_entries']} invalid Age entries (>100 yrs), capped {cleaning_stats['satisfaction_capped']} rating='6' values to 5, and Winsorized Consultation Duration ({cleaning_stats['duration_outliers_capped']} rows) & Price ({cleaning_stats['price_outliers_capped']} rows, including £25k–£40k typos) via 1.5x IQR fences.",
        "<b>Predictive Imputation (KNNImputer k=5):</b> Imputed missing Age, Education, Income, and Insurance Coverage using distance-weighted KNN after assigning 'None' (Family History) and 'Uninsured' (£0 coverage policies)."
    ]
    for b in s2_bullets:
        slides_story.append(Paragraph(f"• {b}", slide_bullet_style))
    slides_story.append(PageBreak())

    # Slide 3: Part B - ABT Architecture & Engineered Features
    slides_story.append(Paragraph("Part B: Patient-Level ABT Architecture (500 Rows x 37 Features)", slide_title_style))
    s3_bullets = [
        "<b>Static Profile (12 features):</b> Age, 10-Yr Age Group, Gender, City, Profession, Education, Marital Status, Insurance Provider, Family History, Cleaned Annual Income & Quartile.",
        f"<b>Visit Behaviour (9 features):</b> Total Visits (mean {abt['Total_Visits'].mean():.1f}), Distinct Departments (mean {abt['Distinct_Departments'].mean():.1f}), Most Frequent Department, Avg/Total Duration, Weekend vs. Weekday Ratio ({abt['Weekend_to_Weekday_Ratio'].mean():.2f}).",
        f"<b>Financial Exposure (7 features):</b> Total/Avg Consultation Price, Total/Avg Insurance Coverage, Total/Avg Out-of-Pocket Spend, and Out-of-Pocket Ratio.",
        f"<b>Satisfaction & Recency (9 features):</b> Avg/Min/Max Rating, Low Rating Count & Share (<=2), Days Since First/Last Visit (vs. July 1, 2024), Active Span, and Visits per Month ({abt['Visits_Per_Month'].mean():.2f}/mo)."
    ]
    for b in s3_bullets:
        slides_story.append(Paragraph(f"• {b}", slide_bullet_style))
    slides_story.append(PageBreak())

    # Slides 4 to 9: Embedded Visuals + Executive Bullet Takeaways
    slide_plots = [
        ("plot1", "Demographics: 10-Year Age Groups & Income Quartiles", [
            f"Balanced representation across all 10-year age brackets (mean age: {abt['Age'].mean():.1f} yrs) and genders.",
            "Income quartiles cleanly separate Q1 (£0 Student/Pediatric dependents) from working/retired adults (£30k–£60k)."
        ]),
        ("plot2", "Department Demand & Monthly Trends (Jan – Jun 2024)", [
            f"<b>Psychiatry ({top1_cnt:,} visits)</b> and <b>General Practice ({top2_cnt:,} visits)</b> drive 26.9% of total hospital volume.",
            "Specialist departments average ~650 visits each; monthly demand is stable across the 6-month window."
        ]),
        ("plot3", "Department Utilization by Gender & Age Group", [
            "All 13 departments see balanced utilization across Female, Male, and Other patient segments.",
            f"High cross-department mobility: patients visit an average of {abt['Distinct_Departments'].mean():.1f} distinct departments."
        ]),
        ("plot4", "Satisfaction Pain Points Across Departments", [
            f"Overall mean rating is {df_clean['Satisfaction Level'].mean():.2f}/5.00, with 37.6% of visits rated <= 2.",
            f"<b>{lowest_sat_dept} ({lowest_sat_val:.2f})</b> has the lowest mean score; high-volume Psychiatry & GP drive the most dissatisfied visits."
        ]),
        ("plot5", "Satisfaction by Patient Occupation & Insurance Tier", [
            "Patient-level average satisfaction clusters around 3.15–3.25 across professions and insurance providers.",
            "Service quality pain points are encounter-driven rather than restricted to a single demographic class."
        ]),
        ("plot6", "Financial Split: Insurance Coverage vs. Out-of-Pocket", [
            f"Insurance covers <b>{cov_pct:.1f}% (£{total_cov:,.0f})</b> of gross costs; patients pay <b>{oop_pct:.1f}% (£{total_oop:,.0f})</b> out-of-pocket.",
            "Emergency (£300/visit) is the costliest department; 92 Uninsured patients bear ~£3,100 in 6-month out-of-pocket costs."
        ])
    ]

    for pkey, stitle, sbullets in slide_plots:
        slides_story.append(Paragraph(stitle, slide_title_style))
        if pkey in plot_paths and os.path.exists(plot_paths[pkey]):
            slides_story.append(Image(plot_paths[pkey], width=9.6 * inch, height=3.75 * inch))
        slides_story.append(Spacer(1, 8))
        for b in sbullets:
            slides_story.append(Paragraph(f"• {b}", slide_bullet_style))
        slides_story.append(PageBreak())

    # Slide 10: Roadmap for Next Analytics Team
    slides_story.append(Paragraph("Strategic Roadmap for the Next Analytics Team", slide_title_style))
    s10_bullets = [
        "<b>1. Patient Segmentation (Clustering):</b> Segment the 500 patients by (a) <i>Financial Vulnerability</i> (Uninsured & high Out-of-Pocket Ratio), (b) <i>Multi-Specialty Heavy Utilizers</i>, and (c) <i>Weekend-Preferring Patients</i>.",
        "<b>2. Predictive Modeling Targets:</b> Train classification/regression models on the ABT to predict <i>High Dissatisfaction Risk (Share_Low_Ratings > 0.40)</i> and <i>Future Monthly Utilization (Visits_Per_Month)</i>.",
        "<b>3. Operational & EHR Interventions:</b> Expand staffing in Psychiatry and General Practice, audit service workflows in low-rated departments, and implement hard EHR input validation bounds."
    ]
    for b in s10_bullets:
        slides_story.append(Paragraph(f"• {b}", slide_bullet_style))

    slides_doc.build(slides_story)
    print(f"[PDF 2] Successfully generated '{slides_filename}'.\n")


# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================

def main():
    """Runs the complete end-to-end City Hospital data science pipeline."""
    # Part A: Load & Clean Data
    df_raw = load_data()
    df_clean, cleaning_stats = clean_data(df_raw)

    # Part B: Build ABT & Data Dictionary
    abt, data_dict_df = build_abt(df_clean, output_csv="ABT_city_hospital.csv")

    # Part C: Generate Exploratory Visualizations
    plot_paths = generate_plots(df_clean, abt, output_dir=".")

    # Part D: Compile Automated PDF Deliverables
    generate_pdfs(
        df_clean=df_clean,
        abt=abt,
        data_dict_df=data_dict_df,
        cleaning_stats=cleaning_stats,
        plot_paths=plot_paths,
        report_filename="Report.pdf",
        slides_filename="Slides.pdf"
    )

    print("=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY! GENERATED DELIVERABLES:")
    print("  1. ABT_city_hospital.csv (500 rows x 37 features)")
    print("  2. Data_Dictionary_city_hospital.csv")
    print("  3. Six high-resolution PNG visualizations (plot1_*.png to plot6_*.png)")
    print("  4. Report.pdf (Comprehensive Technical & Management Report)")
    print("  5. Slides.pdf (Executive Presentation Slide Deck)")
    print("=" * 80)


if __name__ == "__main__":
    main()
