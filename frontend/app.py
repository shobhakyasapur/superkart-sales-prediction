import streamlit as st
import pandas as pd
import requests

BACKEND_URL = "http://backend:7860"

st.set_page_config(
    page_title="SuperKart Sales Prediction",
    layout="centered"
)

st.title("SuperKart Sales Prediction")
st.write(
    "Enter product and store information to predict Product Store Sales Total."
)

Product_Weight = st.number_input(
    "Product Weight",
    min_value=0.0,
    value=12.66
)

Product_Sugar_Content = st.selectbox(
    "Product Sugar Content",
    ["Low Sugar", "Regular", "No Sugar"]
)

Product_Allocated_Area = st.number_input(
    "Product Allocated Area",
    min_value=0.0,
    value=0.027,
    format="%.4f"
)

Product_MRP = st.number_input(
    "Product MRP",
    min_value=0.0,
    value=117.08
)

Store_Size = st.selectbox(
    "Store Size",
    ["Small", "Medium", "High"]
)

Store_Location_City_Type = st.selectbox(
    "Store Location City Type",
    ["Tier 1", "Tier 2", "Tier 3"]
)

Store_Type = st.selectbox(
    "Store Type",
    [
        "Supermarket Type1",
        "Supermarket Type2",
        "Supermarket Type3",
        "Departmental Store",
        "Food Mart"
    ]
)

Product_Id_char = st.selectbox(
    "Product ID Character",
    ["FD", "DR", "NC"]
)

Store_Age_Years = st.number_input(
    "Store Age (Years)",
    min_value=0,
    value=16,
    step=1
)

Product_Type_Category = st.selectbox(
    "Product Type Category",
    ["Perishables", "Non Perishables"]
)

product_data = {
    "Product_Weight": Product_Weight,
    "Product_Sugar_Content": Product_Sugar_Content,
    "Product_Allocated_Area": Product_Allocated_Area,
    "Product_MRP": Product_MRP,
    "Store_Size": Store_Size,
    "Store_Location_City_Type": Store_Location_City_Type,
    "Store_Type": Store_Type,
    "Product_Id_char": Product_Id_char,
    "Store_Age_Years": Store_Age_Years,
    "Product_Type_Category": Product_Type_Category
}

if st.button("Predict", type="primary"):
    try:
        response = requests.post(
            f"{BACKEND_URL}/v1/predict",
            json=product_data,
            timeout=30
        )

        if response.status_code == 200:
            predicted_sales = response.json()["Sales"]
            st.success(
                f"Predicted Product Store Sales Total: ₹{predicted_sales:,.2f}"
            )
        else:
            st.error(
                f"Prediction failed: {response.text}"
            )

    except requests.RequestException as exc:
        st.error(
            f"Unable to connect to backend: {exc}"
        )


st.divider()
st.subheader("Batch Prediction")

uploaded_file = st.file_uploader(
    "Upload Batch_Data_SuperKart.csv",
    type=["csv"]
)

if uploaded_file is not None:
    st.dataframe(
        pd.read_csv(uploaded_file),
        use_container_width=True
    )

    uploaded_file.seek(0)

    if st.button("Predict Batch", type="primary"):
        try:
            response = requests.post(
                f"{BACKEND_URL}/v1/predictbatch",
                files={
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        "text/csv"
                    )
                },
                timeout=60
            )

            if response.status_code == 200:
                results = response.json()
                results_df = pd.DataFrame(results)

                st.success(
                    "Batch predictions completed successfully."
                )

                st.dataframe(
                    results_df,
                    use_container_width=True
                )
            else:
                st.error(
                    f"Batch prediction failed: {response.text}"
                )

        except requests.RequestException as exc:
            st.error(
                f"Unable to connect to backend: {exc}"
            )
