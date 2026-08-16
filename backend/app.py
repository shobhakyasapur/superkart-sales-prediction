import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

superkart_api = Flask("SuperKart")

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "superkart_model.joblib"
)

model = joblib.load(MODEL_PATH)

EXPECTED_COLUMNS = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_char",
    "Store_Age_Years",
    "Product_Type_Category"
]


@superkart_api.get("/")
def home():
    return jsonify(
        {
            "message": "Welcome to the SuperKart Sales Prediction API",
            "status": "healthy"
        }
    )


@superkart_api.post("/v1/predict")
def predict_sales():
    try:
        data = request.get_json(force=True)

        missing = [
            col for col in EXPECTED_COLUMNS
            if col not in data
        ]

        if missing:
            return jsonify(
                {
                    "error": "Missing required fields",
                    "missing_fields": missing
                }
            ), 400

        sample = {
            col: data[col]
            for col in EXPECTED_COLUMNS
        }

        input_data = pd.DataFrame([sample])

        prediction = float(
            model.predict(input_data)[0]
        )

        return jsonify(
            {
                "Sales": round(prediction, 2)
            }
        )

    except Exception as exc:
        return jsonify(
            {
                "error": str(exc)
            }
        ), 500


@superkart_api.post("/v1/predictbatch")
def predict_sales_batch():
    try:
        if "file" not in request.files:
            return jsonify(
                {
                    "error": "CSV file is required using form field 'file'"
                }
            ), 400

        file = request.files["file"]
        input_data = pd.read_csv(file)

        missing = [
            col for col in EXPECTED_COLUMNS
            if col not in input_data.columns
        ]

        if missing:
            return jsonify(
                {
                    "error": "Batch file is missing required columns",
                    "missing_columns": missing
                }
            ), 400

        input_data = input_data[EXPECTED_COLUMNS]

        predictions = model.predict(input_data)

        output = [
            {
                "row": int(i),
                "Predicted_Sales": round(float(pred), 2)
            }
            for i, pred in enumerate(predictions)
        ]

        return jsonify(output)

    except Exception as exc:
        return jsonify(
            {
                "error": str(exc)
            }
        ), 500


if __name__ == "__main__":
    superkart_api.run(
        host="0.0.0.0",
        port=7860,
        debug=True,
        use_reloader=False
    )
