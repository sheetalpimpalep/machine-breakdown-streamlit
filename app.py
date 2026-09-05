
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Machine Breakdown Risk Predictor",
    page_icon="⚙️",
    layout="wide"
)

application_directory = Path(
    __file__
).resolve().parent

metadata_path = (
    application_directory
    / "model_metadata.json"
)


@st.cache_resource
def load_application_assets():
    with open(
        metadata_path,
        "r",
        encoding="utf-8"
    ) as metadata_file:
        metadata = json.load(
            metadata_file
        )

    model_path = (
        application_directory
        / metadata["model_file"]
    )

    model = joblib.load(
        model_path
    )

    return model, metadata


try:
    model, metadata = (
        load_application_assets()
    )

except Exception as error:
    st.error(
        "The application could not load the "
        "model or metadata."
    )
    st.exception(error)
    st.stop()


st.title(
    "⚙️ Machine Breakdown Risk Predictor"
)

st.warning(
    metadata["warning"]
)

st.write(
    "Enter the current machine and operating "
    "conditions, then select Predict."
)

with st.expander(
    "Model information"
):
    st.write(
        "Selected model:",
        metadata["model_type"]
    )

    st.write(
        "Target:",
        metadata["target_column"]
    )

    st.write(
        "Scenario:",
        metadata["modelling_scenario"]
    )

    st.write(
        "Decision threshold:",
        round(
            float(
                metadata[
                    "decision_threshold"
                ]
            ),
            4
        )
    )


with st.form(
    "prediction_form"
):
    st.subheader(
        "Machine operating conditions"
    )

    left_column, right_column = (
        st.columns(2)
    )

    entered_values = {}

    for position, feature in enumerate(
        metadata["numeric_features"]
    ):
        specification = metadata[
            "numeric_inputs"
        ][feature]

        selected_column = (
            left_column
            if position % 2 == 0
            else right_column
        )

        with selected_column:
            entered_values[feature] = (
                st.number_input(
                    label=feature.replace(
                        "_",
                        " "
                    ),
                    min_value=float(
                        specification[
                            "minimum"
                        ]
                    ),
                    max_value=float(
                        specification[
                            "maximum"
                        ]
                    ),
                    value=float(
                        specification[
                            "default"
                        ]
                    ),
                    step=float(
                        specification[
                            "step"
                        ]
                    ),
                    format="%.2f"
                )
            )

    for position, feature in enumerate(
        metadata["categorical_features"]
    ):
        category_options = metadata[
            "categorical_inputs"
        ][feature]

        selected_column = (
            left_column
            if position % 2 == 0
            else right_column
        )

        with selected_column:
            entered_values[feature] = (
                st.selectbox(
                    label=feature.replace(
                        "_",
                        " "
                    ),
                    options=category_options
                )
            )

    predict_button = (
        st.form_submit_button(
            "Predict breakdown risk",
            type="primary"
        )
    )


if predict_button:
    try:
        expected_features = metadata[
            "model_features"
        ]

        input_record = pd.DataFrame([
            {
                feature: entered_values[
                    feature
                ]
                for feature
                in expected_features
            }
        ])

        predicted_probability = float(
            model.predict_proba(
                input_record
            )[0, 1]
        )

        if not np.isfinite(
            predicted_probability
        ):
            raise ValueError(
                "The model returned an invalid "
                "probability."
            )

        decision_threshold = float(
            metadata[
                "decision_threshold"
            ]
        )

        predicted_class = int(
            predicted_probability
            >= decision_threshold
        )

        if predicted_probability < 0.30:
            risk_level = "Low"

        elif predicted_probability < 0.70:
            risk_level = "Medium"

        else:
            risk_level = "High"

        st.divider()
        st.subheader(
            "Prediction result"
        )

        probability_column, class_column = (
            st.columns(2)
        )

        with probability_column:
            st.metric(
                "Breakdown probability",
                f"{predicted_probability:.1%}"
            )

        with class_column:
            st.metric(
                "Predicted class",
                (
                    "Breakdown risk"
                    if predicted_class == 1
                    else "No breakdown risk"
                )
            )

        if risk_level == "High":
            st.error(
                f"Risk level: {risk_level}"
            )

        elif risk_level == "Medium":
            st.warning(
                f"Risk level: {risk_level}"
            )

        else:
            st.success(
                f"Risk level: {risk_level}"
            )

        st.progress(
            min(
                max(
                    int(
                        predicted_probability
                        * 100
                    ),
                    0
                ),
                100
            )
        )

        st.caption(
            "Practice inference only. This output "
            "must not initiate real maintenance work."
        )

        with st.expander(
            "Show submitted inputs"
        ):
            st.dataframe(
                input_record,
                use_container_width=True,
                hide_index=True
            )

    except Exception as error:
        st.error(
            "Prediction failed. The record should "
            "be sent for manual review."
        )
        st.exception(error)


st.divider()

st.caption(
    "Synthetic machine-breakdown model — "
    "practice Streamlit deployment."
)
