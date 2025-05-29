import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Load model and SHAP explainer using new cache APIs
@st.cache_resource
def load_model():
    # Model was trained on ordinal-encoded features
    return joblib.load("house_price_model.pkl")

@st.cache_resource
def load_explainer():
    return joblib.load("shap_explainer.pkl")

model = load_model()
explainer = load_explainer()

# Sidebar: User Persona
st.sidebar.title("Housing Price Tool")
persona = st.sidebar.selectbox(
    "I am a…", 
    ["First-time Buyer", "Real Estate Agent", "Data Scientist"]
)

# Sidebar: Property features Input
st.sidebar.header("Property Features")
area = st.sidebar.number_input("Area (sqft)", min_value=100, value=1000)
bedrooms = st.sidebar.number_input("Bedrooms", min_value=1, value=3)
bathrooms = st.sidebar.number_input("Bathrooms", min_value=1, value=2)
stories = st.sidebar.number_input("Stories", min_value=1, value=1)
parking = st.sidebar.number_input("Parking spaces", min_value=0, value=1)

binary_options = ["yes", "no"]
mainroad = st.sidebar.selectbox("Main Road", binary_options)
guestroom = st.sidebar.selectbox("Guest Room", binary_options)
basement = st.sidebar.selectbox("Basement", binary_options)
hotwaterheating = st.sidebar.selectbox("Hot Water Heating", binary_options)
airconditioning = st.sidebar.selectbox("Air Conditioning", binary_options)
prefarea = st.sidebar.selectbox("Preferred Area", binary_options)

furn_options = ["furnished", "semi-furnished", "unfurnished"]
furnishingstatus = st.sidebar.selectbox("Furnishing Status", furn_options)

# Prepare and encode input DataFrame
input_dict = {
    "area": area,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "stories": stories,
    "mainroad": mainroad,
    "guestroom": guestroom,
    "basement": basement,
    "hotwaterheating": hotwaterheating,
    "airconditioning": airconditioning,
    "parking": parking,
    "prefarea": prefarea,
    "furnishingstatus": furnishingstatus
}
# Create raw DataFrame
input_df = pd.DataFrame([input_dict])

# Encode binaries: yes->1, no->0
binary_cols = ["mainroad","guestroom","basement","hotwaterheating","airconditioning","prefarea"]
for col in binary_cols:
    input_df[col] = input_df[col].map({"yes":1, "no":0})

# Encode furnishingstatus ordinally to match training
furn_map = {"furnished":0, "semi-furnished":1, "unfurnished":2}
input_df["furnishingstatus"] = input_df["furnishingstatus"].map(furn_map)

# Ensure column order matches training
feature_cols = ["area","bedrooms","bathrooms","stories", "mainroad","guestroom","basement","hotwaterheating","airconditioning","parking","prefarea","furnishingstatus"]
input_df = input_df[feature_cols]

# Compare feature names
trained_feats = list(model.feature_names_in_)
input_feats = list(input_df.columns)
# Show mismatches
missing_in_input = set(trained_feats) - set(input_feats)
extra_in_input = set(input_feats) - set(trained_feats)

if missing_in_input:
    st.error(f"Missing in input: {missing_in_input}")
if extra_in_input:
    st.error(f"Extra in input: {extra_in_input}")

# Only predict if no mismatches
if not missing_in_input and not extra_in_input:
    pred_price = model.predict(input_df)[0]
    st.metric("Predicted Price", f"${pred_price:,.0f}")
else:
    st.stop()

# Persona-specific insights
if persona == "First-time Buyer":
    st.header("Why this price?")
    n_cols = 4  
    cols = st.columns(n_cols)

    # pre-define a color for each feature (or generate at runtime)
    color_map = {
        "area": "#fde0dd",
        "bedrooms": "#fa9fb5",
        "bathrooms": "#c51b8a",
        "stories": "#e7298a",
        "parking": "#df65b0",
        "mainroad": "#d4b9da",
        "guestroom": "#d9f0d3",
        "basement": "#a6dba0",
        "hotwaterheating": "#5aae61",
        "airconditioning": "#1b7837",
        "prefarea": "#b8e186",
        "furnishingstatus": "#7fbc41"
    }

    for i, feat in enumerate(feature_cols):
        val = input_dict[feat]
        col = cols[i % n_cols]  # wrap to next row automatically
        with col:
            st.markdown(
                f"""
                <div style="
                    background-color: {color_map.get(feat, '#eeeeee')};
                    padding: 10px;
                    border-radius: 8px;
                    text-align: center;
                    ">
                  <strong>{feat.replace('_',' ').title()}</strong><br>
                  {val}
                </div>
                """,
                unsafe_allow_html=True
            )
    if st.button("What if area increases by 10%?"):
        mod_df = input_df.copy()
        mod_df.at[0,"area"] *= 1.1
        new_price = model.predict(mod_df)[0]
        st.write(f"**New predicted price:** ${new_price:,.0f}")
    shap_values = explainer(input_df)
    shap_val = shap_values[0]
    fig, ax = plt.subplots()
    st.subheader("Local Explanation (SHAP)")
    shap.plots.waterfall(shap_val,  show=False)
    st.pyplot(fig)

elif persona == "Real Estate Agent":
    st.header("Market Comparison")
    @st.cache_data
    def load_listings():
        return pd.read_csv("comparison_listings.csv")
    df = load_listings()
    # Apply same encoding to listings before predict
    for col in binary_cols:
        df[col] = df[col].map({"yes":1, "no":0})
    df["furnishingstatus"] = df["furnishingstatus"].map(furn_map)
    df["predicted_price"] = model.predict(df[feature_cols])
    st.line_chart(df.set_index("id")["predicted_price"])
    csv = df.to_csv(index=False)
    st.download_button("Download Market Report", csv, "report.csv")

else:
    st.header("Global Feature Importance")
    @st.cache_data
    def load_train_features():
        return pd.read_csv("train_features.csv")
    X_train = load_train_features()[feature_cols]
    shap_values_full = explainer(X_train)
    import numpy as np
    import pandas as _pd
    glob_imp = _pd.Series(np.abs(shap_values_full.values).mean(axis=0), index=feature_cols).sort_values(ascending=False)
    st.bar_chart(glob_imp)
    st.subheader("Local SHAP Values Table")
    idx = st.number_input("Select Sample Index", 0, len(X_train)-1, 0)
    local_shap = explainer(X_train.iloc[[idx]])
    st.table(_pd.DataFrame({"feature": feature_cols, "shap_value": local_shap.values[0]}).sort_values(by="shap_value", ascending=False))
