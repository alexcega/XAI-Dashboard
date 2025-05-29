import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# get max values
df = pd.read_csv("Housing.csv")      # or your chosen CSV
area_min   = df["area"].min()  
bedrooms_min   = df["bedrooms"].min()
bathrooms_min = df["bathrooms"].min()
stories_min = df["stories"].min()
parking_min = df["parking"].min()

area_max   = df["area"].max()  
bedrooms_max   = df["bedrooms"].max()
bathrooms_max = df["bathrooms"].max()
stories_max = df["stories"].max()
parking_max = df["parking"].max()


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
area = st.sidebar.slider("Area (sqft)", min_value=area_min, max_value=area_max ,value=area_max // 2)
bedrooms = st.sidebar.slider("Bedrooms", min_value=bedrooms_min, max_value=bedrooms_max  ,value=3)
bathrooms = st.sidebar.slider("Bathrooms", min_value=bathrooms_min, max_value=bathrooms_max  ,value=2)
stories = st.sidebar.slider("Stories", min_value=stories_min, max_value=stories_max  ,value=1)
parking = st.sidebar.slider("Parking spaces", min_value=parking_min, max_value=parking_max  ,value=1)


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
    st.title("Predicted Price", )
    st.metric("", f"${pred_price:,.0f}")
else:
    st.stop()

# Persona-specific insights
if persona == "First-time Buyer":
    age = st.slider("How old are you?", 0, 130, 25)
    st.write("I'm ", age, "years old")
    st.header("Why this price?")
    n_cols = 4  
    cols = st.columns(n_cols)

    # pre-define a color for each feature (or generate at runtime)
    color_map = {
        "area": "#05445E",
        "bedrooms": "#189AB4",
        "bathrooms": "#B5E5CF",
        "stories": "#1B7E89",
        "parking": "#189AB4",
        "mainroad": "#75E6DA",
        "guestroom": "#d9f0d3",
        "basement": "#a6dba0",
        "hotwaterheating": "#5aae61",
        "airconditioning": "#1b7837",
        "prefarea": "#b8e186",
        "furnishingstatus": "#bc4141"
    }
    binary_color = {"yes": ["#74A483", "#98D7C2", '#167D7F', '#107869'], "no": "red"}
    green_shades = ["#58B072", "#669E8C", '#167D7F', '#107869', 'green', 'green']
    red_shades   = ["#B13636", '#F8BABA', '#E05A5A', '#C43E3E', 'red', 'darkred']

    furn_color  = {
        "furnished":    "#green",   # pale green
        "semi-furnished":"#yellow",  # pale yellow
        "unfurnished":  "#red"    # pale red
    }   
    yes_color = dict(zip(binary_cols, green_shades))
    no_color  = dict(zip(binary_cols, red_shades))
    for i, feat in enumerate(feature_cols):
        raw = input_dict[feat]
        # pick color based on feature type
        if feat in binary_cols:
            bg = yes_color[feat] if raw == "yes" else no_color[feat]
        elif feat == "furnishingstatus":
            bg = furn_color.get(raw, "#000000")
        else:
            bg = color_map.get(feat, "#eeeeee")

        col = cols[i % n_cols]
        with col:
            # feature name
            st.markdown(f"<div style='font-size:18px; font-weight:bold; text-align:center, color:black;'>{feat.replace('_',' ').title()}</div>", unsafe_allow_html=True)
            # colored box with the value
            st.markdown(f"""
                <div style="
                    background-color: {bg};
                    padding: 30px;
                    border-radius: 6px;
                    text-align: center;
                    font-size:32px;
                ">
                {raw}
                </div>
                """, unsafe_allow_html=True)
        
    st.markdown("---")
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
