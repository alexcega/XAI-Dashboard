import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import numpy as np
import squarify  

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

st.set_page_config(
    page_title="Housing Price Tool",
    layout="wide",               # full-width mode
    initial_sidebar_state="auto"
)


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
bedrooms = st.sidebar.slider("Bedrooms", min_value=bedrooms_min, max_value=bedrooms_max  ,value=2)
bathrooms = st.sidebar.slider("Bathrooms", min_value=bathrooms_min, max_value=bathrooms_max  ,value=2)
stories = st.sidebar.slider("Stories", min_value=stories_min, max_value=stories_max  ,value=1)
parking = st.sidebar.slider("Parking spaces", min_value=parking_min, max_value=parking_max  ,value=1)


binary_options = ["yes", "no"]
mainroad = st.sidebar.checkbox("Main Road", value=True)
guestroom = st.sidebar.checkbox("Guest Room",value=True)
basement = st.sidebar.checkbox("Basement", value=True)
hotwaterheating = st.sidebar.checkbox("Hot Water Heating", value=True)
airconditioning = st.sidebar.checkbox("Air Conditioning", value=True)
prefarea = st.sidebar.checkbox("Preferred Area", value=True)

furn_options = ["furnished", "semi-furnished", "unfurnished"]
furnishingstatus = st.sidebar.selectbox("Furnishing Status", furn_options)

# Prepare and encode input DataFrame
input_dict = {
    "area": area,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "stories": stories,
    "mainroad": mainroad,  # Ensure boolean
    "guestroom": guestroom,  # Ensure boolean
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
# Convert boolean columns to 1/0

binary_cols = ["mainroad","guestroom","basement","hotwaterheating","airconditioning","prefarea"]

input_df[binary_cols] = input_df[binary_cols].astype(int)
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
    col1, col2 ,col3 = st.columns([2,5,2], gap="medium")
    with col2:
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
        green_shades = ["#58B072", "#669E8C", '#167D7F', '#107869', 'green', "#2AAE9A"]
        red_shades   = ["#B13636", '#F8BABA', '#E05A5A', '#C43E3E', 'red', 'darkred']

        furn_color  = {
            "furnished":    "#green",   # pale green
            "semi-furnished":"#yellow",  # pale yellow
            "unfurnished":  "#red"    # pale red
        }   
        st.write()
        yes_color = dict(zip(binary_cols, green_shades))
        no_color  = dict(zip(binary_cols, red_shades))
        for i, feat in enumerate(feature_cols):
            raw = input_dict[feat]
            # pick color based on feature type
            if feat in binary_cols:
                bg = yes_color[feat] if raw == True else no_color[feat]
            elif feat == "furnishingstatus":
                bg = furn_color.get(raw, "#000000")
            else:
                bg = color_map.get(feat, "#eeeeee")

            col = cols[i % n_cols]
            with col:                
                st.markdown(f"<div style='font-size:16px; font-weight:bold; text-align:center, color:black;'>{feat.replace('_',' ').title()}</div>", unsafe_allow_html=True)
                # colored box with the value
                st.markdown(f"""
                    <div style="
                        background-color: {bg};
                        padding: 30px;
                        border-radius: 6px;
                        text-align: center;
                        font-size:30px;
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

    st.markdown("---")
    st.subheader("Feature Impact (Local Explanation)")
    col1, col2 ,col3 = st.columns([1,5,1], gap="medium")
    with col2:
        ccol1, ccol2 = st.columns([3,2], gap="medium")
        with ccol1:
        #* Local SHAP Explanation
        
            shap_values = explainer(input_df)
            
            sv = shap_values.values[0]
            feat_names = [f.replace('_',' ').title() for f in feature_cols]

                
            # Compute sizes and percentages
            sizes = np.abs(sv)
            total = sizes.sum()
            percs = [(v/total)*100 for v in sizes]
            labels = [f"{name}\n{perc:.1f}%" for name, perc in zip(feat_names, percs)]

            # Build a continuous red/green map by sign & magnitude
            norm = Normalize(vmin=-max(abs(sv)), vmax=+max(abs(sv)))
            cmap_pos = cm.get_cmap("Greens")
            cmap_neg = cm.get_cmap("Reds")
            colors = [cmap_pos(norm(v)) if v>=0 else cmap_neg(norm(v)) for v in sv]

            # Draw the treemap
            fig, ax = plt.subplots(figsize=(6,6))
            squarify.plot(
                sizes=sizes,
                label=labels,
                color=colors,
                pad=True,
                text_kwargs={"weight":"bold"},
                ax=ax
            )
            ax.axis('off')

            # Auto-scale font & rotate tiny boxes
            texts = ax.texts
            rects = ax.patches
            min_a, max_a = sizes.min(), sizes.max()
            areas = np.array([rect.get_width()*rect.get_height() for rect in rects])
            for txt, rect, area in zip(texts, rects, areas):
                # scale font size based on area
                fs = 8 + (area - min_a) / (max_a - min_a) * 10
                txt.set_fontsize(fs)
                # measure if text width > rect width
                renderer = fig.canvas.get_renderer()
                bb = txt.get_window_extent(renderer=renderer)
                # convert display coords to data coords
                inv = ax.transData.inverted()
                bb_data = inv.transform(bb)
                text_width = bb_data[1,0] - bb_data[0,0]
                text_height = bb_data[1,1] - bb_data[0,1]
                w, h = rect.get_width(), rect.get_height()
                # 3) if overflow horizontally, rotate
                if text_width > w and not text_height > h:
                    txt.set_rotation(90)
                    txt.set_va("center")
                    txt.set_ha("center")

            # 6) Render
            st.pyplot(fig)

        with ccol2:
            # max_pct = max(p for _, p in percs)
            feat_perc = list(zip(feat_names, percs))
            feat_perc.sort(key=lambda x: x[1], reverse=True)
            counter = 1
            for name, pct in feat_perc:
                # compute bar width as a percentage of the full column
                width = (pct / 100) *200
                st.markdown(f"""
                <div style="display: flex; align-items: center; margin-bottom: 4px;">
                <!-- feature name -->
                <div style="flex: 1; font-weight: bold; font-size: 1.2em;">{counter}. {name}</div>
                <!-- colored bar -->
                <div style="
                    flex: ;
                    background-color: #539ecd;
                    height: 14px;
                    width: {width}%;
                    border-radius: 4px;
                    margin-left: 8px;
                    position: relative;
                ">
                    <!-- overlay text showing the percentage -->
                    <span style="
                        position: absolute;
                        right: 4px;
                        top: -18px;
                        font-size: 0.75em;
                    ">
                    {pct:.1f}%
                    </span>
                </div>
                </div>
                """, unsafe_allow_html=True)
                counter += 1
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
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm

    st.header("Global Feature Importance")
    feature_cols = ['area','bedrooms','bathrooms','stories','mainroad','guestroom','basement','hotwaterheating','airconditioning','parking','prefarea','furnishingstatus']
    @st.cache_data
    def load_train_features():
        return pd.read_csv("train_features.csv")
    
    X_train = load_train_features()[feature_cols]
    shap_values_full = explainer(X_train)

    glob_imp = pd.Series(np.abs(shap_values_full.values).mean(axis=0), index=feature_cols).sort_values(ascending=False)
    n = len(glob_imp)
    cmap = cm.get_cmap('viridis', n)
    max_height = glob_imp.values.max()

    bar_width = 60
    chart_height = 320
    bar_gap = 20

    # First row: bars with value % on top
    bars_html = "<div style='display: flex; align-items: flex-end; height: {}px; gap: {}px;'>".format(chart_height, bar_gap)
    for i, (name, val) in enumerate(glob_imp.items()):
        color = mcolors.to_hex(cmap(i / (n-1) if n > 1 else 0))
        pct = (val / max_height) * 100
        height_px = max(int((val / max_height) * (chart_height - 60)), 20)
        bars_html += f"""
        <div style='flex:1; display: flex; flex-direction: column; align-items: center; justify-content: flex-end;'>
            <div style='
                font-size: 1.08em; 
                margin-bottom: 4px; 
                color: #fff; 
                font-weight: 600; 
                text-shadow: 0 1px 4px #000c;
                height: 1.5em;
            '>{pct:.1f}%</div>
            <div style='
                width: {bar_width}px;
                height: {height_px}px;
                background: {color};
                border-radius: 9px 9px 0 0;
                box-shadow: 0 2px 8px #0002;
                margin-bottom: 0;
                display: flex;
                align-items: flex-end;
            ' title="{name}: {val:.2f}"></div>
        </div>"""
    bars_html += "</div>"

    # Second row: feature names, rotated, spaced to match bars
    labels_html = "<div style='display: flex; gap: {}px; margin-top: 30px;'>".format(bar_gap)
    for name in glob_imp.keys():
        labels_html += f"""
        <div style='flex:1; display: flex; justify-content: center;'>
            <div style='
                font-size: 1em;
                text-align: center;
                max-width: 110px;
                overflow: hidden;
                white-space: normal;
                word-break: break-word;
                transform: rotate(-60deg);
                margin-top: 0px;
                margin-bottom: 2px;
                color: #fff;
                text-shadow: 0 1px 6px #000a;
            '>{name}</div>
        </div>"""
    labels_html += "</div>"

    st.markdown(f"""
    <div style='width: 100%;'>
    <div style='font-size:1.3em; font-weight:700; margin-bottom:20px; color:#fff;'>
        Global Feature Importance
    </div>
    {bars_html}
    {labels_html}
    </div>
    """, unsafe_allow_html=True)
    bar_html = "<div style='display: flex; align-items: flex-end; height: 240px; gap: 14px;'>"

    bar_html += "</div>"



    st.subheader("Local SHAP Values Table")
    idx = st.number_input("Select Sample Index", 0, len(X_train)-1, 0)
    local_shap = explainer(X_train.iloc[[idx]])
    st.table(pd.DataFrame({"feature": feature_cols, "shap_value": local_shap.values[0]}).sort_values(by="shap_value", ascending=False))
