import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.ticker as ticker
import numpy as np
import squarify  
import random
import os
import seaborn as sns

# st configuration
st.set_page_config(
    page_title="Housing Price Tool",
    layout="wide",               # full-width mode
    initial_sidebar_state="auto",

)

# Page syle
st.markdown("""
<style>
/* Default: normal size */
.responsive-cell {
    padding: 30px;
    font-size: 30px;
    border-radius: 6px;
    text-align: center;
    transition: padding 0.2s, font-size 0.2s;
}
/* When window is small (sidebar likely overlaying) */
@media (max-width: 1200px) {
    .responsive-cell {
        padding: 16px !important;
        font-size: 18px !important;
    }
}
</style>
""", unsafe_allow_html=True)



# Functions
# @st.cache_resource
def load_model():
    # Model was trained on ordinal-encoded features
    return joblib.load("house_price_rf.pkl")

# @st.cache_resource
def load_explainer():
    return joblib.load("shap_explainer.pkl")

# Global vars
model = load_model()
explainer = load_explainer()
    # image directories
house_dir = "Data_houses/"
house_files = [f for f in os.listdir(house_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

# get max values for users input
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


## not in session elements
if "current_house_file" not in st.session_state:
    st.session_state.current_house_file = random.choice(house_files)

if 'area' not in st.session_state:
    st.session_state.area = area_max // 2

# if 'pred_price' not in st.session_state:
#     temp_input = input_df.copy()
#     temp_input.at[0, "area"] = st.session_state.area
#     st.session_state.pred_price = model.predict(temp_input)[0]


# Sidebar: User Persona
st.sidebar.title("Housing Price Tool")
persona = st.sidebar.selectbox(
    "I am a…", 
    ["First-time Buyer", "Real Estate Agent", "Data Scientist"]
)

# Sidebar: Property features Input
st.sidebar.header("Property Features")
area = st.sidebar.slider("Area (sqft)", min_value=area_min, max_value=area_max ,value=area_max // 2)

# Sidebar slider uses the session state value, and updates session state
st.session_state.area = st.sidebar.slider(
    "Area (sqft)",
    min_value=area_min,
    max_value=area_max,
    value=st.session_state.area,
    key="area_slider"
)

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

st.title("House price prediction", )
if missing_in_input:
    st.error(f"Missing in input: {missing_in_input}")
if extra_in_input:
    st.error(f"Extra in input: {extra_in_input}")

# Only predict if no mismatches
if not missing_in_input and not extra_in_input:
    pred_price = model.predict(input_df)[0]
    st.markdown(
        f"""
        <div style='
            font-size: 64px; 
            font-weight: bold; 
            color: #white; 
            text-align: right;
            margin-bottom: 12px;
        '>
            ${pred_price:,.0f}
        </div>
        <div style='font-size: 20px; color: #777; text-align: right;'>
            Predicted Price
        </div>
        """, 
        unsafe_allow_html=True
    )
else:
    st.stop()

# Persona-specific insights
if persona == "First-time Buyer":
    
    col1, col2 ,col3 = st.columns([2.5,3,2], gap="large")
    # grid of attributes
    with col1:
        st.markdown("### Why this price?")
        n_cols = 3  
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
        green_shades = ["#58B072", "#669E8C", '#167D7F', '#107869', 'green', "#2AAE9A"]
        red_shades   = ["#B13636", '#F8BABA', '#E05A5A', '#C43E3E', 'red', 'darkred']

        furn_color  = {
            "furnished":    "#177048",   # pale green
            "semi-furnished":"#E0DA28",  # pale yellow
            "unfurnished":  "#CE3A1C"    # pale red
        }   

        furn_display = {
            0: "Yes",            # if encoded (int)
            1: "Semi",
            2: "No",
            "furnished": "Yes",  # if not encoded (str)
            "semi-furnished": "Semi",
            "unfurnished": "No"
        }
        yes_color = dict(zip(binary_cols, green_shades))
        no_color  = dict(zip(binary_cols, red_shades))
        for i, feat in enumerate(feature_cols):
            raw = input_dict[feat]
            if feat == "furnishingstatus":
                display_val = furn_display.get(raw, str(raw))
            else:
                display_val = raw

            # pick color based on feature type
            if feat in binary_cols:
                bg = yes_color[feat] if raw == True else no_color[feat]
            elif feat == "furnishingstatus":
                bg = furn_color.get(raw, "#000000")
            else:
                bg = color_map.get(feat, "#eeeeee")

            col = cols[i % n_cols]
            with col:  
                # st.write(feat)    
                if feat == 'hotwaterheating':
                    feat = 'Heating'
                if feat.title() == 'Furnishingstatus':
                    feat = "Furnished"
                if feat == "hotwaterheating":
                    feat ="Heating"       
                if feat == 'airconditioning':
                    feat = 'A/C'
                st.markdown(f"<div style='font-size:16px; font-weight:bold; text-align:center, color:black;'>{feat.replace('_',' ').title()}</div>", unsafe_allow_html=True)
                # colored box with the value
                st.markdown(f"""
                <div class="responsive-cell" style="
                    background-color: {bg};
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    height: 100%;   /* fill cell height */
                    min-height: 48px; /* optional: avoid tiny cells */
                    text-align: center;
                    border-radius: 6px;
                ">
                    {display_val}
                </div>
                """, unsafe_allow_html=True)
    # show image of houses
    with col2:
        st.markdown("### House preview")
        if st.sidebar.button("Show"):
            st.session_state.current_house_file = random.choice(house_files)
            house_dir = "Data_houses/"
            random_file=random.choice(os.listdir(house_dir))
            st.image("Data_houses/"+random_file,  use_container_width=True, caption="Example Home")  # Use your image path

    #* Local SHAP Explanation
    shap_values = explainer(input_df)
    sv = shap_values.values[0]
    feat_names = [f.replace('_',' ').title() for f in feature_cols]
    # Compute sizes and percentages
    sizes = np.abs(sv)
    total = sizes.sum()
    percs = [(v/total)*100 for v in sizes]
    feat_perc = list(zip(feat_names, percs))
    feat_perc.sort(key=lambda x: x[1], reverse=True)
    # user questions

    with col3:
        row1, row2 = st.columns([1, 1])  # Use columns to split vertically (Streamlit doesn't support sub-rows directly, so this is a trick)
        # But for real vertical splitting, use container or just stacked elements:
        st.markdown("### What if Area Changes?")
        # e.g. area increase
        pct1 = st.slider("Increase area by (%)", 0, 50, 10, key="area_inc")
        new_area = int(input_df.at[0, "area"] * (1 + pct1/100))
        st.write(f"**New area:** {new_area} sqft ({pct1}% increase)")
        if st.button("Predict New Price", key="p1"):
            mod_df = input_df.copy()
            input_df.at[0, "area"] = new_area
            new_price = model.predict(input_df)[0]
            st.success(f"New predicted price: ${new_price:,.0f}")

            # Update price
            # total_price.

            # update house

        st.markdown("---")
        st.markdown("### What are the least important features?")
        for name, percentage in feat_perc:
            if percentage < 3:
                st.write(name, "{:.2f}".format(percentage), "%")

    st.markdown("---")
    st.subheader("Feature Impact (Local Explanation)")

    col1, col2 ,col3 = st.columns([2,3,2], gap="medium")
    with col1:
        row1, row2 = st.columns([1, 1])
        st.markdown("""
            <div style='display: gird; align-items: center; gap: 24px; margin-bottom: 18px;'>
                <div style='display: flex; align-items: center; gap: 8px;'>
                    <div style='width: 24px; height: 24px; background: #2ecc40; border-radius: 4px;'></div>
                    <span style='font-size: 1.08em;'>Increase Price</span>
                </div>
                <div style='display: flex; align-items: center; gap: 8px;'>
                    <div style='width: 24px; height: 24px; background: #ff4136; border-radius: 4px;'></div>
                    <span style='font-size: 1.08em;'>Decrease Price</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        your_shap = explainer(input_df).values[0]
        feat_names = [col.replace('_',' ').title() for col in feature_cols]

        df_shap = pd.DataFrame({
            'feature': feat_names,
            'shap': your_shap
        })
        df_shap['abs_shap'] = np.abs(df_shap['shap'])
        df_shap = df_shap.sort_values(by='abs_shap', ascending=False)

        # Pick top 3 positive, top 2 negative
        top_pos = df_shap[df_shap['shap'] > 0].head(3)
        top_neg = df_shap[df_shap['shap'] < 0].head(2)

        up_str = ', '.join(top_pos['feature'])
        down_str = ', '.join(top_neg['feature'])

        explanation_positive = f"This house's high price is mostly due to: \n `{up_str}`."
        st.subheader("Insights") 
        st.markdown(f":bulb: {explanation_positive}")

        explanation_negative = f""
        if len(top_neg) > 0:
            explanation_negative += f" Having `{down_str}`   slightly lowered the predicted price."
        st.markdown(f":bulb: {explanation_negative}")

    with col2:
            labels = [f"{name}\n{perc:.1f}%" for name, perc in zip(feat_names, percs)]

            # Build a continuous red/green map by sign & magnitude
            norm = Normalize(vmin=-max(abs(sv)), vmax=+max(abs(sv)))
            cmap_pos = cm.get_cmap("Greens")
            cmap_neg = cm.get_cmap("Reds_r")
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

    with col3:
            st.subheader("Ranking of Features")
            counter = 1
            for name, pct in feat_perc:
                # compute bar width as a percentage of the full column
                width = (pct / 100) *200
                st.markdown(f"""
                <div style="display: flex; align-items: center; margin-bottom: 4px;">
                <!-- feature name -->
                <div style="flex: 1;  font-size: 1em;">{counter}. {name}</div>
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

    st.markdown("---")

    col1,col2 = st.columns([5,2],  gap="medium")
    with col1:  
        similars = df  # or your subset
        user_area = input_df.at[0, "area"]
        user_price = pred_price
        # 2. Set a modern style
        sns.set_theme(style="whitegrid")

        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(7, 4), facecolor='none')

        # Plot data
        sns.scatterplot(
            data=similars,
            x="area",
            y="price",
            ax=ax,
            s=60,
            color="#66b3ff",
            alpha=0.6,
            edgecolor="w",
            linewidth=0.5
        )

        # User's house
        ax.scatter(
            [user_area], [user_price],
            color="#e74c3c",
            s=180,
            marker='*',
            label="Your house",
            zorder=10,
            edgecolor='white',
            linewidth=1.3
        )

        # White text everywhere
        ax.set_xlabel("Area (sqft)", fontsize=12, weight='bold', color="white")
        ax.set_ylabel("Price", fontsize=12, weight='bold', color="white")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'${int(x):,}'))

        # Set tick colors to white
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')

        # Make all axes spines white or invisible
        for spine in ax.spines.values():
            spine.set_color('white')

        # Remove grid/background
        ax.set_facecolor('none')
        fig.patch.set_alpha(0.0)

        # Annotate with white text
        ax.annotate("Your house",
                    xy=(user_area, user_price),
                    xytext=(user_area+50, user_price+0.03*user_price),
                    arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=2),
                    fontsize=12, color="white", weight='bold'
        )

        st.pyplot(fig, transparent=True)
    with col2:
        # Suppose:
        area_min, area_max = df["area"].min(), df["area"].max()
        price_min, price_max = df["price"].min(), df["price"].max()
        user_area = input_df.at[0, "area"]
        user_price = pred_price

        # Get positions as % (0-100)
        area_pct = int(100 * (user_area - area_min) / (area_max - area_min))
        price_pct = int(100 * (user_price - price_min) / (price_max - price_min))

        slider_style = """
        <div style="margin-bottom:18px;">
        <div style='font-size:1.8em; margin-bottom:2px; color:white;'>Area: {user_area} sqft</div>
        <div style='position: relative; height: 18px; background: #222; border-radius: 8px;'>
            <div style='position: absolute; left: {area_pct}%; top: -6px;'>
            <div style='
                width: 24px; height: 24px;
                border-radius: 50%; background: #66b3ff;
                border: 3px solid white; box-shadow: 0 2px 8px #0006;
                display: flex; align-items: center; justify-content: center;
                font-size: 0.95em; color: white; font-weight: 600;'>
            </div>
            </div>
            <div style='position: absolute; left: 0; top: 20px; font-size:0.95em; color: #aaa;'>Min: {area_min}</div>
            <div style='position: absolute; right: 0; top: 20px; font-size:0.95em; color: #aaa;'>Max: {area_max}</div>
        </div>
        </div>
        <div style="margin-bottom:18px;">
        <div style='font-size:1.8em; margin-bottom:2px; color:white;'>Price: ${user_price:,.0f}</div>
        <div style='position: relative; height: 18px; background: #222; border-radius: 8px;'>
            <div style='position: absolute; left: {price_pct}%; top: -6px;'>
            <div style='
                width: 24px; height: 24px;
                border-radius: 50%; background: #e74c3c;
                border: 3px solid white; box-shadow: 0 2px 8px #0006;
                display: flex; align-items: center; justify-content: center;
                font-size: 0.95em; color: white; font-weight: 600;'>
            </div>
            </div>
            <div style='position: absolute; left: 0; top: 20px; font-size:0.95em; color: #aaa;'>Min: ${price_min:,.0f}</div>
            <div style='position: absolute; right: 0; top: 20px; font-size:0.95em; color: #aaa;'>Max: ${price_max:,.0f}</div>
        </div>
        </div>
        """.format(
            user_area=user_area,
            area_pct=area_pct,
            area_min=area_min,
            area_max=area_max,
            user_price=user_price,
            price_pct=price_pct,
            price_min=price_min,
            price_max=price_max
        )

        st.markdown("""
        <div style='
            display: flex;
            flex-direction: column;
            justify-content: center;
            min-height: 500px;    /* tweak to fit your layout */
        '>
        """ + slider_style + """
        </div>
        """, unsafe_allow_html=True)



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
