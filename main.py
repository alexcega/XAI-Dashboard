import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
import matplotlib.ticker as ticker
import numpy as np
import squarify  
import random
import os
import seaborn as sns
import streamlit.components.v1 as components
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
@st.cache_resource
def load_model():
    # Model was trained on ordinal-encoded features
    return joblib.load("house_price_rf.pkl")

@st.cache_resource
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


# Sidebar: User Persona
st.sidebar.title("Housing Price Tool")
persona = st.sidebar.selectbox(
    "I want to …", 
    ["See my house", "Understand the prediction", "Understand the data", "Understand the model"]
)

# Sidebar: Property features Input
st.sidebar.header("Property Features")
area = st.sidebar.slider("Area (sqft)", min_value=area_min, max_value=area_max ,value=area_max // 2)

# Sidebar slider uses the session state value, and updates session state
# st.session_state.area = st.sidebar.slider(
#     "Area (sqft)",
#     min_value=area_min,
#     max_value=area_max,
#     value=st.session_state.area,
#     key="area_slider"
# )

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

col1, col2  = st.columns([1,1], gap="medium")
with col1:
    st.markdown("**Welcome to the XAI dashboard!**")
    st.markdown("Change the data on the side bar and understand the result on the different module here")
with col2:
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
with st.expander("🏡 For Buyers", expanded=(persona == "See my house")):
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

        st.markdown("---")
        st.markdown("### What are the least important features?")
        for name, percentage in feat_perc:
            if percentage < 3:
                st.write(name, "{:.2f}".format(percentage), "%")

    st.markdown("---")
    st.subheader("Understanding the market")
    # Persona-specific insights
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
        
        
        area_median = df["area"].mean()
        price_median = df["price"].mean()
        if user_area >= area_median:
            area_txt = f"This house is among the top {100 - area_pct:.1f}% largest of properties on the market."
        else:
            area_txt = f"This house is among the top {area_pct:.1f}% smallest of properties on the market."

        if user_price >= price_median:
            price_txt = f"This house is one of the top {100 - price_pct:.1f}% most expensive in the market"
        else:
            price_txt = f"This house is one of the top ({price_pct:.1f}% cheapest in the market."

        st.markdown(f"""
        <div style="font-size:1.0em; ">
            🏠 {area_txt}<br>
            💰 {price_txt}
        </div>
        """, unsafe_allow_html=True)

        slider_style = """
        <div style="margin-bottom:18px;">
        <div style='font-size:1.2em; margin-bottom:2px; color:white;'>Area: {user_area} sqft</div>
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
        <div style='font-size:1.2em; margin-bottom:2px; color:white;'>Price: ${user_price:,.0f}</div>
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
            min-height: 200px;    /* tweak to fit your layout */
        '>
        """ + slider_style + """
        </div>
        """, unsafe_allow_html=True)

#############################################################


#############################################################
with st.expander("☝️🤓 Understanding the prediction", expanded=(persona == "Understand the prediction")):
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
            st.markdown("### Impact by size")
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
#########################################

#########################################
with st.expander("🔢 Understanding the data", expanded=(persona == "Understand the data")):
    @st.cache_resource

    def plot_feature_distribution(df, feature, value):
        # Set modern whitegrid theme
        sns.set_theme(style="whitegrid")

        fig, ax = plt.subplots(figsize=(6, 3.5), facecolor='none')

        # Plot histogram
        sns.histplot(df[feature], bins=30, color="#3891A6", alpha=0.8, ax=ax)
        
        # Vertical line for the selected value
        ax.axvline(value, color="#e74c3c", linestyle="--", lw=3)

        # Set titles and labels
        ax.set_title(f"{feature.title()} Distribution", fontsize=16, weight='bold', color="white")
        ax.set_xlabel(feature.title(), fontsize=12, weight='bold', color="white")
        ax.set_ylabel("Count", fontsize=12, weight='bold', color="white")

        # Set tick colors to white
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')

        # Make all axes spines white
        for spine in ax.spines.values():
            spine.set_color('white')

        # Remove grid and set background transparent
        ax.set_facecolor('none')
        fig.patch.set_alpha(0.0)

        # Tight layout
        fig.tight_layout()

        # Plot in Streamlit
        st.pyplot(fig, transparent=True)

    col1, col2= st.columns([2,3], gap="medium")
    with col1:
        row1, row2 = st.columns([1,1])
        plot_feature_distribution(df, "area",   input_df.at[0, "area"])
        st.markdown("---")

        plot_feature_distribution(df, "price", pred_price)
    with col2:
        total_houses = df["area"].count()
        st.markdown(
        f"""
        <div style='font-size: 20px; color: #white; text-align: right;'>
            Total houses 
        </div>
        <div style='
            font-size: 50px; 
            font-weight: bold; 
            color: #white; 
            text-align: right;
            margin-bottom: 12px;
        '>
            {total_houses}
        </div>
        """, 
        unsafe_allow_html=True
        )
        # Efect of var on price
        cat_cols = [
            "mainroad",
            "guestroom",
            "basement",
            "hotwaterheating",
            "airconditioning",
            "prefarea",
            "furnishingstatus",
        ]

        non_cat_feature_cols = [col for col in feature_cols if col not in cat_cols]

        feature_to_vary = st.selectbox("Explore effect of...", non_cat_feature_cols)

        vals = np.linspace(df[feature_to_vary].min(), df[feature_to_vary].max(), 40)
        input_copy = input_df.copy()
        preds = []

        for v in vals:
            input_copy.at[0, feature_to_vary] = v
            preds.append(model.predict(input_copy)[0])

        # Set theme
        sns.set_theme(style="whitegrid")

        fig, ax = plt.subplots(figsize=(6, 3), facecolor='none')

        # Plot line
        ax.plot(vals, preds, color="#cb4b16", lw=3)

        # Set labels and title with white font
        ax.set_xlabel(feature_to_vary.title(), fontsize=12, weight='bold', color="white")
        ax.set_ylabel("Predicted Price", fontsize=12, weight='bold', color="white")
        ax.set_title(f"Effect of {feature_to_vary.title()} on Price", fontsize=12, weight='bold', color="white")

        # Set tick colors to white
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')

        # Make all axes spines white
        for spine in ax.spines.values():
            spine.set_color('white')

        # Set transparent backgrounds
        ax.set_facecolor('none')
        fig.patch.set_alpha(0.0)

        # Optional: if you want to remove grid lines completely
        # ax.grid(False)

        fig.tight_layout()
        st.pyplot(fig, transparent=True)


####################################################

####################################################
with st.expander("🤖 Understanding the model", expanded=(persona == "Understand the model")):
    st.header("Global Feature Importance")
    feature_cols = ['area','bedrooms','bathrooms','stories','mainroad','guestroom','basement','hotwaterheating','airconditioning','parking','prefarea','furnishingstatus']
    @st.cache_data
    def load_train_features():
        return pd.read_csv("train_features.csv")

    X_train = load_train_features()[feature_cols]
    shap_values_full = explainer(X_train)

    # Sort the global importance
    glob_imp = pd.Series(np.abs(shap_values_full.values).mean(axis=0), index=feature_cols).sort_values(ascending=False)
    glob_imp_sorted = dict(sorted(glob_imp.items(), key=lambda item: item[1], reverse=True))

    # Top N and Bottom N
    top_n = 5
    bottom_n = 5

    top_features = list(glob_imp_sorted.items())[:top_n]
    bottom_features = list(glob_imp_sorted.items())[-bottom_n:]

    col1, col2= st.columns([3,1], gap="medium")

    with col2: 
        # Sort the global importance
        glob_imp_sorted = dict(sorted(glob_imp.items(), key=lambda item: item[1], reverse=True))

        # Top N and Bottom N
        top_n = 5
        bottom_n = 5

        top_features = list(glob_imp_sorted.items())[:top_n]
        bottom_features = list(glob_imp_sorted.items())[-bottom_n:]

        markdown_html = f"""
        <div style='width: 100%;'>
            <div style='font-size:1.3em; font-weight:700; margin-bottom:10px; color:white'>
                🔥 Most Important Features Overall
            </div>
            <div style='display: grid; grid-template-columns: 1fr auto; gap: 10px; font-size:1em; line-height:1.8;'>
        """

        for i, (name, val) in enumerate(top_features, 1):
            val_formatted = f"{int(val):,}"
            markdown_html += f"""
            <div style=color:white><b>{i}° {name}</b></div>
            <div style='text-align: right; color:white'> ${val_formatted}</div>
            """

        markdown_html += """
            </div>
            <div style='font-size:1.3em; font-weight:700; margin:30px 0 10px; color:white'>
                🧊 Least Important Features Overall
            </div>
            <div style='display: grid; grid-template-columns: 1fr auto; gap: 10px; font-size:1em; line-height:1.8;'>
        """

        for i, (name, val) in enumerate(bottom_features, 1):
            val_formatted = f"{int(val):,}"
            markdown_html += f"""
            <div style=color:white><b>{i}° {name}</b></div>
            <div style='text-align: right; color:white'> ${val_formatted}</div>
            """

        markdown_html += """
            </div>
        </div>
        """
        components.html(markdown_html, height=600) 


    with col1:
        n = len(glob_imp)
        cmap = cm.get_cmap('viridis', n)
        max_height = glob_imp.values.max()

        bar_gap = "1%"  # use percentage gap so it scales better
        chart_height = 320

        bars_html = f"<div style='display: flex; align-items: flex-end; height: {chart_height}px; gap: {bar_gap};'>"
        for i, (name, val) in enumerate(glob_imp.items()):
            color = mcolors.to_hex(cmap(i / (n-1) if n > 1 else 0))
            max_height = glob_imp.values.max()
            height_px = max(int((val / max_height) * (chart_height - 60)), 20)        
            
            bars_html += f"""
            <div style='flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: flex-end;'>
                <div style='
                    font-size: 1.08em; 
                    margin-bottom: 4px; 
                    font-weight: 600; 
                    height: 1.5em;
                '>{i+1}°</div>
                <div style='
                    height: {height_px}px;
                    background: {color};
                    border-radius: 9px 9px 0 0;
                    box-shadow: 0 2px 8px #0002;
                    margin-bottom: 0;
                    display: flex;
                    width: 80%; /* make bar thinner inside its flex box */
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
                '>{name}</div>
            </div>"""
        labels_html += "</div>"

        st.markdown(f"""
        <div style='width: 100%;'>
        <div style='font-size:1.3em; font-weight:700; margin-bottom:20px;'>
            Ranking of global features
        </div>
        {bars_html}
        {labels_html}
        </div>
        """, unsafe_allow_html=True)

