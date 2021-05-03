def visualize_confidence_level(prediction_proba):
    """
    this function uses matplotlib to create inference bar chart rendered with streamlit in real-time 
    return type : matplotlib bar chart  
    """
    st.subheader('Results')
    col1, col2, col3 = st.beta_columns([4, 1])
    col1.subheader("OffInvoice Discount")
    col1.write(offdisc)
    col2.subheader("OnInvoice Discount")
    col2.write(ondisc)
    col3.subheader("Total Discount")
    col3.write(offdisc+ondisc)
    return
st.write("""
# ML Web-App To Recommend Customized Discounts
This app recommends ** Customized Discounts ** to customers based on their business significance and performance using the following Input Features via the **side panel** 
""")

#read in wine image and render with streamlit
image = Image.open('abI.png')
st.image(image, use_column_width=True)

st.sidebar.header('User Input Parameters') #user input parameter collection with streamlit side bar


def get_user_input():
    """
    this function is used to get user input using sidebar slider and selectbox 
    return type : pandas dataframe

    """
    ship_to_ID = st.sidebar.number_input('Shipping ID of POC', min_value=0, step=1)
    Volume_2019=st.sidebar.number_input('Total Volume sold to the POC in 2019 (in Hectolitres)')
    Volume_2018=st.sidebar.number_input('Total Volume sold to the POC in 2018 (in Hectolitres)')
    sfdc_tier=st.sidebar.selectbox('Urbanicity of the POC',('Tier 0','Tier 1','Tier 2'))
    poc_image=st.sidebar.selectbox('Tier of the POC',('Mainstream','Premium'))
    segment=st.sidebar.selectbox('Segment of the POC',('Drink Led','Food Led','Entertainment Led','Institutional','Wholesaler','Not applicable'))
    sub_segment=st.sidebar.selectbox('Sub-Segment of the POC',('Bar','Hybrid','Beer bar','Restaurant','Institutional','Sports Venue','Party Place','Events','Recreational','Quick Dining','Not applicable'))
    brand=st.sidebar.selectbox('Brand',('BASS',"BECK'S",'BELLE VUE','BIRRA DEL BORGO','CORONA','CUBANISTO','DEUS','DIEKIRCH','GINDER-ALE','GINETTE','GOOSE ISLAND','HOEGAARDEN','HORSE ALE','JUPILER','KRUGER','KWAK','LEFFE','PIEDBOEUF','PURE BLONDE','SAFIR','SCOTCH CTS','STELLA ARTOIS','TRIPEL KARMELIET','VIEUX TEMPS'))
    sub_brand=st.sidebar.text_input('Sub-Brand')
    pack_type=st.sidebar.selectbox('Form of Container',('BOTTLE','KEG','CAN','PERFECTDRAFT','BULK'))
    returnalility=st.sidebar.selectbox('Returnability',('RETURNABLE','OW'))
    GTO_2019=st.sidebar.number_input('GTO 2019')
    
    features = {'Ship_to_ID': ship_to_ID,
            'Volume_2019': Volume_2019,
            'Volume_2018': Volume_2018,
            'Urbanicity': sfdc_tier,
            'Tier': poc_image, 
            'Segment': segment,
            'Sub-Segment': sub_segment,
            'Brand': brand,
            'Sub-Brand': sub_brand,
            'Container': pack_type,
            'Returnability': returnalility,
            'GTO_2019': GTO_2019
            }
    data = pd.DataFrame(features,index=[0])

    return data

user_input_df = get_user_input()
processed_user_input = data_preprocessor(user_input_df)

st.subheader('User Input parameters')
st.write(user_input_df)

prediction = model.predict(processed_user_input)
prediction_proba = model.predict_proba(processed_user_input)

visualize_confidence_level(prediction_proba)
