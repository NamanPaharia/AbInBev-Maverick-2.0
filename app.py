import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
from sklearn.utils import check_array
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from xgboost import XGBClassifier
#from lazypredict.Supervised import LazyRegressor
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
#import autosklearn.classification
import pickle
import gdown
import urllib.request
import streamlit as st
from PIL import Image

class Infer:

  def __init__(self,
               Ship_to_ID,Volume_2019,Volume_2018,sfdc_tier,poc_image,segment,sub_segment,Product_Set,Brand,Sub_Brand,Pack_Type,
               Returnalility,GTO_2019,Volume_2019_Product,Tax,province,poc="none"):
    data={'Ship-to ID':[Ship_to_ID],'Volume_2019':[Volume_2019],'Volume_2018':[Volume_2018],'sfdc_tier':[sfdc_tier],
          'poc_image':[poc_image],'segment':[segment],'sub_segment':[sub_segment],'Product Set':[Product_Set],'Brand':[Brand],
          'Sub-Brand':[Sub_Brand],'Pack_Type':[Pack_Type],'Returnalility':[Returnalility],'GTO_2019':[GTO_2019],
          'Volume_2019 Product':[Volume_2019_Product],'Tax':[Tax],'province':[province]}
    self.df=pd.DataFrame(data=data)
    self.ispoc=poc

    url = "https://www.dropbox.com/s/hks3pb01hc8jc0a/off_xgbreg.pickle.dat?dl=1"  # dl=1 is important
    urllib.request
    u = urllib.request.urlopen(url)
    data = u.read()
    u.close()
    with open('off_reg.pickle.dat', "wb") as f :
      f.write(data)

    self.regressor_off=pickle.load(open("off_reg.pickle.dat", "rb"))

    url = "https://www.dropbox.com/s/tjt29nwyok3v2x0/off_clf1.pickle.dat?dl=1"  # dl=1 is important
    urllib.request
    u = urllib.request.urlopen(url)
    data = u.read()
    u.close()
    with open('off_clf1.pickle.dat', "wb") as f :
      f.write(data)

    self.clf_off=pickle.load(open("off_clf1.pickle.dat", "rb"))

    url = "https://www.dropbox.com/s/lq4qeaz3k13qwax/off_clf2.pickle.dat?dl=1"  # dl=1 is important
    urllib.request
    u = urllib.request.urlopen(url)
    data = u.read()
    u.close()
    with open('off_clf2.pickle.dat', "wb") as f :
      f.write(data)

    self.clf2_off=pickle.load(open("off_clf2.pickle.dat", "rb"))

    url = "https://www.dropbox.com/s/ml4a9jilm2vcnuj/on_xgbreg.pickle.dat?dl=1"  # dl=1 is important
    urllib.request
    u = urllib.request.urlopen(url)
    data = u.read()
    u.close()
    with open('on_reg.pickle.dat', "wb") as f :
      f.write(data)

    self.regressor_on=pickle.load(open("on_reg.pickle.dat", "rb"))

    url = "https://www.dropbox.com/s/0aplcor3qhin2j1/on_clf1.pickle.dat?dl=1"  # dl=1 is important
    urllib.request
    u = urllib.request.urlopen(url)
    data = u.read()
    u.close()
    with open('on_clf1.pickle.dat', "wb") as f :
      f.write(data)

    self.clf_on=pickle.load(open("on_clf1.pickle.dat", "rb"))

    url = "https://www.dropbox.com/s/4amnhgx27b7jzgf/on_clf2.pickle.dat?dl=1"  # dl=1 is important
    urllib.request
    u = urllib.request.urlopen(url)
    data = u.read()
    u.close()
    with open('on_clf2.pickle.dat', "wb") as f :
      f.write(data)

    self.clf2_on=pickle.load(open("on_clf2.pickle.dat", "rb"))
   
    url = 'https://drive.google.com/uc?id=13celyxcevwAaKGcNXjJhHZAHMp0Yl_u8'
    output = 'data.xlsx'
    gdown.download(url, output, quiet=False)
    orig = pd.read_excel('data.xlsx',engine='openpyxl')
    orig=orig.drop(['OnInvoice Discount(LCU)','OffInvoice Discount(LCU)','Discount_Total'],axis=1)

    for i in range(len(orig)):
      if (abs(orig.loc[i,'Volume_2019 Product'])<0.0005):
        orig.loc[i,'Volume_2019 Product']=0
      
      if (orig.loc[i,'poc_image']==0):
        orig.loc[i,'poc_image']='Mainstream'

    orig.loc[9,'segment']='Institutional'
    orig=orig.iloc[0:-4]


    self.df=pd.concat([self.df, orig], axis=0).reset_index().drop(['index'],axis=1)

  def create_poc_dict(self):
    self.poc_disc=dict()
    self.poc_disc['BOTTLE']={'on':0.191348,'off':0.166302}
    self.poc_disc['KEG']={'on':0.164330,'off':0.150819}
    self.poc_disc['BULK']={'on':0.129303,'off':0.218504}
    self.poc_disc['CAN']={'on':0.184136,'off':0.168655} 

  def drop(self):
    drop_cols=['Ship-to ID','Product Set']
    self.df=self.df.drop(drop_cols,axis=1)

  def one_hot(self):
    oh_cols=['poc_image','segment','Brand','sub_segment','Pack_Type','Returnalility','province','Sub-Brand']
    self.df=pd.get_dummies(self.df,columns=oh_cols)

  def label_encode(self):
    self.le = preprocessing.LabelEncoder()
    self.le.fit(self.df.sfdc_tier)
    self.df.sfdc_tier=self.le.transform(self.df.sfdc_tier)

  def predict(self):
    self.drop()
    self.one_hot()
    self.label_encode()
    # self.df=self.df.iloc[0:1,:]

    self.cl_preds_off=self.clf_off.predict(self.df)
    self.cl2_preds_off=self.clf2_off.predict(self.df)
    self.preds_off=self.regressor_off.predict(self.df)

    self.preds_off=self.preds_off*self.cl_preds_off*self.cl2_preds_off

    self.cl_preds_on=self.clf_on.predict(self.df)
    self.cl2_preds_on=self.clf2_on.predict(self.df)
    self.preds_on=self.regressor_on.predict(self.df)

    self.preds_on=self.preds_on*self.cl_preds_on*self.cl2_preds_on

    offdisc=self.preds_off[0]
    ondisc=self.preds_on[0]

    return [offdisc,ondisc]

  def poc_disc(self,poc):
    self.create_poc_dict()
    return self.poc_disc[poc]

  def result(self):
    if (self.ispoc=="none"):
      disc=self.predict()
      return disc

def visualize_confidence_level(prediction_proba):
    """
    this function uses matplotlib to create inference bar chart rendered with streamlit in real-time 
    return type : matplotlib bar chart  
    """
    st.subheader('Results')
    col1, col2, col3 = st.beta_columns(3)
    col1.subheader("OffInvoice Discount")
    col1.write(prediction_proba[0])
    col2.subheader("OnInvoice Discount")
    col2.write(prediction_proba[1])
    col3.subheader("Total Discount")
    col3.write(prediction_proba[0]+prediction_proba[1])
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
    sub_brand=st.sidebar.selectbox('Sub-Brand',('BASS PALE ALE',"BECK'S REGULAR",'BELLE VUE EXTRA KRIEK','BELLE VUE GUEUZE','BELLE VUE KRIEK CLASSIQUE','BIRRA DEL BORGO CASTAGNALE','CORONA EXTRA','CUBANISTO PHENOMENAL','CUBANISTO RUM','DEUS','DIEKIRCH BRUIN','DIEKIRCH GRAND CRU','DIEKIRCH PILS','DIEKIRCH XMAS BEER','FLAVOURED ALCOHOLIC','GINDER-ALE',"GOOSE ISLAND HONKER'S ALE",'GINETTE BLANCHE','GINETTE BLONDE','GINETTE FRUIT','GINETTE TRIPEL','GINETTE LAGER','GOOS 312','GOOSE ISLAND IPA','GOOSE ISLAND MIDWAY IPA','HOEGAARDEN 0,0','HOEG RADLER LEMON 0,0','HOEGAARDEN BEATRIX','HOEGAARDEN FORBIDDEN FRUIT','HOEGAARDEN GRAND CRU','HOEGAARDEN JULIUS','HOEGAARDEN RADLER AGRUM 0,0','HOEGAARDEN ROSE 0,0','HOEGAARDEN ROSEE','HOEGAARDEN WHITE','HOEGAARDEN YELLOW','HORSE ALE','JUPILER 0,0','JUPILER BLUE','JUPILER PILS','KRUGER EXPORT','KWAK','LEFF NECT','LEFFE','LEFFE BLONDE','LEFFE BRUNE','LEFFE MIXED','LEFFE RADIEUSE','LEFFE RITUEL 9','LEFFE ROYALE','LEFFE ROYALE CASCADE IPA','LEFFE ROYALE IPA','LEFFE ROYALE MAPUCHE','LEFFE RUBY','LEFFE SANS ALCOOL/ALCOHOLVRIJ','LEFFE TRIPLE','PIEDBOEUF BLONDE','PIEDBOEUF FONCEE','PIEDBOEUF TRIPLE','PURE BLONDE REGULAR','SAFIR REGULAR','SCOTCH CTS','STELLA ARTOIS REGULAR','TRIPEL KARMELIET','VIEUX TEMPS REGULAR'))
    pack_type=st.sidebar.selectbox('Form of Container',('BOTTLE','KEG','CAN','PERFECTDRAFT','BULK'))
    returnalility=st.sidebar.selectbox('Returnability',('RETURNABLE','OW'))
    volume_2019_product=st.sidebar.number_input('Volume 2019 Product')
    GTO_2019=st.sidebar.number_input('GTO 2019')
    tax=st.sidebar.number_input('Tax')
    province=st.sidebar.selectbox('Province',('West Flanders','Brussels Capital','Liège','Flemish Brabant','East Flanders','Hainaut','Antwerp','Limburg','Namur','Walloon Brabant'))
    
    features = {'Ship_to_ID': ship_to_ID,
            'Volume_2019': Volume_2019,
            'Volume_2018': Volume_2018,
            'Urbanicity': sfdc_tier,
            'Tier': poc_image, 
            'Segment': segment,
            'Sub-Segment': sub_segment,
            'Product Set': returnalility+"_"+pack_type+"_"+brand+"_"+sub_brand,
            'Brand': brand,
            'Sub-Brand': sub_brand,
            'Container': pack_type,
            'Returnability': returnalility,
            'GTO_2019': GTO_2019,
            'Volume 2019 Product': volume_2019_product,
            'Tax': tax,
            'Province': province
            }
    data = pd.DataFrame(features,index=[0])

    return data

data = get_user_input()
model=Infer(data.iloc[0,0],data.iloc[0,1],data.iloc[0,2],data.iloc[0,3],data.iloc[0,4],data.iloc[0,5],data.iloc[0,6],
data.iloc[0,7],data.iloc[0,8],data.iloc[0,9],data.iloc[0,10],data.iloc[0,11],data.iloc[0,12],data.iloc[0,13],
data.iloc[0,14],data.iloc[0,15])

st.subheader('User Input parameters')
st.write(data)

prediction = model.predict()
visualize_confidence_level(prediction)
