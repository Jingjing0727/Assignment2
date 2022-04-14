import joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import newmodel


##################page_setting###########################
st.set_page_config(page_title="My Streamlit App", layout="wide")
st.markdown('''
<style>
    #MainMenu
    {
        display: none;
    }
    .css-18e3th9, .css-1d391kg
    {
        padding: 1rem 2rem 2rem 2rem;
    }
</style>
''', unsafe_allow_html=True)

################## Page Header ##################################
st.header("Predicting which people could get more than 50K salary.")
st.write("Our application uses Artificial Intelligence to predict which factors affect people's income >50K.")
st.markdown('---')

st.markdown("""
 Emojis: :+1: :sunglasses:
From: [Github](https://github.com/Jingjing0727/Assignment2)""")
st.write('This Github url have all my file about this assignment2.')
################## Sidebar Menu #################################
page_selected = st.sidebar.radio("Menu", ["Home", "Predict","Model"])
################## Home Page ###################################
################################################################
if page_selected == "Home":  
  st.write('This training dataset and testing data come from the public dataset. We want to know if we want to get income equal to or more than 50k, we have to meet what conditions.')
  st.write('We use the original dataset to train.')
  st.write("You can free choose the factors' range or values. Then you can see you choose factors how to affect income. ")
    
    #######age to income >50K###################################
  oda = st.checkbox('Training Dataset')
  if oda:
    st.write("You see the Training dataset factors how affect income.")

    df = pd.read_csv('train1new.csv')
    ages= df['age'].unique().tolist()
    education = df['educational-num'].unique().tolist()
    whours= df['hours-per-week'].unique().tolist()
    sex= df['gender'].unique().tolist()
    country = df['native-country'].unique().tolist()
    income =df['income_>50K'].unique().tolist()

    Choice = st.radio("The impact of different conditions on income.",('Age','Educational_Level','Work_hours_per_work','Country','Gender')) 

    if Choice =='Age':

      st.title("The effect of different age ranges on income.")
      age_selection = st.slider('Age:',
                              min_value =min(ages),
                              max_value = max(ages),
                              value = (min(ages),max(ages)))
      st.write('You selected:', age_selection)

      mask =(df['age'].between(*age_selection))
      df_grouped =df[mask].groupby(by=['income_>50K']).count()[['age']]
      df_grouped = df_grouped.rename(columns={'age':'count'})
      df_grouped = df_grouped.reset_index()

      import plotly.express as px
      bar_chart = px.bar(df_grouped,
                       x='income_>50K',
                       y='count',
                       text='count',
                       color='income_>50K',
                       template='plotly_white')
      st.plotly_chart(bar_chart)

#####education #######################
    elif Choice=='Educational_Level':
       st.title("The effect of different Educational Level ranges on income.")
       edulevel_selection = st.slider('Educational Level:',
                                   min_value = min(education),
                                   max_value = max(education),
                                   value = (min(education),max(education)))
       st.write('You selected:', edulevel_selection)

       mask2 =(df['educational-num'].between(*edulevel_selection))
       df_grouped2 = df[mask2].groupby(by=['income_>50K']).count()[['educational-num']]
       df_grouped2 = df_grouped2.rename(columns={'educational-num':'count'})
       df_grouped2 = df_grouped2.reset_index()
      
       st.write('You can free choose the Educational level range. Then you can see want this different educational level range effect income.')
       bar_chart2 = px.bar(df_grouped2,
                       x='income_>50K',
                       y='count',
                       text='count',
                       color='income_>50K',
                       template='plotly_white')
       st.plotly_chart(bar_chart2)

########Work hours per week####################################################
    elif Choice=='Work_hours_per_work':
      st.title("The effect of Work hours per week on income.")
      whours_selection = st.slider('Work hours per week:',
                                 min_value = min(whours),
                                 max_value = max(whours),
                                 value = (min(whours),max(whours)))
      st.write('You selected:', whours_selection)
      mask3 =(df['hours-per-week'].between(*whours_selection))
      df_grouped3=df[mask3].groupby(by=['income_>50K']).count()[['hours-per-week']]
      df_grouped3 = df_grouped3.rename(columns={'hours-per-week':'count'})
      df_grouped3 = df_grouped3.reset_index()

      bar_chart3 = px.bar(df_grouped3,
                       x='income_>50K',
                       y='count',
                       text='count',
                       color='income_>50K',
                       template='plotly_white')
      st.plotly_chart(bar_chart3)
##########Native Country###############################################
    elif Choice=='Country':
       st.title("The effect of differen countries on income.")
       country_selection = st.selectbox('Native Country:',
                                       country)
       mask4 = df[(df['native-country']==country_selection)]
       df_grouped4=mask4.groupby(by=['income_>50K']).count()[['native-country']]
       df_grouped4 = df_grouped4.rename(columns={'native-country':'count'})
       df_grouped4 = df_grouped4.reset_index()
       st.write('You selected:', country_selection)
       st.write('')
       bar_chart4 = px.bar(df_grouped4,
                       x='income_>50K',
                       y='count',
                       text='count',
                       color='count',
                       template='plotly_white')
       st.plotly_chart(bar_chart4)
############Gender#################################################################
    else:
        st.title('The effect of Gender on income.')
        sex_selection = st.selectbox('Gender:',sex)
        mask5 = df[(df['gender']==sex_selection )]
        df_grouped5=mask5.groupby(by=['income_>50K']).count()[['gender']]
        df_grouped5 = df_grouped5.rename(columns={'gender':'count'})
        df_grouped5 = df_grouped5.reset_index()
        st.write('You selected:', sex_selection)
        st.write('')
        bar_chart5 = px.bar(df_grouped5,
                       x='income_>50K',
                       y='count',
                       text='count',
                       color='count',
                       template='plotly_white')
        st.plotly_chart(bar_chart5)
  st.write('We use this module to find which factors affect the income to 50k.')
  st.write('Now we use test1.csv to predict which factors affect the income can get equal to or more than 50K.')
  pda = st.checkbox('Predict Dataset')
  if pda:
    st.write("You can see we use pipeline to predict the 'income_>50K' values")
    df = pd.read_csv('test1.csv')
    pipeline = joblib.load('pipeline.pkl')
    pf = pipeline.predict(df)
    df["income_>50K"]=pf
    st.write("")

    ages= df['age'].unique().tolist()
    education = df['educational-num'].unique().tolist()
    whours= df['hours-per-week'].unique().tolist()
    sex= df['gender'].unique().tolist()
    country = df['native-country'].unique().tolist()
    income =df['income_>50K'].unique().tolist()

    st.write('')
    Choice = st.selectbox(
     'The impact of different conditions on income.',
     ('Age','Educational_Level','Work_hours_per_work','Country','Gender'))
    if Choice =='Age':

      st.title("The effect of different age ranges on income.")
      age_selection = st.slider('Age:',
                              min_value =min(ages),
                              max_value = max(ages),
                              value = (min(ages),max(ages)))
      st.write('You selected:', age_selection)

      mask =(df['age'].between(*age_selection))
      df_grouped =df[mask].groupby(by=['income_>50K']).count()[['age']]
      df_grouped = df_grouped.rename(columns={'age':'count'})
      df_grouped = df_grouped.reset_index()

    
      bar_chart = px.bar(df_grouped,
                       x='income_>50K',
                       y='count',
                       text='count',
                       color='income_>50K',
                       template='plotly_white')
      st.plotly_chart(bar_chart)

#####education #######################
    elif Choice=='Educational_Level':
       st.title("The effect of different Educational Level ranges on income.")
       edulevel_selection = st.slider('Educational Level:',
                                   min_value = min(education),
                                   max_value = max(education),
                                   value = (min(education),max(education)))
       st.write('You selected:', edulevel_selection)

       mask2 =(df['educational-num'].between(*edulevel_selection))
       df_grouped2 = df[mask2].groupby(by=['income_>50K']).count()[['educational-num']]
       df_grouped2 = df_grouped2.rename(columns={'educational-num':'count'})
       df_grouped2 = df_grouped2.reset_index()
      
       st.write('You can free choose the Educational level range. Then you can see want this different educational level range effect income.')
       bar_chart2 = px.bar(df_grouped2,
                       x='income_>50K',
                       y='count',
                       text='count',
                       color='income_>50K',
                       template='plotly_white')
       st.plotly_chart(bar_chart2)

########Work hours per week####################################################
    elif Choice=='Work_hours_per_work':
      st.title("The effect of Work hours per week on income.")
      whours_selection = st.slider('Work hours per week:',
                                 min_value = min(whours),
                                 max_value = max(whours),
                                 value = (min(whours),max(whours)))
      st.write('You selected:', whours_selection)
      mask3 =(df['hours-per-week'].between(*whours_selection))
      df_grouped3=df[mask3].groupby(by=['income_>50K']).count()[['hours-per-week']]
      df_grouped3 = df_grouped3.rename(columns={'hours-per-week':'count'})
      df_grouped3 = df_grouped3.reset_index()

      bar_chart3 = px.bar(df_grouped3,
                       x='income_>50K',
                       y='count',
                       text='count',
                       color='income_>50K',
                       template='plotly_white')
      st.plotly_chart(bar_chart3)
##########Native Country###############################################
    elif Choice=='Country':
       st.title("The effect of differen countries on income.")
       country_selection = st.selectbox('Native Country:',
                                       country)
       mask4 = df[(df['native-country']==country_selection)]
       df_grouped4=mask4.groupby(by=['income_>50K']).count()[['native-country']]
       df_grouped4 = df_grouped4.rename(columns={'native-country':'count'})
       df_grouped4 = df_grouped4.reset_index()
       st.write('You selected:', country_selection)
       st.write('')
       bar_chart4 = px.bar(df_grouped4,
                       x='income_>50K',
                       y='count',
                       text='count',
                       color='count',
                       template='plotly_white')
       st.plotly_chart(bar_chart4)
############Gender#################################################################
    else:
        st.title('The effect of Gender on income.')
        sex_selection = st.selectbox('Gender:',sex)
        mask5 = df[(df['gender']==sex_selection )]
        df_grouped5=mask5.groupby(by=['income_>50K']).count()[['gender']]
        df_grouped5 = df_grouped5.rename(columns={'gender':'count'})
        df_grouped5 = df_grouped5.reset_index()
        st.write('You selected:', sex_selection)
        st.write('')
        bar_chart5 = px.bar(df_grouped5,
                       x='income_>50K',
                       y='count',
                       text='count',
                       color='count',
                       template='plotly_white')
        st.plotly_chart(bar_chart5)

  st.write("You can upload your want to predict CSV file. We can use this app to help you predict. Or you can go to the web page 'predict' to predict if your income can get equal to or more than 50K. ")
  uploaded_file = st.file_uploader("Choose a file")
  if uploaded_file is not None:
    import joblib
    df= pd.read_csv(uploaded_file)
    pipeline = joblib.load('pipeline.pkl')
    pf = pipeline.predict(df)
    df["income_>50K"]=pf
    st.write("This is your cvs file predict result.")
    df_c = st.checkbox('Predict Result')
    if df_c:
        df

    ages= df['age'].unique().tolist()
    education = df['educational-num'].unique().tolist()
    whours= df['hours-per-week'].unique().tolist()
    sex= df['gender'].unique().tolist()
    country = df['native-country'].unique().tolist()
    income =df['income_>50K'].unique().tolist()

    Choice = st.selectbox(
     'The impact of different conditions on income.',
     ('Age','Educational_Level','Work_hours_per_work','Country','Gender'))
    if Choice =='Age':

      st.title("The effect of different age ranges on income.")
      age_selection = st.slider('Age:',
                              min_value =min(ages),
                              max_value = max(ages),
                              value = (min(ages),max(ages)))
      st.write('You selected:', age_selection)

      mask =(df['age'].between(*age_selection))
      df_grouped =df[mask].groupby(by=['income_>50K']).count()[['age']]
      df_grouped = df_grouped.rename(columns={'age':'count'})
      df_grouped = df_grouped.reset_index()

    
      bar_chart = px.bar(df_grouped,
                       x='income_>50K',
                       y='count',
                       text='count',
                       color='income_>50K',
                       template='plotly_white')
      st.plotly_chart(bar_chart)

#####education #######################
    elif Choice=='Educational_Level':
       st.title("The effect of different Educational Level ranges on income.")
       edulevel_selection = st.slider('Educational Level:',
                                   min_value = min(education),
                                   max_value = max(education),
                                   value = (min(education),max(education)))
       st.write('You selected:', edulevel_selection)

       mask2 =(df['educational-num'].between(*edulevel_selection))
       df_grouped2 = df[mask2].groupby(by=['income_>50K']).count()[['educational-num']]
       df_grouped2 = df_grouped2.rename(columns={'educational-num':'count'})
       df_grouped2 = df_grouped2.reset_index()
      
       st.write('You can free choose the Educational level range. Then you can see want this different educational level range effect income.')
       bar_chart2 = px.bar(df_grouped2,
                       x='income_>50K',
                       y='count',
                       text='count',
                       color='income_>50K',
                       template='plotly_white')
       st.plotly_chart(bar_chart2)

########Work hours per week####################################################
    elif Choice=='Work_hours_per_work':
      st.title("The effect of Work hours per week on income.")
      whours_selection = st.slider('Work hours per week:',
                                 min_value = min(whours),
                                 max_value = max(whours),
                                 value = (min(whours),max(whours)))
      st.write('You selected:', whours_selection)
      mask3 =(df['hours-per-week'].between(*whours_selection))
      df_grouped3=df[mask3].groupby(by=['income_>50K']).count()[['hours-per-week']]
      df_grouped3 = df_grouped3.rename(columns={'hours-per-week':'count'})
      df_grouped3 = df_grouped3.reset_index()

      bar_chart3 = px.bar(df_grouped3,
                       x='income_>50K',
                       y='count',
                       text='count',
                       color='income_>50K',
                       template='plotly_white')
      st.plotly_chart(bar_chart3)
##########Native Country###############################################
    elif Choice=='Country':
       st.title("The effect of differen countries on income.")
       country_selection = st.selectbox('Native Country:',
                                       country)
       mask4 = df[(df['native-country']==country_selection)]
       df_grouped4=mask4.groupby(by=['income_>50K']).count()[['native-country']]
       df_grouped4 = df_grouped4.rename(columns={'native-country':'count'})
       df_grouped4 = df_grouped4.reset_index()
       st.write('You selected:', country_selection)
       st.write('')
       bar_chart4 = px.bar(df_grouped4,
                       x='income_>50K',
                       y='count',
                       text='count',
                       color='count',
                       template='plotly_white')
       st.plotly_chart(bar_chart4)
############Gender#################################################################
    else:
        st.title('The effect of Gender on income.')
        sex_selection = st.selectbox('Gender:',sex)
        mask5 = df[(df['gender']==sex_selection )]
        df_grouped5=mask5.groupby(by=['income_>50K']).count()[['gender']]
        df_grouped5 = df_grouped5.rename(columns={'gender':'count'})
        df_grouped5 = df_grouped5.reset_index()
        st.write('You selected:', sex_selection)
        st.write('')
        bar_chart5 = px.bar(df_grouped5,
                       x='income_>50K',
                       y='count',
                       text='count',
                       color='count',
                       template='plotly_white')
        st.plotly_chart(bar_chart5)

  st.write("This page just describes this app can do what. If you want to know more about how to build this model and pipeline, you can go to the web page 'Model'. The 'Model' more clearly describes the model and pipeline how to build.")
  st.write('Country and Educational levels have corresponding numbers. This is the corresponding numbers table.') 

  data = st.checkbox('Corresponding Numbers table')
  if data:
   df_n = pd.read_csv('transfer_data.csv')
   df_n
##############################Predict##################################

if page_selected =="Predict":
    st.write('Country and Educational levels have corresponding numbers. This is the corresponding numbers table.') 
    data = st.checkbox('Corresponding Numbers table')
    df_n = pd.read_csv('transfer_data.csv')
    df_n

    df=pd.read_csv('train1.csv')
    ages= df['age'].unique().tolist()
    education = df['educational-num'].unique().tolist()
    whours= df['hours-per-week'].unique().tolist()
    sex= df['gender'].unique().tolist()
    country = df['native-country'].unique().tolist()
    income =df['income_>50K'].unique().tolist()
    Age=st.text_input('Age:',ages)
    Education = st.text_input('Educational Level:',education)
    gender = st.text_input('Gender:',sex)
    Whours = st.text_input('Work hours per week:',whours)
    Country = st.text_input('Country:',country)
    df = pd.DataFrame([{'age':Age,'educational-num':Education,'gender':gender,'hours-per-week':Whours,'native-country':Country}])
    pipeline = joblib.load('pipeline.pkl')
    predictions = pipeline.predict(df)
    #return predictions or outcomes
    prf=np.average(predictions)
    if prf == 1:
        st.info('You can get income_> 50K!!!')
        st.balloons()
    else:
        st.info('You can not get income_> 50K.')
    

##############################Model####################################
if page_selected == "Model":
  st.write("This is my original data. You can see native_country has many different names in train.csv. 'Show train dataframe has a training dataset and some bar chart. You can use it to know this data factors type.  ")

  if st.checkbox('Show train dataframe'):
    st.write('This dataset data is clean.Now this dataset not missing value.')
    df =pd.read_csv('train1new.csv')
    df
    st.write('We can see this dataset values most is a categorical value. Country and gender values need we transform into the number. Then we need to check which different factors make that affect income. We can use these charts to find some results.')
    
    if st.checkbox('Gender'):
       df = pd.read_csv('train1new.csv')
       df_g1 =df.groupby(by=['gender']).count()[['income_>50K']]
       df_g1 = df_g1.rename(columns={'income_>50K':'count'})
       df_g1 = df_g1.reset_index()
       st.write('')
       bar_1 = px.bar(df_g1,
                   x ='gender',
                   y ='count',
                   color='gender',
                   template='plotly_white')
       st.plotly_chart(bar_1)
    
    if st.checkbox('Age'):
        df = pd.read_csv('train1new.csv')
        df_g2 =df.groupby(by=['age']).count()[['income_>50K']]
        df_g2 = df_g2.rename(columns={'income_>50K':'count'}) 
        df_g2 = df_g2.reset_index()
        bar_2 = px.bar(df_g2,
                   x ='age',
                   y ='count',
                   color='age',
                   template='plotly_white')
        st.plotly_chart(bar_2)
    
    if st.checkbox('Educational Level'):
        df = pd.read_csv('train1new.csv')
        df_g3 =df.groupby(by=['educational-num']).count()[['income_>50K']]
        df_g3 = df_g3.rename(columns={'income_>50K':'count'}) 
        df_g3 = df_g3.reset_index()
        bar_3 = px.bar(df_g3,
                   x ='educational-num',
                   y ='count',
                   text = 'count',
                   color='educational-num',
                   template='plotly_white')
        st.plotly_chart(bar_3)
  st.write('Then we can which different factors affect income. First We see that factors have how many unique values in this dataset.')
  df =pd.read_csv('train1new.csv')
  table_df=df.nunique()
  table_df

  st.write('We see how many unique values the different factors have, and then we see correlations between the factors for this training dataset. ')
  df_corr=pd.read_csv('train1.csv')
  df_corrt=df_corr.corr()
  st.table(df_corrt)
  fig=plt.figure(figsize=(10,8),dpi=100)
  sns.heatmap(df_corr.corr(),cmap="viridis",annot=True,linewidth=0.5)
  st.pyplot(fig)
  st.write("From the above results, we know that most of the data are categorical data in this dataset. ")
  st.write("When I build models, I choose algorithms that are basically classification-based. For example: RandomForestClassifier, DecisionTreeClassifier, KNeighborsClassifier. Because of the 'age', I added an additional LogisticRegression. Among the four, RandomForestClassifier has the highest prediction. In the end, I chose RandomForestClassifier to build the model and pipeline.")
  
  if st.button('model code'):
    with st.echo():
        import pandas as pd
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import confusion_matrix as cm
        from sklearn.pipeline import Pipeline
        import joblib
        import matplotlib.pyplot as plt
        #data
        df_train = "train1.csv"
        df_test = "test1.csv"
        #X,y
        def load_prepare():
            df = pd.read_csv(df_train)
            X = df.drop(columns=['income_>50K'])
            y = df['income_>50K']
            return X, y
        #final model:
        def build_pipeline_final(X, y):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
            pipeline = Pipeline([('sc', StandardScaler()),
            ('RFC', RandomForestClassifier(max_depth = 6,random_state=300))])
            pipeline.fit(X_train, y_train)
            y_predict = pipeline.predict(X_test)
            training_accuracy = accuracy_score(y_test, y_predict)
            confusion_matrix = cm(y_test, y_predict)
            joblib.dump(pipeline, 'pipeline.pkl')
            # return training accuracy, sklearn confusion matrix (from validation step) and sklearn pipeline object
            return training_accuracy, confusion_matrix, pipeline
            
        def apply_pipeline():
            pipeline = joblib.load('pipeline.pkl')
            df = pd.read_csv(df_test)
            predictions = pipeline.predict(df)
            # return predictions or outcomes
            return predictions
  st.write("Then we use this pipeline to predict the value of 'income_>50K' for the test dataset.")  
  if st.button('Test code'):
      with st.echo():
          import pandas as pd
          import newmodel
          X, y = newmodel.load_prepare()
          training_accuracy, confusion_matrix, pipeline = newmodel.build_pipeline_final(X, y)
          print('final model traning accuracy', training_accuracy)
          print('confusion matrix', confusion_matrix)
          predictions = newmodel.apply_pipeline()
          print(predictions)
  st.write("This is my analysis of this dataset all thinking and solution. And if you want to analyze how can get income equal to or more than 50K. You can use the page 'home' or 'predict'. Those pages can predict if you can get an income equal to more than 50K. ")


    




    
    #if st.checkbox():

    
    


  
