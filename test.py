### This is only a starter code to run your tests
### Write/edit/extend this code to fully execute your pipeline and test your assignment1.py

import pandas as pd
import newmodel
import pickle

X, y = newmodel.load_prepare()
training_accuracy, confusion_matrix, pipeline = newmodel.build_pipeline_final(X, y)
pickle.dump(pipeline, open('pipeline.pkl', 'wb'))
print('final model traning accuracy', training_accuracy)
print('confusion matrix', confusion_matrix)
predictions = newmodel.apply_pipeline()
print(predictions)

#to find some different condition had what influence to income
pre_income= predictions.astype(int)
df=pd.read_csv('test1.csv')
#add you want to add columns
gender = df['gender']
country = df['native-country']
workhour =df['hours-per-week']
education =df['educational-num']
#...
pre_Df= pd.DataFrame( 
    {'country':country,'income': pre_income}
)

print(pre_Df)
#print(pre_Df[pre_Df.gender == 1].value_counts())
#print(pre_Df[pre_Df.gender == 0].value_counts())
print(pre_Df[pre_Df.income == 1].value_counts())
#pre_Df.to_csv('incomemore50Kpre.csv')

