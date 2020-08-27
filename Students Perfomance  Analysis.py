import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings("ignore")
from math import *
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


path ="StudentsPerformance.csv"
class DataFrame_Loader():
    data = path
    
    def __init__(self):
        
        print("Loadind DataFrame")
        
    def read_csv(self,data):
        self.df = pd.read_csv(data)
        
    def load_csv(self):
        return self.df


class DataFrame_Information():

    def __init__(self):
        
        print("Attribute Information object created")
        
    def Attribute_information(self,df):
    
        data_info = pd.DataFrame(
                                columns=['No of observation',
                                        'No of Variables',
                                        'No of Numerical Variables',
                                        'No of Factor Variables',
                                        'No of Categorical Variables',
                                        'No of Logical Variables',
                                        'No of Date Variables',
                                        'No of zero variance variables'])


        data_info.loc[0,'No of observation'] = df.shape[0]
        data_info.loc[0,'No of Variables'] = df.shape[1]
        data_info.loc[0,'No of Numerical Variables'] = df._get_numeric_data().shape[1]
        data_info.loc[0,'No of Factor Variables'] = df.select_dtypes(include='category').shape[1]
        data_info.loc[0,'No of Logical Variables'] = df.select_dtypes(include='bool').shape[1]
        data_info.loc[0,'No of Categorical Variables'] = df.select_dtypes(include='object').shape[1]
        data_info.loc[0,'No of Date Variables'] = df.select_dtypes(include='datetime64').shape[1]
        data_info.loc[0,'No of zero variance variables'] = df.loc[:,df.apply(pd.Series.nunique)==1].shape[1]

        data_info =data_info.transpose()
        data_info.columns=['value']
        data_info['value'] = data_info['value'].astype(int)


        return data_info

    def __get_missing_values(self,data):
        
        #Getting sum of missing values for each feature
        missing_values = data.isnull().sum()
        #Feature missing values are sorted from few to many
        missing_values.sort_values(ascending=False, inplace=True)
        
        #Returning missing values
        return missing_values

    def Generate_Schema(self,data):
        
        feature_dtypes=data.dtypes
        self.missing_values=self.__get_missing_values(data)

        print("=" * 110)

        print("{:16} {:16} {:20} {:16}".format("Feature Name".upper(),
                                            "Data Type".upper(),
                                            "# of Missing Values".upper(),
                                            "Samples".upper()))
        for feature_name, dtype, missing_value in zip(self.missing_values.index.values,
                                                      feature_dtypes[self.missing_values.index.values],
                                                      self.missing_values.values):
            print("{:18} {:19} {:19} ".format(feature_name, str(dtype), str(missing_value)), end="")
            for v in data[feature_name].values[:5]:
                print(v, end=",")
            print()

        print("="*110)
        
    def Agg_Tabulation(self,data):
        
        print("=" * 110)
        print("Aggregation of Table")
        print("=" * 110)
        table = pd.DataFrame(data.dtypes,columns=['dtypes'])
        table1 =pd.DataFrame(data.columns,columns=['Names'])
        table = table.reset_index()
        table= table.rename(columns={'index':'Name'})
        table['No of Missing'] = data.isnull().sum().values    
        table['No of Uniques'] = data.nunique().values
        table['Percent of Missing'] = ((data.isnull().sum().values)/ (data.shape[0])) *100
        table['First Observation'] = data.loc[0].values
        table['Second Observation'] = data.loc[1].values
        table['Third Observation'] = data.loc[2].values
        for name in table['Name'].value_counts().index:
            table.loc[table['Name'] == name, 'Entropy'] = round(stats.entropy(data[name].value_counts(normalize=True), base=2),2)
        return table
    
        print("=" * 110)
        
    def __IQR(self,x):
        return x.quantile(q=0.75) - x.quantile(q=0.25)

    def __Outlier_Count(self,x):
        upper_out = x.quantile(q=0.75) + 1.5 * self.__IQR(x)
        lower_out = x.quantile(q=0.25) - 1.5 * self.__IQR(x)
        return len(x[x > upper_out]) + len(x[x < lower_out])

    def Numeric_Count_Summary(self,df):
        df_num = df._get_numeric_data()
        data_info_num = pd.DataFrame()
        i=0
        for c in  df_num.columns:
            data_info_num.loc[c,'Negative values count']= df_num[df_num[c]<0].shape[0]
            data_info_num.loc[c,'Positive values count']= df_num[df_num[c]>0].shape[0]
            data_info_num.loc[c,'Zero count']= df_num[df_num[c]==0].shape[0]
            data_info_num.loc[c,'Unique count']= len(df_num[c].unique())
            data_info_num.loc[c,'Negative Infinity count']= df_num[df_num[c]== -np.inf].shape[0]
            data_info_num.loc[c,'Positive Infinity count']= df_num[df_num[c]== np.inf].shape[0]
            data_info_num.loc[c,'Missing Percentage']= df_num[df_num[c].isnull()].shape[0]/ df_num.shape[0]
            data_info_num.loc[c,'Count of outliers']= self.__Outlier_Count(df_num[c])
            i = i+1
        return data_info_num
    
    def Statistical_Summary(self,df):
    
        df_num = df._get_numeric_data()

        data_stat_num = pd.DataFrame()

        try:
            data_stat_num = pd.concat([df_num.describe().transpose(),
                                       pd.DataFrame(df_num.quantile(q=0.10)),
                                       pd.DataFrame(df_num.quantile(q=0.90)),
                                       pd.DataFrame(df_num.quantile(q=0.95))],axis=1)
            data_stat_num.columns = ['count','mean','std','min','25%','50%','75%','max','10%','90%','95%']
        except:
            pass

        return data_stat_num
    
    def group_by_Num_variables(self,df,x,y):
        d=df.groupby([x])[y].describe()
        data_stat_num = pd.DataFrame()

        try:
                data_stat_num = pd.concat([d,
                                           pd.DataFrame(df_num.quantile(q=0.10)),
                                           pd.DataFrame(df_num.quantile(q=0.90)),
                                           pd.DataFrame(df_num.quantile(q=0.95))],axis=1)
                data_stat_num.columns = ['count','mean','std','min','25%','50%','75%','max','10%','90%','95%']
        except:
            pass

        return data_stat_num



class DataFrame_Visualizer():

    def __init__(self):
        
        print("Visualizer object created")
        
    def Bar_graph(self,df,x):
        plt.figure(figsize=(20,7))
        x.value_counts(normalize = True)
        x.value_counts(dropna = False).plot.bar(color='blue')
        plt.xlabel('variable')
        plt.ylabel('count')
        plt.show()
        
    def cross_tab_with_stacked_bar_chart(self,df,x,y):
        x = pd.crosstab(x, y)
        return x.div(x.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True, figsize = (4, 4))
    
    def count_plot_for_variables(self,df,x,y,z):
        sns.countplot(x = x, data = df, hue = y, palette = z)
        plt.show()
        
    def Calculate_Pass_Math_with_math_score(self,df):    
        passmarks = 40

        # creating a new column pass_math, this column will tell us whether the students are pass or fail
        df['pass_math'] = np.where(df['math_score']< passmarks, 'Fail', 'Pass')
        df['pass_math'].value_counts(dropna = False).plot.bar(color = 'black', figsize = (5, 3))

        plt.title('Comparison of students passed or failed in maths')
        plt.xlabel('status')
        plt.ylabel('count')
        plt.show()

    def Calculate_Marks_with_math_score(self,df):
    
        passmarks = 40

        df['pass_writing'] = np.where(df['math_score']< passmarks, 'Fail', 'Pass')
        df['pass_writing'].value_counts(dropna = False).plot.bar(color = 'green', figsize = (5, 3))

        plt.title('Comparison of students passed or failed in maths')
        plt.xlabel('status')
        plt.ylabel('count')
        plt.show()
        
    def Calculate_Marks_with_writing_score(self,df):
        
        passmarks = 40

        df['pass_writing'] = np.where(df['writing_score']< passmarks, 'Fail', 'Pass')
        df['pass_writing'].value_counts(dropna = False).plot.bar(color = 'blue', figsize = (5, 3))

        plt.title('Comparison of students passed or failed in maths')
        plt.xlabel('status')
        plt.ylabel('count')
        plt.show()
        
    def Calculate_total_Score_with_math_score(self,df):
        
        df['total_score'] = df['math_score'] + df['reading_score'] + df['writing_score']

        df['total_score'].value_counts(normalize = True)
        df['total_score'].value_counts(dropna = True).plot.bar(color = 'red', figsize = (40, 8))

        plt.title('comparison of total score of all the students')
        plt.xlabel('total score scored by the students')
        plt.ylabel('count')
        plt.show()
        
     
    def Calculate_percentage_with_total_Score(self,df):

        df['percentage'] = df['total_score']/3

        for i in range(0, 1000):
            df['percentage'][i] = ceil(df['percentage'][i])

        df['percentage'].value_counts(normalize = True)
        df['percentage'].value_counts(dropna = False).plot.bar(figsize = (16, 8), color = 'red')

        plt.title('Comparison of percentage scored by all the students')
        plt.xlabel('percentage score')
        plt.ylabel('count')
        plt.show()
        
    def Calculate_pass_reading_with_reading_score(self,df):    
        
        passmarks = 40
        df['pass_reading'] = np.where(df['reading_score']< passmarks, 'Fail', 'Pass')
        df['pass_reading'].value_counts(dropna = False).plot.bar(color = 'brown', figsize = (5, 3))

        plt.title('Comparison of students passed or failed in maths')
        plt.xlabel('status')
        plt.ylabel('count')
        plt.show()
        
    def Calculate_status_with_pass_math_and_pass_writing(self,df):
        
        passmarks = 40
        df['status'] = df.apply(lambda x : 'Fail' if x['pass_math'] == 'Fail' or 
                           x['pass_reading'] == 'Fail' or x['pass_writing'] == 'Fail'
                           else 'pass', axis = 1)

        df['status'].value_counts(dropna = False).plot.bar(color = 'gray', figsize = (3, 3))
        plt.title('overall results')
        plt.xlabel('status')
        plt.ylabel('count')
        plt.show()
        
    def pie_chart(self):
    
        labels = ['Grade 0', 'Grade A', 'Grade B', 'Grade C', 'Grade D', 'Grade E']
        sizes = [58, 156, 260, 252, 223, 51]
        colors = ['yellow', 'gold', 'lightskyblue', 'lightcoral', 'pink', 'cyan']
        explode = (0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001)

        patches, texts = plt.pie(sizes, colors=colors, shadow=True, startangle=90)
        plt.legend(patches, labels)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
        
    def getgrade(self,percentage, status):
        if status == 'Fail':
            return 'E'
        if(percentage >= 90):
            return 'O'
        if(percentage >= 80):
            return 'A'
        if(percentage >= 70):
            return 'B'
        if(percentage >= 60):
            return 'C'
        if(percentage >= 40):
            return 'D'
        else :
            return 'E'



class Base_Feature_Engineering():

    def __init__(self):
        print("Feature Engineering object created")
    
    def _Label_Encoding(self,data):
        category_col =[var for var in data.columns if data[var].dtypes =="object"] 
        labelEncoder = preprocessing.LabelEncoder()
        mapping_dict={}
        for col in category_col:
            data[col] = labelEncoder.fit_transform(data[col])
            le_name_mapping = dict(zip(labelEncoder.classes_, labelEncoder.transform(labelEncoder.classes_)))
            mapping_dict[col]=le_name_mapping
            return mapping_dict




class Model_Selector():

    def __init__(self,n_estimators=100,random_state=42,max_depth=10):
        print("Model Selector object created")
        
    def Regression_Model_Selector(self,df):
        seed = 42
        models = []
        models.append(("LR", LogisticRegression()))
        models.append(("RF", RandomForestClassifier()))
        models.append(("KNN", KNeighborsClassifier()))
        models.append(("CART", DecisionTreeClassifier()))
        models.append(("XGB", XGBClassifier()))
        result = []
        names = []
        scoring = 'accuracy'
        seed = 42

        for name, model in models:
            x = df.drop(['grades'],axis=1)
            y = df['grades']
            x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=42)
            kfold = KFold(n_splits = 5, random_state =seed)# 5 split of data (value of k)
            cv_results = cross_val_score(model, x_train, y_train, cv = kfold, scoring = scoring)
            result.append(cv_results)
            names.append(name)
            msg = (name, cv_results.mean(), cv_results.std())
            print(msg)
        fig = plt.figure(figsize = (8,4))
        fig.suptitle('Algorithm Comparison')
        ax = fig.add_subplot(1,1,1)
        plt.boxplot(result)
        ax.set_xticklabels(names)
        plt.show()


from sklearn.metrics import accuracy_score
x = df.drop(['grades'],axis=1)
y = df['grades']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=42)
class Data_Modelling():

    def __init__(self,n_estimators=100,random_state=42,max_depth=10):
        print("Data Modelling object created")

    def Decision_Tree_Model(self,df):
        Classifier = DecisionTreeClassifier(random_state=29)
        clf=Classifier.fit(x_train,y_train)
        DT_pred=Classifier.predict(x_test)
        print("confusion matrix",confusion_matrix(y_test, DT_pred))
        print("classification_report",classification_report(y_test,DT_pred))
        return accuracy_score(y_test,DT_pred)
    
    def Random_Forest_Model(self,df):
        Classifier = RandomForestClassifier(n_estimators=100,random_state=29,max_depth=12)
        clf=Classifier.fit(x_train,y_train)
        RF_pred=Classifier.predict(x_test)
        print("confusion matrix",confusion_matrix(y_test, RF_pred))
        print("classification_report",classification_report(y_test,RF_pred))
        return accuracy_score(y_test,RF_pred)

    def Extreme_Gradient_Boosting_Model(self,df):
        Classifier = XGBClassifier(n_estimators=100,random_state=29,max_depth=9,learning_rate=0.07)
        clf=Classifier.fit(x_train,y_train)
        XGB_pred=Classifier.predict(x_test)
        print("confusion matrix",confusion_matrix(y_test, XGB_pred))
        print("classification_report",classification_report(y_test,XGB_pred))
        return accuracy_score(y_test,XGB_pred)


if __name__ == '__main__':


	data = DataFrame_Loader()
	data.read_csv(path)
	df=data.load_csv()
	df = df.rename(columns={'math score':'math_score','reading score':'reading_score','writing score':'writing_score'})



	Info = DataFrame_Information()
	Info.Attribute_information(df)
	Info.Generate_Schema(df)
	Info.Agg_Tabulation(df)
	Info.Numeric_Count_Summary(df)
	Info.Statistical_Summary(df)


	visualizer = DataFrame_Visualizer()
	visualizer.Bar_graph(df,df['gender'])
	visualizer.Bar_graph(df,df['lunch'])
	visualizer.Bar_graph(df,df['race/ethnicity'])
	visualizer.Bar_graph(df,df['math_score'])
	visualizer.Bar_graph(df,df['reading_score'])
	visualizer.Bar_graph(df,df['writing_score'])
	visualizer.cross_tab_with_stacked_bar_chart(df,df['gender'],df['race/ethnicity'])
	visualizer.cross_tab_with_stacked_bar_chart(df,df['race/ethnicity'],df['parental level of education'])
	visualizer.cross_tab_with_stacked_bar_chart(df,df['race/ethnicity'],df['lunch'])
	visualizer.cross_tab_with_stacked_bar_chart(df,df['race/ethnicity'],df['gender'])
	visualizer.cross_tab_with_stacked_bar_chart(df,df['parental level of education'],df['race/ethnicity'])
	visualizer.count_plot_for_variables(df,'parental level of education','test preparation course','dark')
	visualizer.count_plot_for_variables(df,'race/ethnicity','test preparation course','bright')
	visualizer.count_plot_for_variables(df,'lunch','test preparation course','rocket')
	visualizer.Calculate_Marks_with_math_score(df)
	visualizer.Calculate_Marks_with_writing_score(df)
	visualizer.Calculate_Pass_Math_with_math_score(df)
	visualizer.Calculate_total_Score_with_math_score(df)
	visualizer.Calculate_percentage_with_total_Score(df)
	visualizer.Calculate_pass_reading_with_reading_score(df)
	visualizer.Calculate_status_with_pass_math_and_pass_writing(df)
	visualizer.pie_chart()
	df['grades'] = df.apply(lambda x: visualizer.getgrade(x['percentage'], x['status']), axis = 1 )
    df['grades'].value_counts()
    visualizer.cross_tab_with_stacked_bar_chart(df,df['parental level of education'],df['grades'])
    visualizer.count_plot_for_variables(df,'parental level of education','grades','dark')



    df['race/ethnicity'] = df['race/ethnicity'].replace('group A', 1)
	df['race/ethnicity'] = df['race/ethnicity'].replace('group B', 2)
	df['race/ethnicity'] = df['race/ethnicity'].replace('group C', 3)
	df['race/ethnicity'] = df['race/ethnicity'].replace('group D', 4)
	df['race/ethnicity'] = df['race/ethnicity'].replace('group E', 5)

	df['race/ethnicity'].value_counts()

	df['grades'] = df['grades'].replace('O', 0)
	df['grades'] = df['grades'].replace('A', 1)
	df['grades'] = df['grades'].replace('B', 2)
	df['grades'] = df['grades'].replace('C', 3)
	df['grades'] = df['grades'].replace('D', 4)
	df['grades'] = df['grades'].replace('E', 5)

	df['race/ethnicity'].value_counts()


	fe = Base_Feature_Engineering()
	fe._Label_Encoding(df)


	MS = Model_Selector()
	MS.Regression_Model_Selector(df)


	model = Data_Modelling()
	model.Decision_Tree_Model(df)
	model.Random_Forest_Model(df)
	model.Extreme_Gradient_Boosting_Model(df)




