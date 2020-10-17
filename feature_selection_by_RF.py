# This has been run locally to yield the input features of subsequent machine learning tasks. The output files are "X_ordered_by_importance.csv" and "y.csv".

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestRegressor

# write the csv file into a fasta file that can be used by the iFeature package
data=pd.read_csv("data1.2013_furin_cleavage_screening.s001.csv",index_col=0)

f=open("13furin.fa","w")
for i in range(len(data.index)):
    f.write(f">{data.index[i]}\n{data.iloc[i,0]}\n")
f.close()


from sklearn.metrics import mean_squared_error, r2_score

class Feature_Select(object):
    """
    A class representation of the Feature selection method.
    """
    
    def __init__(self):
        """
        Initialize.
        
        """
        self.all_feat_mat=[] 
        self.X=[]   #  :params X: a np.array of input variables, length = number of samples.
        self.y=[]   # :params y: a np.array of output labels, length = number of samples.
        self.is_train=[]
        self.is_test=[]
        self.MSE=[]
        self.r2=[]
        self.importance=0
        self.top_importance=[]
        
    def aggregate_all_features(self,feature_path, feature_list):
        """
        aggregating the output of iFeature into a single file

        :params feature_path: the path of "2013furin_{feature}.txt".
        :params feature_list: a list of feature names (char).
        return a pd.DataFrame of rows = samples, cols = features.
        """

        self.all_feat_mat=pd.read_csv(f"{feature_path}2013furin_{feature_list[0]}.txt",index_col=0,sep="\t")
        print(f"n_feature of {feature_list[0]}:{len(self.all_feat_mat.columns)}")
        for feature in feature_list[1:]:
            feat_mat=pd.read_csv(f"{feature_path}2013furin_{feature}.txt",index_col=0,sep="\t")
            print(f"n_feature of {feature}:{len(feat_mat.columns)}")
            self.all_feat_mat=pd.merge(self.all_feat_mat,feat_mat,how='inner',left_index=True,right_index=True)
        # inner: use intersection of keys from both frames
        # 5867 samples × 3625 features
        
    def use_top_feature(self,original_instance, no_feature):
        """
        use preselected features from another feature selection instance to build a model.

        :params original_instance: the original model with all features.
        :params no_feature: the number of features to retain in the model.
        return a pd.DataFrame of rows = samples, cols = features.
        """
        
        feature_list = original_instance.top_importance[:no_feature]
        self.all_feat_mat = original_instance.all_feat_mat.loc[:,feature_list]
        
    def split_train_test(self,n_splits):
        """
        return index after random spliting training and test set for n_splits times.

        :params n_splits: number of folds in cross validation.
        return two numpy arrays of length n_splits.
        """
        random_state = np.random.RandomState(1289237)
        kfold = KFold(n_splits=n_splits, shuffle=True,random_state=random_state)  #不管是几维的输入，第一维都代表样本数
        self.is_train = np.zeros((n_splits, len(self.X)), dtype=np.bool)
        self.is_test = np.zeros((n_splits, len(self.X)), dtype=np.bool)
        print(f"length X{self.X.shape}")
        for i, (train_index, test_index) in enumerate(kfold.split(self.X,self.y)):
            self.is_train[i, train_index] = 1
            self.is_test[i, test_index]=1  
    
    def random_forest_feature_ranking(self,n_fold,iteration,n_estimator, outfile):
        """
        return feature importance ranking acquired from random forest.

        :params n_fold: number of fold in cross validation.
        :params iteration: denote the number of RF to run. 
        :params n_estimator: the number of estimator to use in RF. 
        :params outfile: the file name of output. 
        return a list of genes with decreasing importance.
        """
        
        # loading the output variables, let's start from Z-7.5; there are also Z-15 and Z-30 in data.iloc[:,2] and data.iloc[:,3]
        self.y = np.array(data.iloc[:,1])
        self.X = np.array(self.all_feat_mat)
        self.split_train_test(n_fold)
        self.importance=np.zeros(self.X.shape[1])  # number of features
        
        train_r2=[]
        train_mse=[]
        for fold in range(n_fold):
            for iter in range(iteration):
                regr = RandomForestRegressor(n_estimators=n_estimator)
                regr.fit(self.X[self.is_train[fold]], self.y[self.is_train[fold]])
                for j in range(self.X.shape[1]):
                    #importance[j]=importance[j]+np.argsort(model.feature_importances_)[j]
                    self.importance[j] = self.importance[j] + regr.feature_importances_[j]
                y_pred = regr.predict(self.X[self.is_test[fold]])
                y_train_pred = regr.predict(self.X[self.is_train[fold]])
                self.r2.append(r2_score(y_pred,self.y[self.is_test[fold]]))
                self.MSE.append(mean_squared_error(y_pred,self.y[self.is_test[fold]]))
                train_r2.append(r2_score(y_train_pred,self.y[self.is_train[fold]]))
                train_mse.append(mean_squared_error(y_train_pred,self.y[self.is_train[fold]]))
                
        print(f"Test MSE:{self.MSE}, mean test MSE:{np.mean(self.MSE)}")
        print(f"Test R2:{self.r2}, mean test R2:{np.mean(self.r2)}") 
        print(f"Train MSE:{train_mse}, mean train MSE:{np.mean(train_mse)}")
        print(f"Train R2:{train_r2}, mean train R2:{np.mean(train_r2)}") 

        ind = np.argsort(self.importance)[::-1]    #ind is the order of importance from smallest to largest, ex.[importance[i] for i in ind] gives the largest importance in decreasing order 
        print(f"most important features:{[self.all_feat_mat.columns[i] for i in ind]}")
        self.top_importance = [self.all_feat_mat.columns[i] for i in ind]
        
        with open(outfile,"w") as f:
            f.write(f"Test MSE:{self.MSE}\nTest R2:{self.r2}\nmost important features:{[self.all_feat_mat.columns[i] for i in ind]}")


# 2nd experiment for feature selection by subtracting features.1

feat_sele = Feature_Select()
feat_sele.aggregate_all_features("/Users/chenxinyi/Desktop/fall_course/CS229/final_project/input_data/2013furin_struct_features/",["AAC","EAAC","CKSAAP","DPC","DDE","GAAC","EGAAC","CKSAAGP","GDPC","GTPC","BINARY","BLOSUM62","CTDC","CTDT","CTDD","CTriad","KSCTriad"])
feat_sele.random_forest_feature_ranking(5,1,500,"1st_experiment_with_RF/all_feature_result.txt")  #n_fold,iteration,n_estimator


# 2nd experiment for feature selection by subtracting features.2 (self-selected features)
feat_sele2 = Feature_Select()
feat_sele2.aggregate_all_features("/Users/chenxinyi/Desktop/fall_course/CS229/final_project/input_data/2013furin_struct_features/",["AAC","EAAC","CKSAAGP","GDPC","GTPC","CTDC","CTDT","CTDD"])
feat_sele2.random_forest_feature_ranking(5,1,500,"1st_experiment_with_RF/preselected_feature_result.txt")  #n_fold,iteration,n_estimator


# 2nd experiment for feature selection by subtracting features.3 (RF-selected features)

for no_feature in [50,20,10]:
    feat_sele3 = Feature_Select()
    feat_sele3.use_top_feature(feat_sele,no_feature)
    feat_sele3.random_forest_feature_ranking(5,1,500,"1st_experiment_with_RF/{no_feature}feature_result.txt")  #n_fold,iteration,n_estimator

# have the features listed in the order of feature importance

a=feat_sele.all_feat_mat.loc[:,feat_sele.top_importance]
a.to_csv("X_ordered_by_importance.csv")
labels = data.iloc[:,1]
labels.to_csv("y.csv")

