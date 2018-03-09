import sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
#from sklearn.pipeline import Pipeline
#from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
class NaiveNLP:

    
    def __init__(self, train_set, valid_set,multi_classification=False):
        self.train_set = train_set
        self.valid_set = valid_set
        self.multi_classification = multi_classification
        self.my_LR = sklearn.linear_model.logistic.LogisticRegression()
        self.my_RF = RandomForestClassifier(criterion='entropy',
                                            max_depth= 50,
                                            min_samples_leaf= 1,
                                            min_samples_split= 3,
                                            n_estimators= 50)
        self.my_P = Perceptron(max_iter=10000,tol=0.1)
        self.my_SVM_rbf = SVC(kernel='rbf', gamma=0.03, C=30,max_iter=10000)
        self.my_SVM_linear = SVC(kernel='linear', gamma=0.03, C=30,max_iter=10000)
        self.my_DT = DecisionTreeClassifier()
        self.my_NB = GaussianNB()
        self.my_KNN = KNeighborsClassifier(n_neighbors=3)
        
    def penalized_accuracy(self,predict,target):
        index_un_p = predict=='unrelated'
        index_un = target=='unrelated'
        index_re = np.where(predict!='unrelated')
        acc1 = np.mean(index_un_p==index_un)
        acc2 = np.mean(predict[index_re]==np.array(target)[index_re])
        return(str(0.25*acc1+0.75*acc2))
        
        
    def method_KNeighborsClassifier(self):
#        pipeline = Pipeline([('clf', KNeighborsClassifier())])
#        parameters = {'clf__n_neighbors': (5, 10, 3, 50)}
#        grid_search = GridSearchCV(pipeline, 
#                                   parameters, 
#                                   verbose=1,
#                                   scoring='accuracy')
#        grid_search.fit(self.train_set[0], self.train_set[1])
#        print('Best score: %0.3f' % grid_search.best_score_)
#        print('Best parameters; ')
#        best_parameters = grid_search.best_estimator_.get_params()
#        for param_name in sorted(best_parameters.keys()):
#            print('\t%s: %r' % (param_name, best_parameters[param_name]))
        self.my_KNN.fit(self.train_set[0], self.train_set[1])
        self.my_KNN_pred = self.my_KNN.predict(self.valid_set[0])
        self.my_KNN_acc = accuracy_score(self.my_KNN_pred, self.valid_set[1])
        print('KNeighborsClassifier accuracy is: ' + str(self.my_KNN_acc))
        if self.multi_classification:
            print('KNeighborsClassifier penalized accuracy is: ' + self.penalized_accuracy(self.my_KNN_pred,self.valid_set[1]))
        
        
    def method_GaussianNB(self):
        self.my_NB.fit(self.train_set[0], self.train_set[1])
        self.my_NB_pred = self.my_NB.predict(self.valid_set[0])
        self.my_NB_acc = accuracy_score(self.my_NB_pred, self.valid_set[1])
        print('GaussianNB accuracy is: ' + str(self.my_NB_acc))
        if self.multi_classification:
            print('GaussianNB penalized accuracy is: ' + self.penalized_accuracy(self.my_NB_pred,self.valid_set[1]))
        
        
        
    def method_LogisticRegression(self):
        self.my_LR.fit(self.train_set[0], self.train_set[1])
        self.my_LR_pred = self.my_LR.predict(self.valid_set[0])
        self.my_LR_acc = accuracy_score(self.my_LR_pred, self.valid_set[1])
        print('LogisticRegression accuracy is: ' + str(self.my_LR_acc))
        
    def method_DecisionTreeClassifier(self):
        self.my_DT.fit(self.train_set[0], self.train_set[1])
        self.my_DT_pred = self.my_DT.predict(self.valid_set[0])
        self.my_DT_acc = accuracy_score(self.my_DT_pred, self.valid_set[1])
        print('DecisionTreeClassifier accuracy is: ' + str(self.my_DT_acc))
        if self.multi_classification:
            print('DecisionTreeClassifier penalized accuracy is: ' + self.penalized_accuracy(self.my_DT_pred,self.valid_set[1]))
        
    def method_RandomForestClassifier(self):
#        pipeline = Pipeline([('clf', RandomForestClassifier(criterion='entropy'))])
#        parameters = {'clf__n_estimators': (5, 10, 20, 50),
#                      'clf__max_depth': (50, 150, 250),
#                      'clf__min_samples_split': (1.0, 2, 3),
#                      'clf__min_samples_leaf': (1, 2, 3)}
#        grid_search = GridSearchCV(pipeline, 
#                                   parameters, 
#                                   #n_jobs=-1,
#                                   verbose=1,
#                                   scoring='accuracy')
#        grid_search.fit(self.train_set[0], self.train_set[1])
#        print('Best score: %0.3f' % grid_search.best_score_)
#        print('Best parameters; ')
#        best_parameters = grid_search.best_estimator_.get_params()
#        for param_name in sorted(best_parameters.keys()):
#            print('\t%s: %r' % (param_name, best_parameters[param_name]))
#        self.my_RF_score = self.scores(grid_search, 
#                                       self.valid_set[0], 
#                                       self.valid_set[1], 
#                                       cv=5)
        self.my_RF.fit(self.train_set[0], self.train_set[1])
        #self.my_RF_score = self.my_RF.score(self.valid_set[0],self.valid_set[1])
        self.my_RF_pred = self.my_RF.predict(self.valid_set[0])
        self.my_RF_acc = accuracy_score(self.my_RF_pred, self.valid_set[1])
        print('RandomForestClassifier accuracy is: ' + str(self.my_RF_acc))
        if self.multi_classification:
            print('RandomForestClassifier penalized accuracy is: ' + self.penalized_accuracy(self.my_RF_pred,self.valid_set[1]))
        
    def method_Perception(self):
        self.my_P.fit(self.train_set[0],self.train_set[1])
        self.my_P_pred = self.my_P.predict(self.valid_set[0])
        self.my_P_acc = accuracy_score(self.my_P_pred, self.valid_set[1])
        print('Perception accuracy is: ' + str(self.my_P_acc))
        if self.multi_classification:
            print('Perception penalized accuracy is: ' + self.penalized_accuracy(self.my_P_pred,self.valid_set[1]))
        
    def method_SVM_rbf(self):
#        pipeline = Pipeline([('clf', sklearn.svm.SVC(kernel='rbf', gamma=0.01, C=100))])
#        parameters = {'clf__gamma': (0.01, 0.03, 0.1, 0.3, 1),    
#                      'clf__C': (0.1, 0.3, 1, 3, 10, 30), }
#        parameters = {'clf__gamma': (0.03),    
#                      'clf__C': (30), }
#        grid_search = GridSearchCV(pipeline, 
#                                   parameters, 
#                                   verbose=1, 
#                                   scoring='accuracy')
#        grid_search.fit(self.train_set[0], self.train_set[1])
#        print('Best score：%0.3f' % grid_search.best_score_)
#        print('Best paragram：')
#        best_parameters = grid_search.best_estimator_.get_params()
#        for param_name in sorted(parameters.keys()):
#            print('\t%s: %r' % (param_name, best_parameters[param_name]))
        self.my_SVM_rbf.fit(self.train_set[0], self.train_set[1])
        self.my_SVM_rbf_pred = self.my_SVM_rbf.predict(self.valid_set[0])
        self.my_SVM_rbf_acc = accuracy_score(self.my_SVM_rbf_pred, self.valid_set[1])
        print('SVM_rbf accuracy is: ' + str(self.my_SVM_rbf_acc))
        if self.multi_classification:
            print('SVM_rbf penalized accuracy is: ' + self.penalized_accuracy(self.my_SVM_rbf_pred,self.valid_set[1]))
        
    def method_SVM_linear(self):
#        pipeline = Pipeline([('clf', SVC(kernel='linear', gamma=0.01, C=100))])
#        parameters = {'clf__gamma': (0.01, 0.03, 0.1, 0.3, 1),    
#                      'clf__C': (0.1, 0.3, 1, 3, 10, 30), }
#        grid_search = GridSearchCV(pipeline, 
#                                   parameters, 
#                                   verbose=1, 
#                                   scoring='accuracy')
#        grid_search.fit(self.train_set[0], self.train_set[1])
#        print('Best score：%0.3f' % grid_search.best_score_)
#        print('Best paragram：')
#        best_parameters = grid_search.best_estimator_.get_params()
#        for param_name in sorted(parameters.keys()):
#            print('\t%s: %r' % (param_name, best_parameters[param_name]))
#        #self.my_SVM_rbf.fit(self.train_set[0], self.train_set[1])
#        self.my_SVM_rbf_score = grid_search.score(self.valid_set[0],self.valid_set[1])
#        print('SVM_rbf score is: ' + str(self.my_SVM_rbf_score))
        self.my_SVM_linear.fit(self.train_set[0], self.train_set[1])
        self.my_SVM_linear_pred = self.my_SVM_linear.predict(self.valid_set[0])
        self.my_SVM_linear_acc = accuracy_score(self.my_SVM_linear_pred, self.valid_set[1])
        print('SVM_linear accuracy is: ' + str(self.my_SVM_linear_acc))
        if self.multi_classification:
            print('SVM_linear penalized accuracy is: ' + self.penalized_accuracy(self.my_SVM_linear_pred,self.valid_set[1]))    
        
        
if __name__ == '__main__':        
    xTrain = x1Train-x2Train
    xValid = x1Valid-x2Valid
#    def string_to_number(yset):
#        y = []
#        for i in range(len(yset)):
#            if yset[i]=='unrelated':
#                y.append(-1)
#            else:
#                y.append(1)
#        return y
#    yTrain_number = string_to_number(yTrain)
#    yValid_number = string_to_number(yValid)
#    nlp_b = NaiveNLP(train_set=[xTrain, yTrain_number], 
#                   valid_set=[xValid, yValid_number])
#    print('-----Below are binary classification result:-----')
#    nlp_b.method_LogisticRegression()
#    nlp_b.method_GaussianNB()
#    nlp_b.method_Perception()
#    nlp_b.method_DecisionTreeClassifier()
#    nlp_b.method_KNeighborsClassifier()
#    nlp_b.method_RandomForestClassifier()
    #nlp_b.method_SVM_linear()
    #nlp_b.method_SVM_rbf()
    nlp_m = NaiveNLP(train_set=[xTrain, yTrain], 
                   valid_set=[xValid, yValid],
                   multi_classification=True)
    print('-----Below are multi-classification result:-----')
    nlp_m.method_GaussianNB()
    nlp_m.method_Perception()
    nlp_m.method_DecisionTreeClassifier()
    nlp_m.method_KNeighborsClassifier()
    nlp_m.method_RandomForestClassifier()
    #nlp_m.method_SVM_linear()
    nlp_m.method_SVM_rbf()
    
    