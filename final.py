import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection,linear_model,metrics
from sklearn.pipeline import make_union
import lightgbm
from sklearn.externals import joblib

if __name__=='__main__':
    train = pd.read_csv('train_2kmZucJ.csv').fillna(' ')
    test = pd.read_csv('test_oJQbWVk.csv').fillna(' ')

    train_text = train['tweet']
    test_text = test['tweet']
    all_text = pd.concat([train_text, test_text])

    word_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\w{1,}',
        ngram_range=(1, 1),
        max_features=6000)
    char_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='char',
        ngram_range=(1, 4),
        max_features=6000)
    vectorizer = make_union(word_vectorizer, char_vectorizer, n_jobs=-1)
    vectorizer.fit(all_text)
    train_features = vectorizer.transform(train_text)
    test_features = vectorizer.transform(test_text)
    submission = pd.DataFrame.from_dict({'id': test['id']})
    skf=model_selection.StratifiedKFold(11)
    classifier=lightgbm.LGBMClassifier(random_state=42)
    score=[]
    prediction=pd.DataFrame({'id':test['id']})
    for fold,(train_idx,val_idx) in enumerate(skf.split(train_features,train.label)):
        xtrain,ytrain=train_features[train_idx],train['label'][train_idx]
        xval,yval=train_features[val_idx],train['label'][val_idx]
        classifier.fit(xtrain,ytrain)
        ypred=classifier.predict(xval)
        fold_score=metrics.f1_score(yval,ypred,average='weighted')
        score.append(fold_score)
        print('{}_score:{}'.format(fold,fold_score))
        test_prediction=classifier.predict(test_features)
        assert test_prediction.shape[0]==test.shape[0]
        prediction['fold_{}'.format(fold)]=test_prediction
    joblib.dump(classifier,'model.pkl')
    prediction['label']=np.round(prediction.iloc[:,1:].mean(axis=1).values)
    submission=pd.DataFrame({'id':prediction['id'],'label':prediction['label']})
    assert submission.shape[0]==test.shape[0]
    print(submission.head())
    print(submission['label'].value_counts()) 
    submission.to_csv('submission.csv',index=False)  


