import pandas as pd
from forest import DecisionTree
# from sklearn.ensemble import RandomForestClassifier
from svm import SVM
def whotmodel():
  data =pd.read_csv('data.csv',nrows=10)
  ins = data.drop(columns=['id','Action'])
  outs = data['Action']
  arrs = [__ for _ in ins.values for __ in _]
  tokens  = {_:i for i,_ in enumerate(set(arrs)) }
  outtokens =  {_:i for i,_ in enumerate(set([_ for _ in list(outs.values)])) }
  insdata = []
  outsdata = []
  for i in ins.values:
      insdata.append([tokens[_] for _ in i])
  newdf = pd.DataFrame(data=insdata,columns=ins.columns)
  newdf.head()
  for i in outs.values:
      outsdata.append([outtokens[_] for _ in i])
  newdf = pd.DataFrame(data=insdata,columns=ins.columns)
  newdf.head()
#   outsdata = []
#   for i in outs.values:
#       outsdata.append([outtokens[_] for _ in i])
#   outdf = pd.DataFrame(data=outsdata,columns=outs.columns)
#   outdf.head()
  #model = RandomForest(n_feature=5)
  model = DecisionTree(max_depth=10)
  #model = SVM()
  model.fit(newdf,outs)
  return {'model':model,'tokens':tokens}

mod = whotmodel()
tokens = mod['tokens']
model = mod['model']
def callmodel(cards,played,model1):
  cards.sort()
  card1,card2,card3,card4 = cards
  return model1.predict([[tokens[card1],tokens[card2],tokens[card3],tokens[card4],tokens[played]]])[0]




#print(callmodel(['circle 1','circle 3','triangle 2','whot 20'],'sqaure 2',model))
