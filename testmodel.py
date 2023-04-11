from whotmodel import PostTrainedWhotAI


model = PostTrainedWhotAI('whotmodel','whottokens')
print(model.predict(['circle 1','circle 2','circle 3','circle 4'],'triangle 4'))