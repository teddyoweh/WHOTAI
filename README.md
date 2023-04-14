# WHOT AI

Custom Built 
- Decision Tree Algorithm
- Random Forest Alogirthm
- Word2Vec/Tokenizers
- AI Model Simulation System


## Introduction 
WHOT is a Nigerian Originated card game, somewhat similar to UNO, where player(s) have to match the card placed with same shapes or numbers.
Official Game Doc []()


### Data Generation.
Generated the dataset to train the model by creating a program using the constraints and logic of the game.
7 Features, 5 Input Features (`Card 1,Card 2,Card 3,Card 4,Played`), 1 Target Feature (`Action`)
```csv
id,Card 1,Card 2,Card 3,Card 4,Played,Action
0,circle 1,circle 2,circle 3,circle 4,circle 1,circle 1
1,circle 1,circle 2,circle 3,circle 4,circle 2,circle 1
2,circle 1,circle 2,circle 3,circle 4,circle 3,circle 1
...
```
#### Time Compplexity Analysis
The time complexity of this program can be broken down into multiple parts:

Initializing the store and deck, which is O(1) since it has a fixed size.
The stacks function has a time complexity of O(nk), where n is the number of players and k is the number of cards per player. This is because for each player, the function iterates through k cards to remove them from the deck.
The combinations function has a time complexity of O(C(n, k)), where n is the number of elements in the deck and k is the size of the combination. In this case, it is O(C(54, 4)).
The nested loop that generates the 'box' list has a time complexity of O(C(n, k) * n), where n is the number of elements in the deck and k is the size of the combination. In this case, it is O(C(54, 4) * 54).
The loop that generates the 'data' list has a time complexity of O(C(n, k) * n), which is the same as the loop generating the 'box' list.
The most time-consuming part of the program is the nested loop that generates the 'box' list and the loop that generates the 'data' list. Therefore, the overall time complexity of the program is:

Theta(C(n, k) * n)

In this specific case, the time complexity is:

Theta(C(54, 4) * 54)