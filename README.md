# WHOT AI

Custom Built 
- Decision Tree Algorithm
- Random Forest Alogirthm
- Word2Vec/Tokenizers
- AI Model Simulation System


## Introduction 
WHOT is a Nigerian Originated card game, somewhat similar to UNO, where player(s) have to match the card placed with same shapes or numbers.
Official Game Doc [https://en.wikipedia.org/wiki/Whot!](https://en.wikipedia.org/wiki/Whot!)


### Data Generation.
Generated the dataset to train the model by creating a program using the constraints and logic of the game.

Base Data Generation Code is located in ```./data/data.py```

Parallelized Data Generated Code is located in ```./data/data-pool.py```

7 Features, 5 Input Features (`Card 1,Card 2,Card 3,Card 4,Played`), 1 Target Feature (`Action`)
```csv
id,Card 1,Card 2,Card 3,Card 4,Played,Action
0,circle 1,circle 2,circle 3,circle 4,circle 1,circle 1
1,circle 1,circle 2,circle 3,circle 4,circle 2,circle 1
2,circle 1,circle 2,circle 3,circle 4,circle 3,circle 1
...
```
#### Time Complexity Analysis (Before Parallelization)
The time complexity of this program can be broken down into multiple parts:

- Initializing the store and deck, which is `O(1)` since it has a fixed size.
- The stacks function has a time complexity of `O(nk)`, where n is the number of players and k is the number of cards per player. This is because for each player, the function iterates through k cards to remove them from the deck.
- The combinations function has a time complexity of `O(C(n, k))`, where n is the number of elements in the deck and k is the size of the combination. In this case, it is `O(C(54, 4))`.
- The nested loop that generates the 'box' list has a time complexity of `O(C(n, k) * n)`, where n is the number of elements in the deck and k is the size of the combination. In this case, it is `O(C(54, 4) * 54)`.
- The loop that generates the 'data' list has a time complexity of `O(C(n, k) * n)`, which is the same as the loop generating the 'box' list.
- The most time-consuming part of the program is the nested loop that generates the 'box' list and the loop that generates the 'data' list.
- The overall time complexity of the program is: `O(C(n, k) * n)`
- In this specific case, the time complexity is: `O(C(54, 4) * 54)`

#### Time Complexity Analysis (After Parallelization)
- The time complexity of the nested loop generating the 'box' list is `O(C(54, 4) * 54)`. After parallelizing this loop, the time complexity becomes `O(C(54, 4) * 54 / P)`. The time complexity is reduced by a factor of the number of cores (P) available on the system.
- The time complexity of the process_data function is `O(C(n, k) * n)`, which is the same as the loop generating the 'box' list. When parallelized using P cores, the time complexity becomes `O(C(54, 4) * 54 / P)`.
- The rest of the program has the same time complexity as before, which doesn't change significantly when parallelized. 


### Serialized Objects
I have already trained the model, it is saved in the `./objects/whotmodel` alongside the Word2Vec Tokens using in training, saved in `./objects/whottokens`
