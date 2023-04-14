from whotmodel import *
import random

def nextplay(cards,card):
  valid = []
  for i in cards:
    if card[1]==14:
      return 'go market'
    if card[1]==8:
      return 'hold on'
    

    if i[0]==card[0] or i[1]==card[1] :

      return i# valid.append(i)
  
  return 'go market'
 
class Log:
   def __init__(self,file):
      self.file = open(file,'a')
      pass

   def log(self,data):
      self.file.write(data)
      
class Player:
    def __init__(self,name,cards,deck):
        self.name = name
        self.cards = cards
        self.deck = deck
    def play(self,model,hand,played):
        return model.predict(hand,played)

    def setcards(self,cards):
       self.cards = cards

    
class Simulate:
    def __init__(self,players:list[Player],deck,logger:Log,objectspath={'model':'whotmodel','tokens':'whottoken'}) -> None:
        self.model = PostTrainedWhotAI(objectspath['model'],objectspath['tokens'])
        self.players = players
        self.deck = deck
        self.logger = logger
    def run(self):
       win = False
       winner = ''
       cardsplayed = []
       cardplayed = self.deck[0]
       self.deck.pop(0)
       while win==False:
          if len(self.deck) ==2:
             self.deck+=cardsplayed
          for i in range(len(self.players)):
             player = self.players[i]
             if len(player.cards)==0:
                
                winner = player.name
                print(player.name,'won')

                win = True
                
                break

     
             cardplayed1 = nextplay(player.cards,cardplayed)
             #cardplayed1 = player.play(self.model,[" ".join([str(_) for _ in card]) for card in player.cards],str(",".join([str(_) for _ in cardplayed]).replace(","," ")))
             print(cardplayed1,'fff')
             logdata =f"""{player.name} With Deck {player.cards} played {cardplayed1 } on action card {cardplayed}"""
             print(logdata)
             self.logger.log(logdata)
             if cardplayed1=='go market':
                player.cards.append(self.deck[0])
                self.deck.pop(0)
             else:
                 cardplayed1=cardplayed1
                 player.cards.remove(cardplayed1)
             cardplayed = cardplayed1



    def get_cards(self,n:int):
        cards =[]
        for _ in range(n):
           n = self.deck[_]
           cards.append(n)
           self.deck.remove(n)
        return cards



class Data:
    circles = [1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14]
    triangles = [1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14]
    crosses = [1, 2, 3, 5, 7, 10, 11, 13, 14]
    squares = [1, 2, 3, 5, 7, 10, 11, 13, 14]
    stars = [1, 2, 3, 4, 5, 7, 8]
    whots = [20, 20, 20, 20]
    stores = {
    'circle': circles,
    'triangle': triangles,
    'cross': crosses,
    'square': squares,
    'star': stars,
    'whot': whots
}
    def __init__(self):
        self.deck=[(i,j) for i in list(self.stores.keys()) for j in self.stores[i]]

   
    def share(self,n):
         random.shuffle(self.deck)
         cards = []
         for i in range(n):
          #  print(n)
          #  print(n*4,(n+1)*4)
           cards.append(self.deck[i*4:(i+1)*4])
         for _ in cards:
           for __ in _:
            self.deck.remove(__)
         return cards
 
   
logs = Log('simulation1.log')

WHOTPACK = Data()
teddyscards,tylerscard = WHOTPACK.share(2)
teddy = Player(name='Teddy',cards=teddyscards,deck=WHOTPACK.deck)
tyler =  Player(name='Tyler',cards=tylerscard,deck=WHOTPACK.deck)
sim = Simulate(players=[teddy,tyler],deck=WHOTPACK.deck,logger=logs,objectspath={'model':'../objects/whotmodel','tokens':'../objects/whottokens'})
sim.run()