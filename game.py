#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 13:24:35 2018

@author: ugoslight
"""
import random
import simplegui
import string
import re

"""WHOT: A Card Game"""
SHAPE = ('O', 'T', 'X', 'SQ', '*')
#NUMBER = ('1', '2', '3', '4', '5', '7', '8', '10', '11', '12', '13', '14')
NUMBER = ('1', '2', '3', '4', '5', '7', '8')

class Card:
    def __init__(self, shape, number): 
        if (shape in SHAPE) and (number in NUMBER):
            self.shape = shape
            self.number = number
        else:
            self.shape = None
            self.number = None
            print ("Bad Card.")
    def __str__(self):
        return self.shape + self.number
            
class Deck: 
    def __init__(self):
        self.deck = []
        for shape in SHAPE: 
            for number in NUMBER: 
                if shape == 'X' or shape == 'SQ':
                    if number == '4' or number == '8' or number == '12':
                        continue
                if shape == '*':
                    if number == '10' or number == '11' or number == '12' or number == '13' or number == '14':
                        continue
                self.deck.append(shape + number)
             
        
    def __str__(self):
        response = ""
        for ele in self.deck:
            response += ele
            response += " "
        return response
    
    def __iter__(self):
        for card in self.deck:
            print (card)
             
    def __getitem__(self,index):
        return self.deck[index]
        
    def shuffle(self): 
        random.shuffle(self.deck)
            
    def share(self):
        return self.deck.pop()
    
    def recycle(self, card): 
        self.deck.append(card)
        

        
class Hand:
    def __init__(self):
        self.hand = []
        
    def __iter__(self):
        for card in self.hand:
            print (card)
             
    def __getitem__(self,index):
        return self.hand[index]
    
    def __str__(self): 
        response = ""
        for ele in self.hand:
            response += str(ele)
            response += " "
        return response
    
    def add_card(self, card):
        self.hand.append(card)
        
    def display(self): 
        for ele in (self.hand): 
            print (str(ele))
    
    def remove_card(self, num): 
        self.hand.pop(num - 1)
    
    def play_card(self, strr): 
        return self.hand.remove(strr)
    
    def have(self, num):
        anov = []
        for ele in self.hand: 
            anov.append(ele)
        return anov[num]
        
    def length(self): 
        return len(self.hand)
    

def check(carrd): 
    if carrd == 'GEN': 
        return 1

    if bool(re.search(r'\d', carrd)) == False: return 0
    if carrd in string.digits: return 0
        
    play_no = [int(re.findall('\d+', play)[0])]
    card_no = [int(re.findall('\d+', carrd)[0])]
        
    if play_no == card_no:
        return 1
    if play[0] == carrd[0]: 
        return 1
    else: 
        return 0   

def rule(carrd, player_on): 
    global state
        
    if player_on == True: 
        player_on_name = 'Player'
        player_hit_name = 'Computer'
        player_hit = computer
        player_state = False
            
    if player_on == False: 
        player_on_name = 'Computer'
        player_hit_name = 'Player'
        player_hit = player
        player_state = True

    if carrd[-1] in string.digits and (int(re.findall('\d+', carrd)[0]) == 1 or int(re.findall('\d+', carrd)[0]) == 8):
        print (player_hit_name, 'holds', player_on_name, "play's again.")
        state = player_on
        return 1
            
    if carrd[-1] in string.digits and int(re.findall('\d+', carrd)[0]) == 14:       
        print (player_hit_name, 'goes to market,', player_on_name, "play's again.")
        player_hit.add_card(stack.share())
        state = player_on
        return 1
            
    if carrd[-1] in string.digits and int(re.findall('\d+', carrd)[0]) == 2:
        print (player_hit_name, 'picks two,', player_on_name, "play's again.")
        player_hit.add_card(stack.share())
        player_hit.add_card(stack.share())
        state = player_on
        return 1
        
    if carrd[-1] in string.digits and int(re.findall('\d+', carrd)[0]) == 5:
        print (player_hit_name, 'picks three,', player_on_name, "play's again.")
        player_hit.add_card(stack.share())
        player_hit.add_card(stack.share())
        player_hit.add_card(stack.share())
        state = player_on
        return 1
        
    state = player_state
    
def game():
    global state
    if state == 1:
        print ("            Player:" , player)
        print ("Computer:", computer)
        print (" ")
        cidx = str(input("Please play the name of a card:")).upper()
            
        if check(cidx) and cidx != 'GEN':
            play = cidx
            print ("Player played:", cidx)
            rule(cidx, True)
            stack.recycle(player.play_card(cidx))
                                       
        elif cidx == 'GEN': 
            print ("Player played: GEN")
            player.add_card(stack.share())
            state = 0
                
        elif ' ' in cidx or cidx != 'GEN' or check(cidx) == False: 
            print ("Can't play", cidx, "card.")
            state = 1
                
            
        if player.length() < 1: 
            print ("You Win.")
                

            
    if state == 0: 
        for idx in range(computer.length()):                
            if check(computer[idx]):
                print (" ")
                print ("Comp played: ", computer[idx])
                play = computer[idx]
                rule(computer[idx], False)
                stack.recycle(computer.play_card(computer[idx]))
                    
                break
                
                        
            if idx == computer.length() - 1 and (state == 1 or state == 0): 
                print (" ")
                print ("Computer went to the market.")
                computer.add_card(stack.share())
                state = 1
                break

              
        if computer.length() < 1: 
            print ("Computer Wins.")
                
        state = state                        
        print ("Comps deck: ", computer)
    
    print (" ")
    print (" ")
    print ("Table: ", play)

def main(): 
    global state, play
    state = 1
    print ("WHOT: A Card Game against the computer.")
    
    player = Hand()
    computer = Hand()
    stack = Deck()
    stack.shuffle()
    
    for idx in range(6):
        player.add_card(stack.share())
        computer.add_card(stack.share())
        
    play = str(stack.share())
    print ("First card: ", play)
    
    state = 1
    if player.length() >= 1 or computr.length() >= 1:
        game()
    

        
    
    


            
   
    #while player.length() >= 1 or computer.length() >= 1: 
        
        
    
    



"""def button_handler():
    main()
    
frame = simplegui.create_frame('Testing', 75, 75)
button1 = frame.add_button('Reset', button_handler)

frame.start()"""
main()
