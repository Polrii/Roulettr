import random
import time

starting_bet = 3.20
balance = 1000
stop_loss = 6


lost_rounds = 0
current_round = 0
current_bet = starting_bet

try:
    while balance > current_bet:
        current_round += 1
        balance -= current_bet # Bet on a color
        choice = random.randint(0, 36) # Choice of the color
        
        if 0 < choice < 19: # We won
            balance += 2*current_bet
            lost_rounds = 0
            current_bet = starting_bet
            
        else: # We lost
            lost_rounds += 1
            
            if lost_rounds == stop_loss:
                current_bet = starting_bet
                lost_rounds = 0
            else:
                current_bet += starting_bet/(2**lost_rounds)
        
        print(f"Rounds played: {current_round}     Current balance: {round(balance, 2)}     Ball on: {choice}")
        #time.sleep(0.1)
    print(f"You lost all your money in {current_round} rounds")
            
        

except KeyboardInterrupt:
    print(f"Total rounds played: {current_round}     Final balance: {balance}")