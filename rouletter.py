import random
import time
import json

starting_balance = 1000
balance = starting_balance
stop_loss = 6
betting_order = [3.2, 4.8, 8.8, 17.2, 34.2, 68.3]
json_path = "Data.json"


thousand_rounds = 0
max_money = balance

lost_rounds = 0
current_round = 0
current_bet = betting_order[lost_rounds]
data = {}

with open(json_path, 'w') as file:
    try:
        while balance > current_bet:
            current_round += 1
            balance -= current_bet # Bet on a color
            choice = random.randint(0, 36) # Choice of the color
            
            if 0 < choice < 19: # We won
                balance += 2*current_bet
                lost_rounds = 0
                
            else: # We lost
                lost_rounds += 1
                
                if lost_rounds == stop_loss:
                    lost_rounds = 0
            
            current_bet = betting_order[lost_rounds]
            if balance > starting_balance:
                thousand_rounds += 1
                if balance > max_money:
                    max_money = balance
                
            
            print(f"Rounds played: {current_round}     Current balance: {round(balance, 2)}     Ball on: {choice}")
            data[current_round] = {"Current balance": round(balance, 2), "Ball on": choice}
            #time.sleep(0.1)
        print(f"You lost all your money in {current_round} rounds, and spent {thousand_rounds} in positive, with a maximum of {max_money}")
                
            

    except KeyboardInterrupt:
        print(f"Total rounds played: {current_round}     Final balance: {balance}")
    
    file.write("")
    json.dump(data, file, indent=4)