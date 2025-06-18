import random
from datetime import datetime, timedelta

class RestaurantBookingBot:
    def __init__(self):
        self.state = "GREETING"
        self.booking_info = {}

    def respond(self, user_input):
        if self.state == "GREETING":
            return self.handle_greeting(user_input)
        elif self.state == "ASKING_DATE":
            return self.handle_date(user_input)
        elif self.state == "ASKING_TIME":
            return self.handle_time(user_input)
        elif self.state == "ASKING_PARTY_SIZE":
            return self.handle_party_size(user_input)
        elif self.state == "CONFIRMING":
            return self.handle_confirmation(user_input)
        else:
            return "I'm sorry, I'm not sure how to handle that. Let's start over. How can I help you today?"

    def handle_greeting(self, user_input):
        if "reservation" in user_input.lower() or "book" in user_input.lower():
            self.state = "ASKING_DATE"
            return "Certainly! I'd be happy to help you make a reservation. What date would you like to book?"
        elif "menu" in user_input.lower():
            return "You can find our menu at www.restaurant.com/menu. Is there anything else I can help you with?"
        elif "hours" in user_input.lower() or "open" in user_input.lower():
            return "We're open from 11 AM to 10 PM every day. Would you like to make a reservation?"
        else:
            return "I can help you with reservations, our menu, or our hours. What would you like to know?"

    def handle_date(self, user_input):
        # In a real application, you'd want to properly parse and validate the date
        self.booking_info['date'] = user_input
        self.state = "ASKING_TIME"
        return "Great! What time would you like to book?"

    def handle_time(self, user_input):
        # In a real application, you'd want to properly parse and validate the time
        self.booking_info['time'] = user_input
        self.state = "ASKING_PARTY_SIZE"
        return "Excellent! How many people will be in your party?"

    def handle_party_size(self, user_input):
        try:
            party_size = int(user_input)
            self.booking_info['party_size'] = party_size
            self.state = "CONFIRMING"
            return f"Alright, I have a reservation for {party_size} people on {self.booking_info['date']} at {self.booking_info['time']}. Is this correct?"
        except ValueError:
            return "I'm sorry, I didn't understand that. Could you please provide the number of people in your party?"

    def handle_confirmation(self, user_input):
        if "yes" in user_input.lower() or "correct" in user_input.lower():
            self.state = "GREETING"
            return "Wonderful! Your reservation is confirmed. We look forward to seeing you!"
        else:
            self.state = "GREETING"
            return "I apologize for the confusion. Let's start over. How can I help you today?"

# Lancement du bot
bot = RestaurantBookingBot()
print("Bot: Welcome to our restaurant! How can I assist you today?")

while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        print("Bot: Thank you for using our service. Goodbye!")
        break
    response = bot.respond(user_input)
    print("Bot:", response)
