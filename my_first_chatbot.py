import re

# Mémoire des apprentissages : un dictionnaire
knowledge_base = {}


def simple_chatbot(user_input):
    user_input_lower = user_input.lower()

    # Cherche si l'utilisateur a déjà appris cette phrase au bot
    if user_input_lower in knowledge_base:
        return knowledge_base[user_input_lower]

    # Réponses pré-définies
    if re.search(r'\b(hi|hello|hey)\b', user_input_lower):
        return "Hello! How can I help you today?"
    elif re.search(r'\b(bye|goodbye)\b', user_input_lower):
        return "Goodbye! Have a great day!"
    elif re.search(r'\b(thank you|thanks)\b', user_input_lower):
        return "You're welcome!"
    elif re.search(r'\b(weather|forecast)\b', user_input_lower):
        return "I'm sorry, I don't have access to weather information."
    else:
        return None  # Inconnu pour l'instant


# Boucle principale
print("Chatbot: Hi! I'm a learning chatbot. Type 'quit' to exit.")

while True:
    user_input = input("You: ")

    if user_input.lower() == 'quit':
        print("Chatbot: Goodbye!")
        break

    response = simple_chatbot(user_input)

    if response is None:
        print("Chatbot: I don't know how to respond to that. What should I say?")
        teach_response = input("You (teaching the bot): ")
        knowledge_base[user_input.lower()] = teach_response
        print("Chatbot: Got it! I'll remember that.")
    else:
        print("Chatbot:", response)
