import spacy
from spacy.matcher import Matcher

# Charger le modèle anglais de spaCy
nlp = spacy.load("en_core_web_sm")

# Créer un matcher
matcher = Matcher(nlp.vocab)

# Définir des patterns pour les intentions
greeting_pattern = [{"LOWER": {"IN": ["hello", "hi", "hey"]}}]
booking_pattern = [{"LOWER": {"IN": ["book", "reserve", "reservation"]}}]
menu_pattern = [{"LOWER": "menu"}]
hours_pattern = [{"LOWER": {"IN": ["hours", "open"]}}]

# Ajouter les patterns au matcher
matcher.add("GREETING", [greeting_pattern])
matcher.add("BOOKING", [booking_pattern])
matcher.add("MENU", [menu_pattern])
matcher.add("HOURS", [hours_pattern])

class NLPChatbot:
    def __init__(self):
        self.state = "GREETING"

    def process_input(self, user_input):
        # Traiter l'entrée avec spaCy
        doc = nlp(user_input)
        # Trouver les correspondances
        matches = matcher(doc)
        # Déterminer l'intention
        intent = self.get_intent(matches)
        # Extraire les entités
        entities = self.extract_entities(doc)
        return self.generate_response(intent, entities)

    def get_intent(self, matches):
        if not matches:
            return "UNKNOWN"
        # Retourner la première intention trouvée
        match_id = matches[0][0]
        return nlp.vocab.strings[match_id]

    def extract_entities(self, doc):
        entities = {}
        for ent in doc.ents:
            entities[ent.label_] = ent.text
        return entities

    def generate_response(self, intent, entities):
        if intent == "GREETING":
            return "Hello! How can I assist you today?"
        elif intent == "BOOKING":
            return "Certainly! I'd be happy to help you make a reservation. What date would you like to book?"
        elif intent == "MENU":
            return "You can find our menu at www.restaurant.com/menu. Is there anything specific you'd like to know about our dishes?"
        elif intent == "HOURS":
            return "We're open from 11 AM to 10 PM every day. Would you like to make a reservation?"
        else:
            return "I'm not sure I understood that. Could you please rephrase your request?"

# Utilisation du chatbot
bot = NLPChatbot()
print("Bot: Welcome to our restaurant! How can I assist you today?")

while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        print("Bot: Thank you for using our service. Goodbye!")
        break
    response = bot.process_input(user_input)
    print("Bot:", response)