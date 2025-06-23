import spacy
from fuzzywuzzy import process
import re

nlp = spacy.blank("fr")
textcat = nlp.add_pipe("textcat")

INTENTIONS = ["SALUTATION", "DEMANDE_RETOUR", "INFOS_RETOUR", "REMBOURSEMENT", "SUIVI_RETOUR", "QUESTION_GENERALE"]

for label in INTENTIONS:
    textcat.add_label(label)

# Entraînement simple
train_data = [
    ("bonjour", {"cats": {"SALUTATION": 1}}),
    ("je veux renvoyer un produit", {"cats": {"DEMANDE_RETOUR": 1}}),
    ("comment faire un retour ?", {"cats": {"INFOS_RETOUR": 1}}),
    ("je veux un remboursement", {"cats": {"REMBOURSEMENT": 1}}),
    ("où en est mon retour ?", {"cats": {"SUIVI_RETOUR": 1}}),
    ("j'ai une question sur mon retour", {"cats": {"QUESTION_GENERALE": 1}})
]

optimizer = nlp.initialize()
for _ in range(10):
    for text, ann in train_data:
        ex = spacy.training.Example.from_dict(nlp.make_doc(text), ann)
        nlp.update([ex], sgd=optimizer)

def fuzzy_match(text, options):
    match, score = process.extractOne(text, options)
    return match if score > 70 else None

def extract_order_number(text):
    match = re.search(r'CMD-\d{6}', text, re.IGNORECASE)
    return match.group().upper() if match else None


VALID_PRODUCTS = ["Casque Bluetooth", "Chargeur USB", "Coque iPhone", "Souris sans fil"]

# Dictionnaire large des raisons mappé vers des catégories
REASON_CATEGORIES = {
    "Défectueux": ["cassé", "abîmé", "ne fonctionne pas", "endommagé", "il est arrivé cassé", "brisé", "écran fissuré", "manque une pièce"],
    "Taille incorrecte": ["pas la bonne taille", "trop grand", "trop petit", "mauvaise taille"],
    "Mauvaise couleur": ["mauvaise couleur", "pas la bonne couleur", "je voulais une autre couleur"],
    "Ne correspond pas à la description": ["pas comme sur la photo", "ne correspond pas", "différent de la description", "pas conforme"]
}

def match_reason(text):
    flat_list = []
    mapping = {}
    for category, phrases in REASON_CATEGORIES.items():
        for phrase in phrases:
            flat_list.append(phrase)
            mapping[phrase] = category
    match = fuzzy_match(text, flat_list)
    if match:
        return mapping[match]
    return None

class ChatbotNLP:
    def __init__(self):
        self.awaiting_order = False
        self.awaiting_product = False
        self.awaiting_reason = False
        self.order_number = None
        self.product_name = None

    def process_input(self, text):
        text = text.strip()

        if self.awaiting_order:
            order = extract_order_number(text)
            if order:
                self.order_number = order
                self.awaiting_order = False
                self.awaiting_product = True
                return f"Merci. Quel produit de la commande {self.order_number} souhaitez-vous retourner ? (Exemple : Casque Bluetooth)"
            else:
                return "❌ Le numéro de commande est invalide. Il doit être au format CMD-XXXXXX (6 chiffres). Veuillez réessayer."

        if self.awaiting_product:
            match = fuzzy_match(text, VALID_PRODUCTS)
            if match:
                self.product_name = match
                self.awaiting_product = False
                self.awaiting_reason = True
                return f"Merci. Quelle est la raison du retour pour '{self.product_name}' ? (Exemple : Défectueux)"
            else:
                return f"❌ Produit non reconnu. Merci de préciser un produit valide de votre commande : {', '.join(VALID_PRODUCTS)}."

        if self.awaiting_reason:
            category = match_reason(text)
            if category:
                self.awaiting_reason = False
                return (
                    f"✅ Votre demande de retour pour '{self.product_name}' de la commande '{self.order_number}' "
                    f"est enregistrée (Motif : {category}). Vous recevrez un email avec l'étiquette de retour sous peu."
                )
            else:
                motifs = ", ".join(REASON_CATEGORIES.keys())
                return f"❌ Motif non reconnu. Merci de choisir un motif parmi : {motifs}."

        doc = nlp(text)
        best = max(doc.cats, key=doc.cats.get) if doc.cats else None
        if best and doc.cats[best] > 0.5:
            return self.handle_intention(best)

        fuzzy = fuzzy_match(text, KEYWORDS.keys())
        if fuzzy:
            return self.handle_intention(fuzzy)

        return "Je n'ai pas compris votre demande. Pouvez-vous reformuler ?"

    def handle_intention(self, intent):
        if intent == "SALUTATION":
            return "Bonjour 👋 ! Comment puis-je vous aider concernant votre commande MAGUINO ?"
        if intent == "DEMANDE_RETOUR":
            self.awaiting_order = True
            return "Très bien. Pour commencer, merci de me donner votre numéro de commande au format CMD-XXXXXX."
        if intent == "INFOS_RETOUR":
            return "Vous pouvez retourner un produit sous 30 jours. Donnez-moi votre numéro de commande au format CMD-XXXXXX pour commencer."
        if intent == "REMBOURSEMENT":
            return "Les remboursements sont effectués après réception et vérification du produit retourné, sous 5 à 7 jours ouvrés."
        if intent == "SUIVI_RETOUR":
            return "Pour suivre votre retour, consultez votre espace client ou donnez-moi votre numéro de commande au format CMD-XXXXXX."
        if intent == "QUESTION_GENERALE":
            return "Je suis là pour vous aider sur vos retours et remboursements. Que souhaitez-vous savoir ?"
        return "Je n'ai pas compris votre demande. Pouvez-vous reformuler ?"
