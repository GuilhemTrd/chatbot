import spacy
from fuzzywuzzy import process
import re

nlp = spacy.blank("fr")
textcat = nlp.add_pipe("textcat")

INTENTIONS = ["SALUTATION", "DEMANDE_RETOUR", "INFOS_RETOUR", "REMBOURSEMENT", "SUIVI_RETOUR", "QUESTION_GENERALE"]

for label in INTENTIONS:
    textcat.add_label(label)

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

KEYWORDS = {
    "SALUTATION": ["bonjour", "salut"],
    "DEMANDE_RETOUR": ["retour", "renvoyer", "retourner"],
    "INFOS_RETOUR": ["comment", "procédure", "faire retour"],
    "REMBOURSEMENT": ["remboursement", "rembourser"],
    "SUIVI_RETOUR": ["suivi", "où", "avancement"],
    "QUESTION_GENERALE": ["question", "aide", "besoin"]
}

def fuzzy_match(text):
    best_score = 0
    best_label = None
    for label, words in KEYWORDS.items():
        match, score = process.extractOne(text, words)
        if score > best_score:
            best_score = score
            best_label = label
    return best_label if best_score > 70 else None

def is_valid_order_number(text):
    """Vérifie si le texte correspond à CMD- suivi de 6 chiffres"""
    return bool(re.match(r'^CMD-\d{6}$', text.strip(), re.IGNORECASE))

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
            if is_valid_order_number(text):
                self.order_number = text.upper()
                self.awaiting_order = False
                self.awaiting_product = True
                return f"Merci. Quel produit de la commande {self.order_number} souhaitez-vous retourner ?"
            else:
                return "❌ Le numéro de commande est invalide. Il doit être au format CMD-XXXXXX (6 chiffres). Veuillez réessayer."

        if self.awaiting_product:
            self.product_name = text
            self.awaiting_product = False
            self.awaiting_reason = True
            return f"Merci. Quelle est la raison du retour pour {self.product_name} ?"

        if self.awaiting_reason:
            reason = text
            self.awaiting_reason = False
            return (
                f"✅ Votre demande de retour pour le produit '{self.product_name}' de la commande '{self.order_number}' "
                f"a bien été enregistrée pour le motif : '{reason}'. Vous recevrez un email avec l'étiquette de retour sous peu."
            )

        doc = nlp(text)
        best = max(doc.cats, key=doc.cats.get) if doc.cats else None
        if best and doc.cats[best] > 0.5:
            return self.handle_intention(best)

        fuzzy = fuzzy_match(text)
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
            return "Vous pouvez retourner un produit sous 30 jours. Pour commencer un retour, donnez-moi votre numéro de commande au format CMD-XXXXXX."
        if intent == "REMBOURSEMENT":
            return "Les remboursements sont effectués après réception et vérification du produit retourné, sous 5 à 7 jours ouvrés."
        if intent == "SUIVI_RETOUR":
            return "Pour suivre votre retour, consultez votre espace client ou donnez-moi votre numéro de commande au format CMD-XXXXXX."
        if intent == "QUESTION_GENERALE":
            return "Je suis là pour vous aider sur vos retours et remboursements. Que souhaitez-vous savoir ?"
        return "Je n'ai pas compris votre demande. Pouvez-vous reformuler ?"
