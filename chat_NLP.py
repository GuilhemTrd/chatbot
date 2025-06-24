import spacy
from fuzzywuzzy import process
import re
import unicodedata

# Initialiser SpaCy
nlp = spacy.blank("fr")
textcat = nlp.add_pipe("textcat")

# Intentions
INTENTIONS = [
    "SALUTATION", "DEMANDE_RETOUR", "INFOS_RETOUR", "REMBOURSEMENT",
    "SUIVI_RETOUR", "QUESTION_GENERALE", "REMERCIEMENT", "RESET"
]

for label in INTENTIONS:
    textcat.add_label(label)

# Entraînement
train_data = [
    ("bonjour", {"cats": {"SALUTATION": 1}}),
    ("salut", {"cats": {"SALUTATION": 1}}),
    ("merci", {"cats": {"REMERCIEMENT": 1}}),
    ("je veux retourner un produit", {"cats": {"DEMANDE_RETOUR": 1}}),
    ("je veux faire un retour", {"cats": {"DEMANDE_RETOUR": 1}}),
    ("comment faire un retour ?", {"cats": {"INFOS_RETOUR": 1}}),
    ("remboursement", {"cats": {"REMBOURSEMENT": 1}}),
    ("suivi retour", {"cats": {"SUIVI_RETOUR": 1}}),
    ("recommencer", {"cats": {"RESET": 1}})
]

optimizer = nlp.initialize()
for _ in range(10):
    for text, ann in train_data:
        ex = spacy.training.Example.from_dict(nlp.make_doc(text), ann)
        nlp.update([ex], sgd=optimizer)

# Config
KEYWORDS = {
    "SALUTATION": ["bonjour", "salut"],
    "REMERCIEMENT": ["merci", "thanks"],
    "DEMANDE_RETOUR": ["retour", "retourner", "renvoyer"],
    "INFOS_RETOUR": ["comment", "procédure"],
    "REMBOURSEMENT": ["remboursement", "rembourser"],
    "SUIVI_RETOUR": ["suivi", "avancement"],
    "QUESTION_GENERALE": ["question", "aide"],
    "RESET": ["reset", "recommencer", "réinitialiser"]
}

VALID_PRODUCTS = ["Casque Bluetooth", "Chargeur USB", "Coque iPhone", "Souris sans fil"]
REASON_CATEGORIES = {
    "Défectueux": ["cassé", "abîmé", "ne fonctionne pas", "endommagé", "brisé", "il est arrivé cassé"],
    "Taille incorrecte": ["pas la bonne taille", "trop grand", "trop petit"],
    "Mauvaise couleur": ["mauvaise couleur", "pas la bonne couleur"],
    "Ne correspond pas à la description": ["ne correspond pas", "différent de la description"]
}

# Outils fuzzy améliorés
def normalize_text(text):
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )
    return text.lower()

def fuzzy_search_in_phrase(text, options):
    words = [w for w in text.split() if len(w) > 2]
    best_match = None
    best_score = 0

    for word in words:
        match, score = process.extractOne(normalize_text(word), [normalize_text(opt) for opt in options])
        if score > best_score:
            best_match = match
            best_score = score

    if best_match and best_score >= 80:
        for opt in options:
            if normalize_text(opt) == best_match:
                return opt

    match, score = process.extractOne(normalize_text(text), [normalize_text(opt) for opt in options])
    if match and score >= 80:
        for opt in options:
            if normalize_text(opt) == match:
                return opt

    return None

def fuzzy_match_reason(text):
    flat_list = []
    mapping = {}
    for cat, phrases in REASON_CATEGORIES.items():
        for phrase in phrases:
            flat_list.append(phrase)
            mapping[phrase] = cat
    match, score = process.extractOne(normalize_text(text), [normalize_text(p) for p in flat_list])
    if match and score >= 60:
        for p in flat_list:
            if normalize_text(p) == match:
                return mapping[p]
    return None

def extract_cmd(text):
    """
    Recherche un CMD-XXXXXX valide dans le texte, même si entouré de texte.
    """
    matches = re.findall(r'\bCMD-\d{6}\b', text)
    return matches[0] if matches else None

# Classe principale
class ChatbotNLP:
    def __init__(self):
        self.reset_state()

    def reset_state(self):
        self.awaiting_order = False
        self.awaiting_product = False
        self.awaiting_reason = False
        self.order_number = None
        self.product_name = None

    def process_input(self, text):
        text = text.strip()

        if text.upper() == "RESET":
            return self.handle_intention("RESET")

        if self.awaiting_order:
            cmd = extract_cmd(text)
            if cmd:
                self.order_number = cmd
                self.awaiting_order = False
                self.awaiting_product = True
                return f"Merci. Quel produit de la commande {cmd} souhaitez-vous retourner ? (Exemple : Casque Bluetooth)"
            return "❌ Le numéro de commande est invalide. Il doit être au format exact CMD-XXXXXX. Veuillez réessayer."

        if self.awaiting_product:
            match = fuzzy_search_in_phrase(text, VALID_PRODUCTS)
            if match:
                self.product_name = match
                self.awaiting_product = False
                self.awaiting_reason = True
                return f"Merci. Quelle est la raison du retour pour '{match}' ? (Exemple : Défectueux)"
            return f"❌ Produit non reconnu. Produits valides : {', '.join(VALID_PRODUCTS)}."

        if self.awaiting_reason:
            category = fuzzy_match_reason(text)
            if category:
                self.awaiting_reason = False
                return (
                    f"✅ Retour enregistré : '{self.product_name}' de la commande '{self.order_number}' "
                    f"(Motif : {category}). Vous recevrez votre étiquette de retour par email."
                )
            motifs = ", ".join(REASON_CATEGORIES.keys())
            return f"❌ Motif non reconnu. Choisissez parmi : {motifs}."

        doc = nlp(text)
        best = max(doc.cats, key=doc.cats.get) if doc.cats else None
        if best and doc.cats[best] > 0.5:
            return self.handle_intention(best)

        fuzzy = fuzzy_search_in_phrase(text, KEYWORDS.keys())
        if fuzzy:
            return self.handle_intention(fuzzy)

        return (
            "Je n'ai pas compris votre demande. "
            "Pouvez-vous reformuler ou consulter notre aide en ligne : https://www.maguino.com/aide-retour"
        )

    def handle_intention(self, intent):
        if intent == "SALUTATION":
            return "Bonjour 👋 ! Comment puis-je vous aider pour votre retour produit ?"
        if intent == "REMERCIEMENT":
            return "Avec plaisir ! Si vous avez une autre demande, je suis là. 😊"
        if intent == "DEMANDE_RETOUR":
            self.reset_state()
            self.awaiting_order = True
            return "Très bien. Merci de me donner votre numéro de commande au format CMD-XXXXXX."
        if intent == "INFOS_RETOUR":
            return "Vous pouvez retourner un produit sous 30 jours. Donnez-moi votre numéro de commande au format CMD-XXXXXX pour commencer."
        if intent == "REMBOURSEMENT":
            return "Les remboursements sont effectués après réception et vérification du produit retourné sous 5 à 7 jours ouvrés."
        if intent == "SUIVI_RETOUR":
            return "Pour suivre votre retour, consultez votre espace client ou donnez-moi votre numéro de commande au format CMD-XXXXXX."
        if intent == "QUESTION_GENERALE":
            return "Je suis là pour vous aider sur vos retours et remboursements. Que souhaitez-vous savoir ?"
        if intent == "RESET":
            self.reset_state()
            return "✅ La conversation a été réinitialisée. Bonjour 👋 ! Comment puis-je vous aider pour votre retour produit ?"
        return "Je n'ai pas compris votre demande. Pouvez-vous reformuler ?"
