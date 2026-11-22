"""Safety, entity, homonym, and sensitive-topic guardrails (Must-Have #5, #6 & Hard Cases #1-#4)."""

from dataclasses import dataclass, field
import re
from typing import Dict, List, Optional, Tuple


# Common well-known brands & public figures that must NEVER be AI-generated with logos/faces
KNOWN_BRANDS = {
    "apple", "iphone", "macbook", "google", "alphabet", "microsoft", "amazon", "meta",
    "facebook", "instagram", "whatsapp", "tesla", "spacex", "nvidia", "nike", "adidas",
    "coca-cola", "coke", "pepsi", "starbucks", "mcdonalds", "samsung", "sony", "toyota",
    "bmw", "mercedes", "netflix", "spotify", "uber", "airbnb", "openai", "anthropic",
    "tata", "infosys", "reliance", "jio", "flipkart", "zomato", "swiggy", "paytm",
}

BRAND_GENERIC_VISUALS: Dict[str, str] = {
    "apple": "minimalist modern smartphone on clean wooden desk",
    "iphone": "sleek modern smartphone held in hand close-up screen side",
    "macbook": "modern silver laptop open on minimal desk workspace",
    "google": "person searching on clean web browser screen modern office",
    "microsoft": "modern corporate glass office software team working",
    "amazon": "cardboard shipping boxes moving along warehouse conveyor belt",
    "tesla": "modern electric car charging at sleek station clean street",
    "spacex": "rocket launching into clear twilight sky dramatic wide shot",
    "nvidia": "high tech computer circuit board microchip glowing blue",
    "nike": "athlete running on outdoor track sunrise close-up shoes",
    "adidas": "sportsperson jogging urban park morning light",
    "starbucks": "barista pouring latte art in cozy modern cafe",
    "netflix": "family relaxing on couch watching television living room evening",
    "spotify": "person wearing headphones listening to music city walk",
    "tata": "modern industrial manufacturing plant clean wide shot",
    "infosys": "software engineers collaborating in modern glass tech campus",
    "jio": "5g telecommunications tower against clear sky modern city",
    "zomato": "fresh warm restaurant meal being plated in kitchen",
    "swiggy": "delivery rider riding scooter along urban street",
}

KNOWN_PUBLIC_FIGURES = {
    "elon musk", "narendra modi", "donald trump", "joe biden", "barack obama",
    "sundar pichai", "satya nadella", "tim cook", "mark zuckerberg", "sam altman",
    "jeff bezos", "bill gates", "warren buffett", "virat kohli", "sachin tendulkar",
    "ms dhoni", "taylor swift", "shah rukh khan", "mukesh ambani", "ratan tata",
    "gautam adani", "albert einstein", "mahatma gandhi",
}

# Context-sensitive homonym disambiguation rules (Hard Case #3)
HOMONYM_RULES: Dict[str, List[Tuple[List[str], str, List[str]]]] = {
    "cloud": [
        (["computing", "server", "aws", "azure", "gcp", "data", "saas", "deploy", "storage", "kubernetes", "api", "software", "tech"],
         "modern server room data center glowing racks technology", ["sky", "weather", "cumulus", "rain"]),
        (["sky", "rain", "weather", "storm", "sunset", "airplane", "forecast", "wind"],
         "soft white clouds drifting across open blue sky", ["server", "computer"]),
    ],
    "python": [
        (["code", "coding", "programming", "script", "developer", "django", "fastapi", "ai", "ml", "data", "library", "function", "bug"],
         "software developer writing code on laptop dark IDE screen", ["snake", "reptile", "serpent", "zoo"]),
        (["snake", "jungle", "forest", "reptile", "wildlife", "species", "prey"],
         "wild green python coiled on rainforest tree branch nature", ["code", "laptop"]),
    ],
    "mouse": [
        (["click", "cursor", "computer", "laptop", "desk", "keyboard", "gaming", "scroll", "trackpad"],
         "hand clicking ergonomic computer mouse on modern desk", ["rodent", "cheese", "rat", "pet"]),
        (["cheese", "rodent", "lab", "forest", "cat", "trap", "animal", "pet"],
         "small wild field mouse on natural forest floor", ["computer", "keyboard"]),
    ],
    "bank": [
        (["river", "water", "stream", "shore", "muddy", "erosion", "fishing", "boat", "lake"],
         "peaceful flowing river edge green grassy riverbank nature", ["money", "finance", "atm", "vault"]),
        (["account", "loan", "money", "saving", "deposit", "credit", "interest", "finance", "mortgage", "atm", "invest", "currency"],
         "person reviewing personal finance budget banking app on laptop", ["river", "lake", "shore"]),
    ],
    "value": [
        (["customer", "trust", "ethics", "family", "principle", "mission", "brand", "meaning", "core", "team", "long-term", "quality"],
         "artisans and professionals collaborating carefully on quality craft", ["pile of cash", "dollar bills"]),
        (["saving", "budget", "cost", "price", "roi", "portfolio", "stock", "market", "return", "worth", "money"],
         "clean financial growth chart on tablet calm modern workspace", ["cash pile"]),
    ],
}

SENSITIVE_KEYWORDS = {
    "died", "death", "dead", "killed", "tragedy", "tragic", "mourning", "funeral", "grief",
    "disaster", "earthquake", "flood", "cyclone", "famine", "casualties", "victims",
    "war", "attack", "crash", "collapse", "poverty", "crisis", "loss", "memorial",
    "passed away", "condolences", "devastating", "hospital", "cancer", "illness",
    "दुखद", "निधन", "हादसा", "बाढ़", "शोक", "மரணம்", "துயரம்", "விபத்து", "வெள்ளம்",
}

STAT_PATTERN = re.compile(
    r"(\b\d+(?:\.\d+)?\s*%|\$\s*\d+(?:[.,]\d+)?\s*[kmbKMB]?|₹\s*\d+(?:[.,]\d+)?|\b\d+(?:\.\d+)?\s*(?:million|billion|trillion|crore|lakh|percent|x|times)\b)",
    re.IGNORECASE,
)

ABSTRACT_CUES = [
    "trust", "earned slowly", "patience", "integrity", "mindset", "discipline",
    "resilience", "leadership", "consistency", "empathy", "freedom", "clarity",
    "culture", "habit", "focus", "courage", "wisdom",
]


@dataclass
class SafetyAnalysis:
    named_people: List[str] = field(default_factory=list)
    named_brands: List[str] = field(default_factory=list)
    is_sensitive: bool = False
    has_statistics: bool = False
    extracted_stat: Optional[str] = None
    is_abstract: bool = False
    homonym_resolutions: List[str] = field(default_factory=list)
    avoid_terms: List[str] = field(default_factory=list)
    block_ai_generation: bool = False
    suggested_visual_override: Optional[str] = None
    negative_keywords: List[str] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def analyze_scene_safety(
    scene_text: str,
    full_script: str,
    avoid_list: List[str],
) -> SafetyAnalysis:
    analysis = SafetyAnalysis(avoid_terms=[a.strip().lower() for a in avoid_list if a.strip()])
    lower_scene = scene_text.lower()
    lower_full = full_script.lower()

    # 1. Named real people detection
    for person in KNOWN_PUBLIC_FIGURES:
        if person in lower_scene:
            analysis.named_people.append(person.title())

    # Heuristic proper-noun person detection (e.g., "Dr. Jane Smith", "CEO Rajesh Kumar")
    titled_matches = re.findall(
        r"\b(?:Dr|Mr|Mrs|Ms|Prof|President|PM|CEO|Minister|Senator|Governor)\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        scene_text,
    )
    for m in titled_matches:
        if m not in analysis.named_people:
            analysis.named_people.append(m)

    if analysis.named_people:
        analysis.block_ai_generation = True
        analysis.flags.append("NAMED_PERSON_NO_AI_FACE")
        analysis.negative_keywords.extend(["portrait", "celebrity", "close-up face"])
        analysis.warnings.append(
            f"Real person mentioned ({', '.join(analysis.named_people)}): blocked AI face generation and used contextual editorial B-roll."
        )

    # 2. Brand detection
    for brand in KNOWN_BRANDS:
        if re.search(rf"\b{re.escape(brand)}\b", lower_scene):
            analysis.named_brands.append(brand.title())
            if not analysis.suggested_visual_override and brand in BRAND_GENERIC_VISUALS:
                analysis.suggested_visual_override = BRAND_GENERIC_VISUALS[brand]

    if analysis.named_brands:
        analysis.block_ai_generation = True
        analysis.flags.append("BRAND_NO_INVENTED_LOGO")
        analysis.negative_keywords.extend(["logo", "trademark", "brand sign"])
        analysis.warnings.append(
            f"Brand mentioned ({', '.join(analysis.named_brands)}): blocked invented brand logos and selected generic category footage."
        )

    # 3. Sensitive / sad topic detection
    if any(kw in lower_scene for kw in SENSITIVE_KEYWORDS):
        analysis.is_sensitive = True
        analysis.flags.append("SENSITIVE_RESPECTFUL_TONE")
        analysis.negative_keywords.extend(["smiling", "party", "celebration", "happy", "laughing", "graphic", "blood"])
        analysis.warnings.append(
            "Sensitive or solemn topic detected: enforced respectful, subdued B-roll and excluded upbeat or graphic footage."
        )

    # 4. Homonym disambiguation using full-script context
    for word, rules in HOMONYM_RULES.items():
        if re.search(rf"\b{word}s?\b", lower_scene):
            best_query = None
            best_neg: List[str] = []
            best_hits = 0
            for ctx_words, resolved_query, negs in rules:
                hits = sum(1 for cw in ctx_words if cw in lower_full)
                if hits > best_hits:
                    best_hits = hits
                    best_query = resolved_query
                    best_neg = negs
            if best_query:
                analysis.homonym_resolutions.append(f"'{word}' -> {best_query}")
                analysis.negative_keywords.extend(best_neg)
                analysis.flags.append(f"HOMONYM_RESOLVED_{word.upper()}")
                if not analysis.suggested_visual_override:
                    analysis.suggested_visual_override = best_query

    # 5. Statistics & numbers detection
    stat_match = STAT_PATTERN.search(scene_text)
    if stat_match:
        analysis.has_statistics = True
        analysis.extracted_stat = stat_match.group(0).strip()
        analysis.flags.append("STATISTIC_DETECTED")

    # 6. Abstract concept detection
    if any(cue in lower_scene for cue in ABSTRACT_CUES):
        analysis.is_abstract = True
        analysis.flags.append("ABSTRACT_CONCEPT_METAPHOR")

    # 7. Avoid list expansion
    for item in analysis.avoid_terms:
        analysis.negative_keywords.append(item)
        if "face" in item:
            analysis.negative_keywords.extend(["portrait", "selfie", "headshot", "close-up person"])
        if "alcohol" in item:
            analysis.negative_keywords.extend(["beer", "wine", "bar", "cocktail", "whiskey", "drinking"])

    analysis.negative_keywords = sorted(set(analysis.negative_keywords))
    return analysis


def violates_avoid_list(candidate_text: str, avoid_terms: List[str], negative_keywords: List[str]) -> Optional[str]:
    """Quick lexical pre-filter before VLM visual check."""
    blob = candidate_text.lower()
    for term in avoid_terms + negative_keywords:
        t = term.strip().lower()
        if not t or len(t) < 3:
            continue
        if re.search(rf"\b{re.escape(t)}\b", blob):
            return t
    return None
