import json
from dotenv import load_dotenv
load_dotenv()

from groq import Groq
from pydantic import BaseModel, ValidationError
from typing import Literal


MODEL_NAME = "openai/gpt-oss-20b"

client = Groq()  # reads GROQ_API_KEY from environment


class RouteLabel(BaseModel):
    label: Literal["cnss", "labor_code", "out_of_scope"]


SYSTEM_PROMPT = """Tu es un classificateur de requêtes pour un assistant juridique tunisien.

Classe la question suivante dans exactement une des trois catégories :

cnss = Pensions et prêts CNSS (pension de vieillesse, prêts personnels, prêts logement, prêts voitures, prêts universitaires)
labor_code = Code du travail tunisien (les congés, la rémunération, les heures de travail, les droits et les devoirs de l'employeur/employé, licenciement, contrats de travail)
out_of_scope = Toute autre question (hors sujet, ou sujet social non couvert par les deux catégories ci-dessus, comme le chômage ou les allocations familiales)

Réponds UNIQUEMENT avec un objet JSON de la forme {"label": "..."}, où la valeur est exactement l'une de : "cnss", "labor_code", "out_of_scope". Aucun autre texte."""


def route_query(query: str, max_attempts: int = 2) -> str:
    for attempt in range(max_attempts):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            raw = response.choices[0].message.content
            result = RouteLabel.model_validate_json(raw)
            return result.label
        except Exception as e:
            print(f"Attempt {attempt} failed: {e}")
            continue

    return "out_of_scope"