from langchain_core.prompts import ChatPromptTemplate
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_neo4j import Neo4jVector
from langchain_openai import OpenAIEmbeddings
import os

from dotenv import load_dotenv

load_dotenv()


def create_guardrails_prompt(
    domain_description: str, domain_label: str = "in", out_label: str = "out"
) -> ChatPromptTemplate:
    guardrails_system = f"""
    As an intelligent assistant, your primary objective is to decide whether a given question is related to the following domain: {domain_description}.
    If the question is related to this domain, output "{domain_label}". Otherwise, output "{out_label}".
    Assess the question based on its content and determine if it refers to topics or entities commonly associated with this domain.
    Respond with only one word: "{domain_label}" or "{out_label}".
    """

    guardrails_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", guardrails_system),
            ("human", "{question}"),
        ]
    )
    return guardrails_prompt


examples = [
    {
        "question": "How many planets are in the Solar System?",
        "query": "MATCH (p:Planet) RETURN count(p)",
    },
    {
        "question": "List all the moons of Jupiter.",
        "query": "MATCH (:Planet {name: 'Jupiter'})<-[:ORBITS]-(m:Moon) RETURN m.name",
    },
    {
        "question": "What is the average distance of Earth from the Sun?",
        "query": "MATCH (p:Planet {name: 'Earth'}) RETURN p.distanceFromSun",
    },
    {
        "question": "Which planets have rings?",
        "query": "MATCH (p:Planet) WHERE p.hasRings = true RETURN p.name",
    },
    {
        "question": "List all gas giants in the Solar System.",
        "query": "MATCH (p:Planet) WHERE p.type = 'Gas Giant' RETURN p.name",
    },
    {
        "question": "Which planets have more than 10 moons?",
        "query": """
            MATCH (p:Planet)<-[:ORBITS]-(m:Moon)
            WITH p, COUNT(m) as moonCount
            WHERE moonCount > 10
            RETURN p.name, moonCount
        """,
    },
    {
        "question": "Find the planet with the shortest orbital period.",
        "query": "MATCH (p:Planet) RETURN p.name, p.orbitalPeriod ORDER BY p.orbitalPeriod ASC LIMIT 1",
    },
    {
        "question": "Which celestial bodies orbit Mars?",
        "query": "MATCH (c)-[:ORBITS]->(:Planet {name: 'Mars'}) RETURN c.name, labels(c)",
    },
    {
        "question": "Which space missions visited Saturn?",
        "query": "MATCH (m:Mission)-[:VISITED]->(:Planet {name: 'Saturn'}) RETURN m.name",
    },
    {
        "question": "Which planets have an atmosphere containing oxygen?",
        "query": "MATCH (p:Planet) WHERE 'Oxygen' IN p.atmosphere RETURN p.name",
    },
]

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY")),
    Neo4jVector,
    k=5,
    input_keys=["question"],
)

text2cypher_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "Given an input question, convert it to a Cypher query. No pre-amble."
                "Do not wrap the response in any backticks or anything else. Respond with a Cypher statement only!"
            ),
        ),
        (
            "human",
            (
                """You are a Neo4j expert. Given an input question, create a syntactically correct Cypher query to run.
Do not wrap the response in any backticks or anything else. Respond with a Cypher statement only!
Here is the schema information
{schema}

Below are a number of examples of questions and their corresponding Cypher queries.

{fewshot_examples}

User input: {question}
Cypher query:"""
            ),
        ),
    ]
)

validate_cypher_system = """
You are a Cypher expert reviewing a statement written by a junior developer.
"""

validate_cypher_user = """You must check the following:
* Are there any syntax errors in the Cypher statement?
* Are there any missing or undefined variables in the Cypher statement?
* Are any node labels missing from the schema?
* Are any relationship types missing from the schema?
* Are any of the properties not included in the schema?
* Does the Cypher statement include enough information to answer the question?

Examples of good errors:
* Label (:Foo) does not exist, did you mean (:Bar)?
* Property bar does not exist for label Foo, did you mean baz?
* Relationship FOO does not exist, did you mean FOO_BAR?

Schema:
{schema}

The question is:
{question}

The Cypher statement is:
{cypher}

Make sure you don't make any mistakes!"""

validate_cypher_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            validate_cypher_system,
        ),
        (
            "human",
            (validate_cypher_user),
        ),
    ]
)

correct_cypher_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a Cypher expert reviewing a statement written by a junior developer. "
                "You need to correct the Cypher statement based on the provided errors. No pre-amble."
                "Do not wrap the response in any backticks or anything else. Respond with a Cypher statement only!"
            ),
        ),
        (
            "human",
            (
                """Check for invalid syntax or semantics and return a corrected Cypher statement.

Schema:
{schema}

Note: Do not include any explanations or apologies in your responses.
Do not wrap the response in any backticks or anything else.
Respond with a Cypher statement only!

Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.

The question is:
{question}

The Cypher statement is:
{cypher}

The errors are:
{errors}

Corrected Cypher statement: """
            ),
        ),
    ]
)

generate_final_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant",
        ),
        (
            "human",
            (
                """
                    Use the following results retrieved from a database to provide
                    a succinct, definitive answer to the user's question.

                    Respond as if you are answering the question directly.

                    Results: {results}
                    Question: {question}
                """
            ),
        ),
    ]
)

meta_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert in knowledge graph analysis."),
        (
            "human",
            (
                """
                    Below is a schema of a knowledge graph. The schema consists of node types (labels) and the relationships between them.

                    Your task is to analyze the structure of the graph and infer the domain of knowledge it represents.

                    Please respond in JSON format with:
                    - "domain_label": A short identifier (lowercase, underscore_case) describing the domain (e.g. "movies", "solar_system", "clinical_trials").
                    - "domain_description": A short description (1–2 sentences) that explains what the graph is about.

                    GRAPH SCHEMA:
                    {schema}
                """
            ),
        ),
    ]
)
