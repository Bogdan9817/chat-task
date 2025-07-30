from typing import Any, Literal

from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j.chains.graph_qa.cypher_utils import CypherQueryCorrector, Schema
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langgraph.graph import END, START, StateGraph

from neo4j.exceptions import CypherSyntaxError
from store.graph.types import InputState, OutputState, OverallState
from store.graph.prompt import (
    create_guardrails_prompt,
    example_selector,
    text2cypher_prompt,
    validate_cypher_prompt,
    correct_cypher_prompt,
    generate_final_prompt,
    meta_prompt,
)
from store.graph.schema import ValidateCypherOutput, GuardrailsOutput, GraphMeta

NO_RESULTS = "I couldn't find any relevant information in the database"


class KnowledgeGraph:
    def __init__(self, api_key: str, url: str, username: str, password: str) -> None:
        self.graph = Neo4jGraph(
            url=url, username=username, password=password, enhanced_schema=True
        )
        self.knowledge_graph = None
        self.graph.refresh_schema()
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
        self.llm_transformer = LLMGraphTransformer(llm=self.llm)
        self.chain = GraphCypherQAChain.from_llm(
            llm=self.llm,
            graph=self.graph,
            verbose=True,
            allow_dangerous_requests=True,
        )
        self.meta_output = self.prepare_graph_metadata()
        self.setup_graph()

    async def add_documents(self, documents: list[Document]) -> None:
        graph_documents = await self.prepare_graph_documents(documents=documents)
        self.graph.add_graph_documents(graph_documents=graph_documents)
        self.graph.refresh_schema()
        self.meta_output = self.prepare_graph_metadata()

    def prepare_graph_metadata(self) -> GraphMeta:
        meta_chain = meta_prompt | self.llm.with_structured_output(GraphMeta)
        meta_output = meta_chain.invoke({"schema": self.graph.structured_schema})
        return meta_output

    async def prepare_graph_documents(self, documents: list[Document]) -> list[Any]:
        return await self.llm_transformer.aconvert_to_graph_documents(documents)

    def guardrails(self, state: InputState) -> OverallState:
        try:
            guardrails_chain = create_guardrails_prompt(
                self.meta_output.domain_description,
                self.meta_output.domain_label,
                "end",
            ) | self.llm.with_structured_output(GuardrailsOutput)

            guardrails_output = guardrails_chain.invoke(
                {"question": state.get("question")}
            )
            database_records = None
            if guardrails_output.decision == "end":
                database_records = "This questions is not related to particular domain."

            next_action = guardrails_output.decision
            if next_action not in ("astro", "end"):
                next_action = "end"
            return {
                "next_action": next_action,
                "database_records": database_records,
                "steps": ["guardrail"],
            }
        except Exception as e:
            raise e

    def generate_cypher(self, state: OverallState) -> OverallState:
        text2cypher_chain = text2cypher_prompt | self.llm | StrOutputParser()
        NL = "\n"
        fewshot_examples = (NL * 2).join(
            [
                f"Question: {el['question']}{NL}Cypher:{el['query']}"
                for el in example_selector.select_examples(
                    {"question": state.get("question")}
                )
            ]
        )
        generated_cypher = text2cypher_chain.invoke(
            {
                "question": state.get("question"),
                "fewshot_examples": fewshot_examples,
                "schema": self.graph.schema,
            }
        )
        return {"cypher_statement": generated_cypher, "steps": ["generate_cypher"]}

    def validate_cypher(self, state: OverallState) -> OverallState:
        corrector_schema = [
            Schema(el["start"], el["type"], el["end"])
            for el in self.graph.structured_schema.get("relationships")
        ]
        cypher_query_corrector = CypherQueryCorrector(corrector_schema)
        validate_cypher_chain = (
            validate_cypher_prompt
            | self.llm.with_structured_output(ValidateCypherOutput)
        )
        errors = []
        mapping_errors = []
        try:
            self.graph.query(f"EXPLAIN {state.get('cypher_statement')}")
        except CypherSyntaxError as e:
            errors.append(e.message)
        corrected_cypher = cypher_query_corrector(state.get("cypher_statement"))
        if not corrected_cypher:
            errors.append("The generated Cypher statement doesn't fit the graph schema")
        if not corrected_cypher == state.get("cypher_statement"):
            print("Relationship direction was corrected")

        llm_output = validate_cypher_chain.invoke(
            {
                "question": state.get("question"),
                "schema": self.graph.schema,
                "cypher": state.get("cypher_statement"),
            }
        )
        if llm_output.errors:
            errors.extend(llm_output.errors)
        if llm_output.filters:
            for filter in llm_output.filters:
                if (
                    not [
                        prop
                        for prop in self.graph.structured_schema["node_props"][
                            filter.node_label
                        ]
                        if prop["property"] == filter.property_key
                    ][0]["type"]
                    == "STRING"
                ):
                    continue
                mapping = self.graph.query(
                    f"MATCH (n:{filter.node_label}) WHERE toLower(n.`{filter.property_key}`) = toLower($value) RETURN 'yes' LIMIT 1",
                    {"value": filter.property_value},
                )
                if not mapping:
                    print(
                        f"Missing value mapping for {filter.node_label} on property {filter.property_key} with value {filter.property_value}"
                    )
                    mapping_errors.append(
                        f"Missing value mapping for {filter.node_label} on property {filter.property_key} with value {filter.property_value}"
                    )
        if mapping_errors:
            next_action = "end"
        elif errors:
            next_action = "correct_cypher"
        else:
            next_action = "execute_cypher"
        return {
            "next_action": next_action,
            "cypher_statement": corrected_cypher,
            "cypher_errors": errors,
            "steps": ["validate_cypher"],
        }

    def correct_cypher(self, state: OverallState) -> OverallState:
        correct_cypher_chain = correct_cypher_prompt | self.llm | StrOutputParser()
        corrected_cypher = correct_cypher_chain.invoke(
            {
                "question": state.get("question"),
                "errors": state.get("cypher_errors"),
                "cypher": state.get("cypher_statement"),
                "schema": self.graph.schema,
            }
        )
        return {
            "next_action": "validate_cypher",
            "cypher_statement": corrected_cypher,
            "steps": ["correct_cypher"],
        }

    def execute_cypher(self, state: OverallState) -> OverallState:
        records = self.graph.query(state.get("cypher_statement"))
        return {
            "database_records": records if records else NO_RESULTS,
            "next_action": "end",
            "steps": ["execute_cypher"],
        }

    def generate_final_answer(self, state: OverallState) -> OutputState:
        generate_final_chain = generate_final_prompt | self.llm | StrOutputParser()
        final_answer = generate_final_chain.invoke(
            {
                "question": state.get("question"),
                "results": state.get("database_records"),
            }
        )
        return {"answer": final_answer, "steps": ["generate_final_answer"]}

    @staticmethod
    def guardrails_condition(
        state: OverallState,
    ) -> Literal["generate_cypher", "generate_final_answer"]:
        if state.get("next_action") == "end":
            return "generate_final_answer"
        elif state.get("next_action") == "astro":
            return "generate_cypher"

    @staticmethod
    def validate_cypher_condition(
        state: OverallState,
    ) -> Literal["generate_final_answer", "correct_cypher", "execute_cypher"]:
        if state.get("next_action") == "end":
            return "generate_final_answer"
        elif state.get("next_action") == "correct_cypher":
            return "correct_cypher"
        elif state.get("next_action") == "execute_cypher":
            return "execute_cypher"

    def graph_search(self, input: str):
        return self.knowledge_graph.invoke({"question": input})

    def setup_graph(self) -> StateGraph:
        knowledge_graph = StateGraph(OverallState, input=InputState, output=OutputState)
        knowledge_graph.add_node(self.guardrails)
        knowledge_graph.add_node(self.generate_cypher)
        knowledge_graph.add_node(self.validate_cypher)
        knowledge_graph.add_node(self.correct_cypher)
        knowledge_graph.add_node(self.execute_cypher)
        knowledge_graph.add_node(self.generate_final_answer)
        knowledge_graph.add_edge(START, "guardrails")
        knowledge_graph.add_conditional_edges(
            "guardrails",
            self.guardrails_condition,
        )
        knowledge_graph.add_edge("generate_cypher", "validate_cypher")
        knowledge_graph.add_conditional_edges(
            "validate_cypher",
            self.validate_cypher_condition,
        )
        knowledge_graph.add_edge("execute_cypher", "generate_final_answer")
        knowledge_graph.add_edge("correct_cypher", "validate_cypher")
        knowledge_graph.add_edge("generate_final_answer", END)
        self.knowledge_graph = knowledge_graph.compile()
