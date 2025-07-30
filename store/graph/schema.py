from typing import List, Optional
from pydantic import BaseModel, Field


class GuardrailsOutput(BaseModel):
    decision: str = Field(
        description="Decision on whether the question is in-domain or out-of-domain"
    )


class Property(BaseModel):
    node_label: str = Field(
        description="The label of the node to which this property belongs."
    )
    property_key: str = Field(description="The key of the property being filtered.")
    property_value: str = Field(
        description="The value that the property is being matched against."
    )


class ValidateCypherOutput(BaseModel):
    errors: Optional[List[str]] = Field(
        description="A list of syntax or semantical errors in the Cypher statement. Always explain the discrepancy between schema and Cypher statement"
    )
    filters: Optional[List[Property]] = Field(
        description="A list of property-based filters applied in the Cypher statement."
    )


class GraphMeta(BaseModel):
    domain_description: str = Field(
        description="A description of the domain that the graph represents. Example: 'planets, moons, the Solar System, celestial bodies, planetary science...'"
    )
    domain_label: str = Field(description="The label of the domain node in the graph.")
