#!/usr/bin/env python3
"""Recipe-bot: tools to manage and plan recipes."""


from typing import Iterable

from pydantic import BaseModel, Field

from tidyllm.context import get_tool_context
from tidyllm.llm import completion_with_schema
from tidyllm.registry import register
from tidyllm.types import duration
from tidyllm.types.part import (
    ImagePart,
    TextContentPart,
    is_image_part,
    is_text_content_part,
)


class Ingredient(BaseModel):
    """A recipe ingredient with name and amount."""

    name: str = Field(description="Name of the ingredient")
    amount: str = Field(description="Amount needed (e.g., '2 cups', '1 tsp')")


class Recipe(BaseModel):
    """A complete recipe with ingredients, steps, and timing."""

    title: str = Field(description="Recipe title/name")
    ingredients: list[Ingredient] = Field(description="List of ingredients")
    steps: list[str] = Field(description="Cooking instructions in order")
    time: duration.Duration = Field(description="Total cooking time")
    servings: int | None = Field(default=None, description="Number of servings")


def create_recipe_messages(
    input_page: Iterable[ImagePart | TextContentPart],
) -> list[dict]:
    """Create messages for recipe extraction."""
    system_prompt = """You are a recipe extraction expert. Extract complete recipe information including:
- Recipe title
- All ingredients with precise amounts
- Step-by-step cooking instructions
- Total cooking time (prep + cook time)
- Number of servings if mentioned

Be precise with measurements and include all details."""

    user_content = []

    for part in input_page:
        if is_text_content_part(part):
            user_content.append(
                {
                    "type": "text",
                    "text": f"Extract recipe from this content:\n\n{part.text}",
                }
            )
        elif is_image_part(part):
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{part.base64_bytes.decode()}",
                    },
                }
            )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


@register()
def recipe_extract(
    input_page: Iterable[TextContentPart | ImagePart],
) -> Recipe:
    """Extract a recipe from HTML content.

    HTML may be a list of screenshots or HTML page content.
    """
    recipe = completion_with_schema(
        model=get_tool_context().config.fast_model,
        messages=create_recipe_messages(input_page),
        response_schema=Recipe,
    )
    return recipe
