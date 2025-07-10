---
title: "Strategy for Building Data-Oriented LLM Applications"
tags: ["LLM", "Data Model", "Application Development", "Chat Interface", "Productivity Tools", "Software Architecture"]
date: 2023-10-27
---

What are we trying to build? What are the key milestones? Our plan spans a year, not just a day.

# Core Observations and Strategy

## 1. Starting with Observations

A chat interface is interactive and offers a familiar feel, but it is limited in its expressiveness. We have observed how contrived but effective techniques, particularly with coding agents, can leverage [Large Language Models (LLMs)](https://en.wikipedia.org/wiki/Large_language_model) to build impressive things.

## 2. LLM Capabilities

[LLMs](https://en.wikipedia.org/wiki/Large_language_model) excel at generating small, self-contained chunks of code, such as APIs or unit tests. However, they are less effective at building complex infrastructure.

## 3. Data-Oriented Personal Applications

Many applications, particularly for personal use, are small and inherently data-oriented. For instance, rather than building a complex SAT solver, I might need a "recipe solver." Such tools require capabilities like:

*   Adding vocabulary cards to [Anki](https://en.wikipedia.org/wiki/Anki_(software))
*   Discovering new words for learning
*   Practicing reading and obtaining critique from an LLM
*   Logging sessions to history
*   Finding weekly recipes

These are all relatively small applications, without extensive complex logic, and they can be built upon a common substrate encompassing:

*   UI Logic
*   Data management
*   Dependency management

It's crucial to note that to facilitate sharing of functionality, a shared data model is essential. Without a consistent serialization mechanism, especially across different programming languages (e.g., if not everything is [Python](https://en.wikipedia.org/wiki/Python_(programming_language))), it becomes challenging to build generic functionality.

## 4. The Role of a Simple Data Model

A simple, well-defined data model can capture the majority of content we care about. For example, if we define "JSON-W" (an extended [JSON](https://en.wikipedia.org/wiki/JSON) format), it could encompass:

*   Binary data (e.g., tagged with [MIME types](https://en.wikipedia.org/wiki/Media_type))
*   Dates
*   Arrays
*   Tables
*   Streams[^1]

This kind of model allows us to account for most data encountered in real-world scenarios. It enables automatic serialization and analysis of asset values without needing specific code hook-ins.

[^1]: The original notes include "(Streams?)" suggesting this might be a potential future extension or a point of consideration.

### 5. Hypothesis

Using this unified data model allows us to decouple our application logic. While this introduces initial costs related to serialization and boilerplate code, we believe these costs are offset by the common tooling and capabilities it enables over time. Similar to how setting up a [SQL database](https://en.wikipedia.org/wiki/SQL) is initially complex but provides long-term benefits, this foundational investment will pay off.

### 5.1 Ideal World & Current Progress

In an ideal world, we would build a comprehensive ecosystem around these data types, incorporating features like auto-concurrency. Tools like [Polars](https://en.github.com/pola-rs/polars) represent a significant step in this direction. We need to evaluate whether to adopt it now or later, and understand its current limitations.

## 6. Incremental Value and Framework Benefits

We believe there is significant incremental value in building LLM features on top of a "chat" interface. This approach, while leveraging LLMs, might initially seem counter-intuitive compared to direct GUI development.

The "Time-to-App" metric (time to develop a functional application) should be lower with this framework compared to developing individual market-specific applications. The reasons include:

*   **Efficiency**: Is this genuinely faster than developing a non-[Electron](https://en.wikipedia.org/wiki/Electron_(software_framework)) app? This requires further investigation.
*   **Component Reuse**: We need a combination of benefits from productive work and reuse of core components. "Anti-app" tools (general-purpose utilities) can be leveraged across multiple specific applications.
*   **Simplification/Decoupling**: The structured approach helps ensure maintainability of our applications.

## 7. The Chat Interface as a Starting Point

The logical starting point for building these applications is a chat interface, rather than a full [GUI](https://en.wikipedia.org/wiki/Graphical_user_interface) environment. We want to build on existing and useful interfaces.

### Why Chat?

*   **Inspection of Intermediate Results**: A chat interface implicitly allows users to inspect and interact with intermediate results, which is crucial for debugging and understanding LLM behavior.
*   **Session Collapse**: Once a chat session is established, it should be straightforward for the LLM to "collapse" the session into a new, fully functional application, given an appropriate prompt.
*   **Jupyter-like Behavior**: This approach makes the chat interface function somewhat like a variant of [Jupyter Notebooks](https://en.wikipedia.org/wiki/Jupyter_Notebook). Instead of writing every piece of code manually, users interact with the LLM, which can then write or execute code within the environment.

# Recap and Future Steps

## 8. Recap of Our Goals

We are aiming to build a collection of applications that:

*   Can be composed from various tools.
*   Share a common data model (e.g., JSON-W).
*   Can be viewed as addons to a chat experience.
*   Provide each new chat history as an opportunity to create new workflows.

## 9. Open Questions Before We Start

Several key questions need to be addressed:

1.  How does Object Interaction (OI) fit into this flow?
    *   Will we use mini-[Vite](https://vitejs.dev/) applets?
    *   How will they connect via [FastAPI](https://fastapi.tiangolo.com/) to the backend?
2.  How much of the data model do we need to define for each step?
3.  How does our data model integrate with a [SQL database](https://en.wikipedia.org/wiki/SQL)? What is our plan for managing user data?
4.  Is it acceptable, for example, to simply fetch our data classes from a table or query on demand? This seems feasible, albeit potentially tedious to integrate into our data queries. Should we compromise on this aspect?

## 10. Initial Applications

We plan to start with the following applications:

*   **Scan Journal**:
    *   Given physical journal pages.
    *   Scan them into markdown format.
    *   Save as a structured note.
*   **Blog/Weekly Recap**:
    *   Read personal notes.
    *   Recap any marked content.
*   **Anki/Vocab Stuff**:
    *   Automate Anki card creation or vocabulary review.
*   **Recipe Bot**:
    *   Assist in finding and managing recipes.

## 11. Initial Functionality

Regarding core functionality, we will begin by:

*   Leveraging our existing partial attempt at tool registration and serialization.
*   Standardizing this process using [Pydantic](https://en.wikipedia.org/wiki/Pydantic) for robust data validation and settings management.

