# Ideas for TidyLLM/HyperBase

# Apps

## Daily News

I read through the NHK daily news and provide a translation, and the AI critiques my pronuciation & translation.
Add the resulting conversation to my daily notes.

## Recipe Suggestion

* Keep a table of recipes, and review each week how each recipe went.
* Crawl recipes from nytimes etc
* Suggest new recipes based on user preferences

## Link Manager

* Fetch hyperlinks from blog entries
* Create new entries for each link, link back to the original article, summarize the web page

## Journal Scanner

* Given the photos on the page
* Scan them into markdown
* Infer a title, add them as a note

# Infrastructure

## "Source" data type

It looks like I'm duplicating functions to handle different input types. Instead we should
have a `Source` data type like:

```
SourceT = TypeVar("SourceT")
class SourceProtocol(Protocol, Generic[SourceT]):
    read() -> list[SourceT] ?

class ByteSource(Protocol):
    read(sz: int) -> bytes
  
SourceLike = bytes | file | SourceProtocol | ByteSource

read_bytes(source: SourceLike):
  pass
```

Again Python strikes with it's stupid inconsistency. bytes != list[byte]

```
Cell In[1], line 1
----> 1 byte

NameError: name 'byte' is not defined
```

## App Environment

We want to have a GUI app environment we run our apps in, based on an Electron shell.
You can install packages from other apps, show GUI elements etc.

## GUI Rendering

We want to have some kind of GUI system. Naively, I think we do this by having the UI
be a Vite app with a specific template: mantine etc etc is pre-installed, and there's a
specific HTTP API setup for the LLM to fetch() requests back to the server.

Don't worry about additional abstraction beyond that to get started, e.g. having a complex
JS side toolkit. This is intended for simple interactions, so we can keep it simple. We'd 
want generated UI code to be stored for fast reference: we don't want the LLM to reconstruct
it every time.

## "Freezing" conversations

What if we had a conversation with the LLM about say my Japanese lesson? I want to be able
to ask the LLM to easily turn that conversation into a mini-app itself. So maybe we describe
the Agent API for the LLM and it can write code for composite agents for us...?


## "Connectors"

Let you connect to e.g. a storage service of some kind.
This requires some kind of secret or configuration management to make work in the long run.
To start we can just forward env vars.

This would give us another reason to write the "source" library.

https://filesystem-spec.readthedocs.io/en/latest/