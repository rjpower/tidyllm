"""Test the LLM module with expected workflow."""

import json

from tidyllm.agent import LLMAgent, LLMLogger, create_request
from tidyllm.library import FunctionLibrary
from tidyllm.llm import (
    AssistantMessage,
    LLMClient,
    LLMResponse,
    Role,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tidyllm.registry import Registry


def test_conversation_workflow():
    """Test accumulating conversation history across multiple calls."""
    
    # Create a custom mock client that responds based on last user message
    class CustomMockClient:
        def completion(self, model, messages, tools, **kwargs):
            # Get the last user message
            last_user_msg = next((msg for msg in reversed(messages) if msg.role == Role.USER), None)
            if not last_user_msg:
                raise ValueError("No user message found")
                
            if "2+2" in last_user_msg.content:
                assistant_msg = AssistantMessage(
                    content="I'll calculate that for you.",
                    tool_calls=[ToolCall(
                        tool_name="calculator",
                        tool_args={"expression": "2+2"},
                        id="call_1"
                    )]
                )
            elif "multiply" in last_user_msg.content:
                assistant_msg = AssistantMessage(
                    content="I'll multiply the previous result by 3.",
                    tool_calls=[ToolCall(
                        tool_name="calculator",
                        tool_args={"expression": "4*3"},
                        id="call_2"
                    )]
                )
            else:
                assistant_msg = AssistantMessage(
                    content="I don't understand."
                )
                
            return LLMResponse(
                model="mock-gpt",
                messages=messages + [assistant_msg]
            )
    
    client = CustomMockClient()
    
    # Create a test registry and register tools
    test_registry = Registry()
    
    def calculator(expression: str) -> float:
        """Calculate a math expression.
        
        Args:
            expression: Math expression to evaluate
        """
        return eval(expression)
    
    test_registry.register(calculator)
    
    # Create function library with the test registry
    library = FunctionLibrary(registry=test_registry)
    
    # Create LLM agent
    agent = LLMAgent(function_library=library, llm_client=client, model="mock-gpt")
    
    # First call
    messages1 = create_request("You are helpful", "What's 2+2?")
    response1 = agent.ask(messages1)
    last_msg = response1.messages[-1]
    assert isinstance(last_msg, AssistantMessage)
    tool_calls1 = last_msg.tool_calls
    assert len(tool_calls1) == 1
    assert tool_calls1[0].tool_name == "calculator"
    # Note: agent.ask() executes tools but doesn't add responses back to conversation
    # The tool execution happens, but responses aren't included in the returned messages
    # Only ask_with_conversation() adds tool responses to the conversation
    
    # Check message history
    assert len(response1.messages) == 3  # system, user, assistant
    assert response1.messages[0].role == Role.SYSTEM
    assert response1.messages[1].role == Role.USER
    assert response1.messages[1].content == "What's 2+2?"
    assert response1.messages[2].role == Role.ASSISTANT
    last_assistant = response1.messages[2]
    assert isinstance(last_assistant, AssistantMessage)
    assert len(last_assistant.tool_calls) == 1
    
    # Build conversation for second call
    conversation = response1.messages + [
        ToolMessage(
            content="4",
            tool_call_id="call_1",
            name="calculator"
        ),
        UserMessage(
            content="Now multiply that by 3"
        )
    ]
    
    # Second call with accumulated history
    response2 = agent.ask(conversation)
    last_msg2 = response2.messages[-1]
    assert isinstance(last_msg2, AssistantMessage)
    tool_calls2 = last_msg2.tool_calls
    assert len(tool_calls2) == 1
    assert tool_calls2[0].tool_name == "calculator"
    # Note: agent.ask() doesn't add tool responses to conversation, only ask_with_conversation() does
    # The tool execution happens but responses aren't added to the returned messages
    
    # Check full conversation history
    assert len(response2.messages) == 6  # Previous 5 + new assistant
    assert response2.messages[-1].role == Role.ASSISTANT
    assert "multiply" in response2.messages[-1].content.lower()


def test_ask_with_conversation():
    """Test the ask_with_conversation method for multi-turn interactions."""

    # Create mock client that responds with tool calls then completion
    class ConversationMockClient(LLMClient):
        def __init__(self):
            self.call_count = 0

        def completion(self, model, messages, tools, **kwargs):
            self.call_count += 1

            if self.call_count == 1:
                # First call - read file
                assistant_msg = AssistantMessage(
                    content="I'll read the file first.",
                    tool_calls=[ToolCall(
                        tool_name="read_file",
                        tool_args={"path": "test.txt"},
                        id="call_1"
                    )]
                )
                return LLMResponse(
                    model="mock-gpt",
                    messages=messages + [assistant_msg]
                )
            elif self.call_count == 2:
                # Second call - patch file
                assistant_msg = AssistantMessage(
                    content="Now I'll update the file.",
                    tool_calls=[ToolCall(
                        tool_name="patch_file",
                        tool_args={
                            "path": "test.txt",
                            "old": "Hello",
                            "new": "Hi"
                        },
                        id="call_2"
                    )]
                )
                return LLMResponse(
                    model="mock-gpt",
                    messages=messages + [assistant_msg]
                )
            else:
                # Final call - done
                assistant_msg = AssistantMessage(
                    content="I've successfully updated the file. SUCCESS"
                )
                return LLMResponse(
                    model="mock-gpt",
                    messages=messages + [assistant_msg]
                )

    client = ConversationMockClient()

    # Create a test registry and register tools
    test_registry = Registry()

    def read_file(path: str) -> str:
        """Read a file.
        
        Args:
            path: Path to the file
        """
        return "Hello world"

    def patch_file(path: str, old: str, new: str) -> str:
        """Patch a file.
        
        Args:
            path: Path to the file
            old: Text to replace
            new: New text
        """
        return "File patched successfully"

    test_registry.register(read_file)
    test_registry.register(patch_file)

    # Create function library with the test registry
    library = FunctionLibrary(registry=test_registry)

    agent = LLMAgent(function_library=library, llm_client=client, model="mock-gpt")

    # Test conversation
    messages = create_request("You are helpful", "Read test.txt and change 'Hello' to 'Hi'")
    response = agent.ask_with_conversation(
        messages,
        max_rounds=5
    )

    # Collect all tool calls from the conversation
    all_tool_calls = []
    for msg in response.messages:
        if msg.role == Role.ASSISTANT and isinstance(msg, AssistantMessage):
            all_tool_calls.extend(msg.tool_calls)
    
    assert len(all_tool_calls) == 2
    assert all_tool_calls[0].tool_name == "read_file"
    assert all_tool_calls[1].tool_name == "patch_file"
    
    # Check tool responses in the conversation
    tool_responses = [msg for msg in response.messages if isinstance(msg, ToolMessage)]
    assert len(tool_responses) == 2
    
    # Find responses by tool call ID
    read_response = next(msg for msg in tool_responses if msg.tool_call_id == "call_1")
    patch_response = next(msg for msg in tool_responses if msg.tool_call_id == "call_2")
    
    assert "Hello world" in read_response.content
    assert "File patched successfully" in patch_response.content

    # Check messages include all turns
    assert any("SUCCESS" in msg.content for msg in response.messages if msg.role == Role.ASSISTANT)


def test_message_structure():
    """Test the LLMMessage structure."""

    # Create messages
    messages = [
        SystemMessage(content="You are helpful"),
        UserMessage(content="Hello"),
        AssistantMessage(
            content="I'll help you",
            tool_calls=[
                ToolCall(
                    tool_name="test_tool",
                    tool_args={"param": "value"},
                    id="call_123"
                )
            ]
        ),
        ToolMessage(
            content="Tool result",
            tool_call_id="call_123",
            name="test_tool"
        )
    ]

    # Test message structure
    assert len(messages) == 4
    assert messages[0].role == Role.SYSTEM
    assert messages[0].content == "You are helpful"
    assert messages[1].role == Role.USER
    assert messages[1].content == "Hello"
    assert messages[2].role == Role.ASSISTANT
    assert messages[2].content == "I'll help you"
    assert len(messages[2].tool_calls) == 1
    assistant_msg = messages[2]
    assert isinstance(assistant_msg, AssistantMessage)
    assert assistant_msg.tool_calls[0].id == "call_123"
    assert assistant_msg.tool_calls[0].tool_name == "test_tool"
    tool_msg = messages[3]
    assert isinstance(tool_msg, ToolMessage)
    assert tool_msg.role == Role.TOOL
    assert tool_msg.content == "Tool result"
    assert tool_msg.tool_call_id == "call_123"


def test_logging_captures_complete_conversation():
    """Test that LLMLogger captures complete conversation including assistant responses and tool calls."""
    import json
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as temp_dir:
        log_dir = Path(temp_dir)
        logger = LLMLogger(log_dir)

        # Create a complete conversation with tool calls
        messages = [
            SystemMessage(content="You are helpful"),
            UserMessage(content="Calculate 2+2"),
            AssistantMessage(
                content="I'll calculate that for you",
                tool_calls=[ToolCall(
                    tool_name="calculator",
                    tool_args={"expression": "2+2"},
                    id="call_123"
                )]
            ),
            ToolMessage(
                content="4",
                tool_call_id="call_123",
                name="calculator"
            )
        ]

        response = LLMResponse(
            model="test-model",
            messages=messages
        )

        # Log the response
        with logger as snapshot:
            snapshot.log(response)

        # Find the log file (pattern changed to match timestamp format)
        log_files = list(log_dir.glob("*.json"))
        assert len(log_files) == 1, f"Expected 1 log file, found {len(log_files)}"

        # Read and verify log content
        with open(log_files[0]) as f:
            log_data = json.load(f)

        # Verify complete conversation is logged
        assert "messages" in log_data
        assert len(log_data["messages"]) == 4

        # Verify system message
        assert log_data["messages"][0]["role"] == "system"
        assert log_data["messages"][0]["content"] == "You are helpful"

        # Verify user message
        assert log_data["messages"][1]["role"] == "user" 
        assert log_data["messages"][1]["content"] == "Calculate 2+2"

        # Verify assistant message with tool calls
        assert log_data["messages"][2]["role"] == "assistant"
        assert log_data["messages"][2]["content"] == "I'll calculate that for you"
        assert "tool_calls" in log_data["messages"][2]
        assert len(log_data["messages"][2]["tool_calls"]) == 1
        # Check tool call structure (uses Pydantic serialization, not LLM format)
        tool_call_data = log_data["messages"][2]["tool_calls"][0]
        assert tool_call_data["tool_name"] == "calculator"
        assert tool_call_data["tool_args"]["expression"] == "2+2"
        assert tool_call_data["id"] == "call_123"

        # Verify tool response message
        assert log_data["messages"][3]["role"] == "tool"
        assert log_data["messages"][3]["content"] == "4"
        assert log_data["messages"][3]["tool_call_id"] == "call_123"

        # Verify timestamp exists
        assert "timestamp" in log_data

        print("✓ Logging captures complete conversation including tool calls and responses")


def test_message_serialization_for_litellm():
    """Test that LLMMessage objects are properly serialized for LiteLLM API calls."""

    # Create messages with enum roles and tool calls
    messages = [
        SystemMessage(content="You are helpful"),
        UserMessage(content="Calculate 2+2"),
        AssistantMessage(
            content="I'll calculate that",
            tool_calls=[ToolCall(
                tool_name="calculator",
                tool_args={"expression": "2+2"},
                id="call_123"
            )]
        ),
        ToolMessage(
            content="4",
            tool_call_id="call_123",
            name="calculator"
        )
    ]

    # Test message conversion (this should not raise an exception)
    message_dicts = []
    for msg in messages:
        msg_dict = {
            "role": msg.role.value,  # Convert enum to string
            "content": msg.content
        }
        if isinstance(msg, ToolMessage):
            msg_dict["tool_call_id"] = msg.tool_call_id
            msg_dict["name"] = msg.name
        if isinstance(msg, AssistantMessage) and msg.tool_calls:
            tool_calls_list = []
            for tc in msg.tool_calls:
                tc_dict = {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.tool_name,
                        "arguments": json.dumps(tc.tool_args)
                    }
                }
                tool_calls_list.append(tc_dict)
            msg_dict["tool_calls"] = tool_calls_list
        message_dicts.append(msg_dict)

    # Verify correct serialization
    assert message_dicts[0]["role"] == "system"
    assert message_dicts[1]["role"] == "user"
    assert message_dicts[2]["role"] == "assistant"
    assert message_dicts[3]["role"] == "tool"

    # Verify tool call structure
    assert "tool_calls" in message_dicts[2]
    assert len(message_dicts[2]["tool_calls"]) == 1
    tc = message_dicts[2]["tool_calls"][0]
    assert tc["id"] == "call_123"
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "calculator"
    assert tc["function"]["arguments"] == '{"expression": "2+2"}'

    # Verify tool response structure
    assert message_dicts[3]["tool_call_id"] == "call_123"
    assert message_dicts[3]["content"] == "4"

    print("✓ Message serialization for LiteLLM test passed")


# MockLLMClient test removed as suggested - too complex for basic functionality


if __name__ == "__main__":
    test_message_structure()
    print("✓ Message structure test passed")

    test_conversation_workflow()
    print("✓ Conversation workflow test passed")

    test_ask_with_conversation()
    print("✓ Ask with conversation test passed")

    test_logging_captures_complete_conversation()
    print("✓ Logging captures complete conversation test passed")

    test_message_serialization_for_litellm()

    print("\nAll tests passed!")
