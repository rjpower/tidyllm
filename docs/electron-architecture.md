---
title: "TidyLLM Electron Architecture & UI Design"
tags: ["Electron", "Architecture", "UI/UX", "WebSockets", "FastAPI"]
date: 2025-01-09
---

# TidyLLM Electron Architecture

## Overview

The TidyLLM Electron shell provides a desktop environment that seamlessly integrates Python backend services with a modern React/Vite frontend, enabling real-time bidirectional communication and dynamic UI generation.

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    Electron Main Process                     │
│  - Window Management                                         │
│  - IPC Bridge                                                │
│  - Python Process Manager                                    │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────┴──────────────────────────────────────────┐
│                  Electron Renderer Process                   │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │                    Vite React App                        │ │
│ │ ┌─────────────┐ ┌──────────────┐ ┌──────────────────┐  │ │
│ │ │ Chat Panel  │ │ Tool Output  │ │ Dynamic Iframes  │  │ │
│ │ │             │ │   Display    │ │   (Applets)      │  │ │
│ │ └─────────────┘ └──────────────┘ └──────────────────┘  │ │
│ │ ┌─────────────────────────────────────────────────────┐ │ │
│ │ │            WebSocket Client Manager                  │ │ │
│ │ └─────────────────────────────────────────────────────┘ │ │
│ └─────────────────────────────────────────────────────────┘ │
└──────────────────┬──────────────────────────────────────────┘
                   │ WebSocket
┌──────────────────┴──────────────────────────────────────────┐
│                    FastAPI Backend                           │
│ ┌─────────────┐ ┌─────────────┐ ┌────────────────────────┐ │
│ │ Tool Router │ │ Chat Router │ │ Dynamic UI Generator   │ │
│ └─────────────┘ └─────────────┘ └────────────────────────┘ │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │              TidyLLM Registry & Context                  │ │
│ └─────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Electron Main Process

```javascript
// main.js
import { app, BrowserWindow, ipcMain } from 'electron';
import { spawn } from 'child_process';
import path from 'path';

class TidyLLMApp {
  constructor() {
    this.mainWindow = null;
    this.pythonProcess = null;
    this.serverPort = 8000;
  }

  async startPythonServer() {
    // Start FastAPI server
    this.pythonProcess = spawn('uv', [
      'run', 
      'python', 
      '-m', 
      'tidyllm.server',
      '--port', 
      this.serverPort.toString()
    ], {
      cwd: app.getAppPath(),
      env: { ...process.env, PYTHONUNBUFFERED: '1' }
    });

    // Wait for server to be ready
    await this.waitForServer();
  }

  async createWindow() {
    this.mainWindow = new BrowserWindow({
      width: 1400,
      height: 900,
      webPreferences: {
        preload: path.join(__dirname, 'preload.js'),
        contextIsolation: true,
        nodeIntegration: false
      }
    });

    // In development, load from Vite dev server
    if (process.env.NODE_ENV === 'development') {
      await this.mainWindow.loadURL('http://localhost:5173');
    } else {
      await this.mainWindow.loadFile('dist/index.html');
    }
  }
}
```

### 2. FastAPI Backend Structure

```python
# tidyllm/server.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
import json

from tidyllm.adapters.fastapi_adapter import create_fastapi_app
from tidyllm.tools.context import ToolContext
from tidyllm.chat import ChatManager
from tidyllm.ui_generator import UIGenerator

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.context = ToolContext()
    app.state.chat_manager = ChatManager()
    app.state.ui_generator = UIGenerator()
    app.state.connections = set()
    yield
    # Shutdown
    for ws in app.state.connections:
        await ws.close()

app = FastAPI(lifespan=lifespan)

# Enable CORS for Electron renderer
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "file://"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount tool endpoints
tool_app = create_fastapi_app(app.state.context)
app.mount("/api/tools", tool_app)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    app.state.connections.add(websocket)
    
    try:
        while True:
            data = await websocket.receive_json()
            response = await handle_message(data, app.state)
            await websocket.send_json(response)
    except WebSocketDisconnect:
        app.state.connections.remove(websocket)

async def handle_message(data: dict, state):
    """Route messages to appropriate handlers"""
    msg_type = data.get("type")
    
    if msg_type == "chat":
        return await state.chat_manager.process_message(data)
    elif msg_type == "tool":
        return await execute_tool(data, state.context)
    elif msg_type == "ui_request":
        return await state.ui_generator.generate_component(data)
    else:
        return {"error": "Unknown message type"}
```

### 3. Frontend React Components

```typescript
// src/App.tsx
import React, { useState, useEffect } from 'react';
import { ChatPanel } from './components/ChatPanel';
import { ToolOutput } from './components/ToolOutput';
import { DynamicFrame } from './components/DynamicFrame';
import { useWebSocket } from './hooks/useWebSocket';

export function App() {
  const { sendMessage, subscribe } = useWebSocket('ws://localhost:8000/ws');
  const [messages, setMessages] = useState([]);
  const [toolOutputs, setToolOutputs] = useState([]);
  const [dynamicComponents, setDynamicComponents] = useState([]);

  useEffect(() => {
    // Subscribe to different message types
    const unsubChat = subscribe('chat_response', (msg) => {
      setMessages(prev => [...prev, msg]);
    });

    const unsubTool = subscribe('tool_output', (output) => {
      setToolOutputs(prev => [...prev, output]);
    });

    const unsubUI = subscribe('ui_component', (component) => {
      setDynamicComponents(prev => [...prev, component]);
    });

    return () => {
      unsubChat();
      unsubTool();
      unsubUI();
    };
  }, [subscribe]);

  return (
    <div className="app-container">
      <div className="main-layout">
        <ChatPanel 
          messages={messages}
          onSendMessage={(msg) => sendMessage({ type: 'chat', content: msg })}
        />
        <ToolOutput outputs={toolOutputs} />
      </div>
      <div className="dynamic-components">
        {dynamicComponents.map((comp, idx) => (
          <DynamicFrame key={idx} component={comp} />
        ))}
      </div>
    </div>
  );
}
```

### 4. WebSocket Hook for Bidirectional Communication

```typescript
// src/hooks/useWebSocket.ts
import { useRef, useCallback, useEffect } from 'react';

export function useWebSocket(url: string) {
  const ws = useRef<WebSocket | null>(null);
  const listeners = useRef(new Map());
  const messageQueue = useRef([]);

  useEffect(() => {
    ws.current = new WebSocket(url);
    
    ws.current.onopen = () => {
      // Send queued messages
      messageQueue.current.forEach(msg => ws.current?.send(JSON.stringify(msg)));
      messageQueue.current = [];
    };

    ws.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      const handlers = listeners.current.get(data.type) || [];
      handlers.forEach(handler => handler(data));
    };

    return () => {
      ws.current?.close();
    };
  }, [url]);

  const sendMessage = useCallback((message: any) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify(message));
    } else {
      messageQueue.current.push(message);
    }
  }, []);

  const subscribe = useCallback((type: string, handler: Function) => {
    if (!listeners.current.has(type)) {
      listeners.current.set(type, []);
    }
    listeners.current.get(type).push(handler);
    
    // Return unsubscribe function
    return () => {
      const handlers = listeners.current.get(type) || [];
      const index = handlers.indexOf(handler);
      if (index > -1) handlers.splice(index, 1);
    };
  }, []);

  return { sendMessage, subscribe };
}
```

### 5. Dynamic UI Generation System

```python
# tidyllm/ui_generator.py
from typing import Dict, Any
import json
from pathlib import Path
import tempfile

class UIGenerator:
    """Generates React components on demand from LLM instructions"""
    
    def __init__(self):
        self.component_cache = {}
        self.component_dir = Path(tempfile.mkdtemp()) / "components"
        self.component_dir.mkdir(exist_ok=True)
    
    async def generate_component(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a new React component based on LLM instructions"""
        
        component_spec = request.get("spec", {})
        component_name = component_spec.get("name", "DynamicComponent")
        
        # Generate component code
        component_code = await self._generate_react_code(component_spec)
        
        # Save to temporary file
        component_path = self.component_dir / f"{component_name}.tsx"
        component_path.write_text(component_code)
        
        # Build component bundle
        bundle_url = await self._build_component(component_path)
        
        return {
            "type": "ui_component",
            "name": component_name,
            "url": bundle_url,
            "props": component_spec.get("props", {}),
            "layout": component_spec.get("layout", {})
        }
    
    async def _generate_react_code(self, spec: Dict[str, Any]) -> str:
        """Generate React component code from specification"""
        
        # This would use an LLM to generate the component
        # For now, returning a template
        return f"""
import React from 'react';
import {{ Card, Button, Input }} from '../ui';

export function {spec['name']}({{ data, onAction }}) {{
    return (
        <Card>
            <h3>{spec.get('title', 'Dynamic Component')}</h3>
            {self._generate_component_body(spec)}
        </Card>
    );
}}
"""
```

### 6. Chat Interface with Tool Integration

```typescript
// src/components/ChatPanel.tsx
import React, { useState, useRef } from 'react';
import { MessageList } from './MessageList';
import { InputBar } from './InputBar';
import { ToolSuggestions } from './ToolSuggestions';

export function ChatPanel({ messages, onSendMessage }) {
  const [input, setInput] = useState('');
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [attachments, setAttachments] = useState([]);

  const handleSend = () => {
    if (input.trim() || attachments.length > 0) {
      onSendMessage({
        text: input,
        attachments: attachments,
        timestamp: new Date().toISOString()
      });
      setInput('');
      setAttachments([]);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const files = Array.from(e.dataTransfer.files);
    // Handle file drops for processing
    setAttachments(prev => [...prev, ...files]);
  };

  return (
    <div className="chat-panel" onDrop={handleDrop} onDragOver={e => e.preventDefault()}>
      <MessageList messages={messages} />
      {showSuggestions && <ToolSuggestions onSelect={tool => setInput(`/${tool} `)} />}
      <InputBar
        value={input}
        onChange={setInput}
        onSend={handleSend}
        attachments={attachments}
        onCommand={() => setShowSuggestions(true)}
      />
    </div>
  );
}
```

### 7. Dynamic Frame System for Applets

```typescript
// src/components/DynamicFrame.tsx
import React, { useRef, useEffect } from 'react';

export function DynamicFrame({ component }) {
  const frameRef = useRef<HTMLIFrameElement>(null);

  useEffect(() => {
    if (frameRef.current && component.url) {
      // Set up message passing with iframe
      const handleMessage = (event) => {
        if (event.source === frameRef.current?.contentWindow) {
          // Handle messages from dynamic component
          if (event.data.type === 'tool_request') {
            // Forward to backend
            sendMessage({
              type: 'tool',
              tool: event.data.tool,
              args: event.data.args
            });
          }
        }
      };

      window.addEventListener('message', handleMessage);
      return () => window.removeEventListener('message', handleMessage);
    }
  }, [component]);

  return (
    <div className="dynamic-frame" style={component.layout}>
      <iframe
        ref={frameRef}
        src={component.url}
        sandbox="allow-scripts allow-same-origin"
        style={{ width: '100%', height: '100%', border: 'none' }}
      />
    </div>
  );
}
```

## Data Flow Patterns

### 1. Chat to Tool Execution
```
User Input → Chat Panel → WebSocket → Chat Manager → Tool Registry → Tool Execution → Response
```

### 2. File Processing
```
Drag & Drop → File Handler → Upload to Backend → Tool Processing → Stream Results → UI Update
```

### 3. Dynamic UI Generation
```
LLM Response → UI Spec → Component Generator → Bundle Build → Iframe Load → Interactive Component
```

## Development Workflow

### Local Development Setup

```bash
# Terminal 1: Start FastAPI backend
uv run python -m tidyllm.server --reload

# Terminal 2: Start Vite dev server
npm run dev

# Terminal 3: Start Electron in dev mode
npm run electron:dev
```

### Building for Production

```bash
# Build frontend
npm run build

# Build Electron app
npm run electron:build

# Package with Python runtime
npm run package
```

## Security Considerations

1. **Iframe Sandboxing**: Dynamic components run in sandboxed iframes
2. **Tool Permissions**: Tools must declare required permissions
3. **Content Security Policy**: Strict CSP for generated content
4. **Local Execution**: All data processing happens locally

## Next Steps

1. Implement WebSocket reconnection logic
2. Add state persistence across sessions
3. Build component marketplace integration
4. Create visual workflow editor
5. Add multi-window support for complex workflows