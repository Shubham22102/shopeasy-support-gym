---
title: ShopEasy Customer Support Resolution Gym
emoji: 🛒
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - customer-support
  - llm-agent
---

# 🛒 ShopEasy Customer Support Resolution Gym

A **high-fidelity RL environment** for training and evaluating LLM agents on real-world customer support resolution tasks. Built for the [Meta PyTorch OpenEnv Hackathon](https://github.com/meta-pytorch/OpenEnv).

## Overview

ShopEasy Support Gym simulates a realistic e-commerce customer support desk. An AI agent must resolve customer issues by calling backend tools (order lookup, refund processing, knowledge-base search, escalation) and communicating naturally with the customer — all while following company policies.

### Why This Environment?

Customer support resolution is a genuine, high-value real-world task where:
- **Information asymmetry** forces the agent to use tools to discover facts
- **Policy compliance** requires learned knowledge (return windows, fraud detection)
- **Emotional intelligence** is needed to de-escalate upset customers
- **Multi-turn reasoning** spanning tool calls + dialogue is required

---

## Observation Space

`SupportObservation` is returned after every `reset()` and `step()`:

| Field | Type | Description |
|---|---|---|
| `customer_message` | `str` | Latest customer utterance |
| `customer_sentiment` | `calm\|frustrated\|angry\|satisfied` | Inferred emotional state |
| `tool_result` | `dict \| None` | Structured result of last tool call |
| `tool_error` | `str \| None` | Error string if last tool call failed |
| `ticket_id` | `str` | Unique ID for this ticket (`TKT-XXXXX`) |
| `ticket_status` | `open\|pending_info\|resolved\|escalated` | Lifecycle status |
| `issue_type` | `str` | Category (e.g. `refund_request`, `delivery_issue`) |
| `task_id` | `str` | Scenario ID (e.g. `simple_refund`) |
| `difficulty` | `easy\|medium\|hard` | Episode difficulty |
| `verified_facts` | `dict` | Key facts confirmed via tools |
| `conversation_history` | `list[dict]` | Full dialogue so far |
| `step_count` | `int` | Steps taken |
| `max_steps` | `int` | Maximum steps for this episode |
| `steps_remaining` | `int` | `max_steps - step_count` |
| `reward_breakdown` | `dict \| None` | Populated at episode end |
| `done` | `bool` | `True` when episode is over |
| `reward` | `float` | Cumulative reward (0.0–1.0) |

---

## Action Space

`SupportAction` has three modes — the agent picks exactly one per step:

### Tool Call
```json
{"action_type": "tool_call", "tool_name": "lookup_order", "tool_args": {"order_id": "SE-1042"}}
```

### Send Message to Customer
```json
{"action_type": "send_message", "message": "I've processed your refund of ₹2,499. It will appear within 3–5 business days."}
```

### Close Ticket
```json
{"action_type": "close_ticket", "resolution": "resolved"}
```

### Available Tools

| Tool | Args | Description |
|---|---|---|
| `lookup_order` | `order_id` | Retrieve full order details |
| `process_refund` | `order_id, reason, amount` | Issue a refund |
| `search_kb` | `query` | Search the policy knowledge base |
| `escalate_to_human` | `reason, priority` | Escalate ticket to human supervisor |
| `cancel_subscription` | `subscription_id, reason` | Cancel a customer subscription |
| `check_payment` | `order_id` | Verify payment / detect duplicate charges |

---

## Tasks

12 structured tasks spanning 3 difficulty tiers:

### Easy (max 10 steps)
| ID | Description |
|---|---|
| `simple_refund` | Customer requests refund within return window |
| `delivery_tracking` | Customer asks where their order is |
| `kb_policy_question` | Customer asks about return policy |
| `cancellation_request` | Customer wants to cancel a pending order |

### Medium (max 15 steps)
| ID | Description |
|---|---|
| `expired_return` | Return request outside the policy window — must offer store credit |
| `wrong_item_sent` | Customer received incorrect product |
| `duplicate_charge` | Customer was charged twice for same order |
| `partial_order` | Customer only received part of their order |

### Hard (max 20 steps)
| ID | Description |
|---|---|
| `damaged_item` | Item arrived broken — instant refund policy applies |
| `angry_customer` | Extremely angry customer requiring de-escalation |
| `fraud_risk` | Suspicious refund request — must escalate, not refund |
| `vip_warranty_claim` | VIP customer with electronics warranty claim |

---

## Reward Function

3-tier reward (0.0–1.0 total):

| Component | Max | Criteria |
|---|---|---|
| **Outcome** | 0.50 | Correct resolution type (refund/escalate/credit/close) |
| **Process** | 0.30 | Tool usage discipline, policy compliance, empathy |
| **Efficiency** | 0.20 | Steps-to-resolution ratio |

Key policies:
- **ALWAYS** call `lookup_order` before any refund promise
- **ALWAYS** call `search_kb` before citing policy
- **NEVER** process a refund on a fraud-risk order — escalate instead
- Expired return window → offer **store credit**, not full refund
- Damaged items → **instant full refund** regardless of return window

---

## Baseline Performance

Measured with `gpt-4o-mini` as the baseline agent:

| Difficulty | Avg Score | Success Rate |
|---|---|---|
| Easy | ~0.75 | ~85% |
| Medium | ~0.55 | ~60% |
| Hard | ~0.35 | ~35% |

Run your own baseline:
```bash
# Start server
PYTHONPATH=. uvicorn Customer_Support_Gym_2.server.app:app --port 8000

# Run inference (in another terminal)
TASK_ID=simple_refund python inference.py
```

---

## Quick Start

### Option 1: Local Server

```bash
# Clone and set up
git clone https://huggingface.co/spaces/<your-username>/shopeasy-support-gym
cd shopeasy-support-gym

# Create virtual env and install
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Copy .env and add your keys
cp .env.example .env   # Then edit .env with your API keys

# Start the server
uvicorn Customer_Support_Gym_2.server.app:app --port 8000 --reload

# In another terminal, run the baseline agent
python inference.py
```

### Option 2: Docker

```bash
docker build -t shopeasy-support-gym:latest .
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=<your-key> \
  -e MODEL_NAME=gpt-4o-mini \
  shopeasy-support-gym:latest
```

### Option 3: Python Client

```python
from Customer_Support_Gym_2.client import CustomerSupportGym2Env
from Customer_Support_Gym_2.models import SupportAction

env = CustomerSupportGym2Env(base_url="http://localhost:8000")

# Start easy episode
result = env.reset(task_id="simple_refund")
obs = result.observation
print(f"Customer: {obs.customer_message}")

# Agent looks up the order
action = SupportAction(
    action_type="tool_call",
    tool_name="lookup_order",
    tool_args={"order_id": "SE-1042"},
)
result = env.step(action)
print(f"Tool result: {result.observation.tool_result}")

# Agent responds to customer
action = SupportAction(
    action_type="send_message",
    message="I can see your order is within the return window. I'll process your full refund now.",
)
result = env.step(action)

# Close the ticket
action = SupportAction(action_type="close_ticket", resolution="resolved")
result = env.step(action)
print(f"Final reward: {result.reward}")
print(f"Reward breakdown: {result.observation.reward_breakdown}")

env.close()
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/reset` | Start new episode |
| `POST` | `/step` | Execute one action |
| `GET` | `/state` | Get current episode state |
| `GET` | `/schema` | Get action/observation JSON schemas |
| `WS` | `/ws` | Persistent WebSocket session |
| `GET` | `/web` | Interactive web UI |
| `GET` | `/docs` | OpenAPI documentation |

### Reset Request

```json
{
  "task_id": "simple_refund",
  "difficulty": "easy",
  "seed": 42
}
```

### Step Request

```json
{
  "action": {
    "action_type": "tool_call",
    "tool_name": "lookup_order",
    "tool_args": {"order_id": "SE-1042"}
  }
}
```

---

## Project Structure

```
shopeasy-support-gym/
├── openenv.yaml                   # OpenEnv manifest
├── pyproject.toml                 # Package metadata
├── Dockerfile                     # Container definition
├── inference.py                   # Baseline inference script
├── client.py                      # Python client for the env
├── models.py                      # SupportAction + SupportObservation
├── __init__.py                    # Package exports
└── server/
    ├── app.py                     # FastAPI application
    ├── Customer_Support_Gym_2_environment.py  # Core env logic
    ├── data/
    │   ├── orders.py              # 100 synthetic orders DB
    │   ├── customers.py           # Customer persona engine
    │   ├── scenarios.py           # 12 scenario definitions
    │   └── knowledge_base.py     # Policy KB (search_kb tool)
    └── engine/
        ├── tools.py               # Tool executor (6 tools)
        ├── policy_engine.py       # Refund policy rules engine
        └── reward.py              # 3-tier reward calculator
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | Yes | — | OpenAI API key (or any compatible provider) |
| `HF_TOKEN` | Yes | — | Hugging Face token (for openenv push) |
| `API_BASE_URL` | No | `https://api.openai.com/v1` | LLM API base URL |
| `MODEL_NAME` | No | `gpt-4o-mini` | Model identifier |
| `MAX_CONCURRENT_ENVS` | No | `16` | Max parallel RL training sessions |
| `DEFAULT_TASK_ID` | No | *(random)* | Default task if none specified |
| `DEFAULT_DIFFICULTY` | No | *(random)* | Default difficulty filter |
