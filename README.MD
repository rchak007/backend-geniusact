# GeniusAct — AI Agent Backend

FastAPI + LangChain AI agent that powers the GeniusAct chat assistant.
Handles business inquiries, fee calculations, and Calendly booking via WebSocket.

## Tech Stack

- **FastAPI** — WebSocket API server
- **LangChain** — AI agent framework
- **OpenAI GPT-4o-mini** — Language model
- **Python 3.11**

## Agent Tools

| Tool | Description |
|------|-------------|
| `get_product_info` | Returns info about GeniusAct features, pricing, crypto/Stripe/PayPal |
| `get_booking_link` | Returns Calendly link for demo/consultation booking |
| `get_fee_savings` | Calculates exact fee savings vs Stripe/PayPal given a monthly volume |
| `get_demo_link` | Links user to the live checkout demo |

## Local Setup

### 1. Clone & create environment
```bash
git clone https://github.com/rchak007/backend-geniusact.git
cd backend-geniusact

conda create -n geniusAct python=3.11 -y
conda activate geniusAct

pip install -r requirements.txt
```

### 2. Create `.env` file
```bash
cp .env.example .env
# Edit .env and add your keys
```

```env
OPENAI_API_KEY=sk-your-openai-key-here
CALENDLY_LINK=https://calendly.com/geniusact
```

### 3. Run the server
```bash
uvicorn main:app --reload --port 8000
```

### 4. Test it
Open browser: `http://localhost:8000/health`
Should return: `{"status":"ok"}`

## API

### WebSocket Chat
```
ws://localhost:8000/chat/{session_id}
```

- Send: plain text message from user
- Receive: JSON `{"type": "message", "content": "..."}`

### Health Check
```
GET /health
```

## Project Structure

```
backend-geniusact/
├── main.py              ← FastAPI app + LangChain agent + tools
├── requirements.txt     ← Python dependencies
├── .env.example         ← Environment variable template
├── .gitignore
└── README.md
```

## Frontend

The React chat widget lives in the frontend repo:
👉 [frontend-geniusact](https://github.com/zohaibTariq6533/frontend-geniusact)

## Deployment

Backend is deployed on **Render.com** (free tier).
Frontend is deployed on **GoDaddy** shared hosting.

```
keep-empowering.com (GoDaddy)
        ↓ WebSocket
backend on Render.com
        ↓
OpenAI API
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Your OpenAI API key |
| `CALENDLY_LINK` | Your Calendly booking URL |

> ⚠️ Never commit `.env` — it's in `.gitignore`