# GeniusAct — AI Chatbot Setup
## Local Development Guide

---

## FOLDER STRUCTURE YOU'LL HAVE

```
~/GitHubRepos/GeniusAct/
├── frontend-geniusact/        ← your existing React site
└── backend-geniusact/         ← new Python backend (create this)
    ├── main.py
    ├── requirements.txt
    └── .env
```

---

## STEP 1 — Set Up the Backend

```bash
# Create the backend folder next to your frontend
cd ~/GitHubRepos/GeniusAct
mkdir backend-geniusact
cd backend-geniusact
```

Copy `main.py` and `requirements.txt` into this folder, then:

```bash
# Create Python virtual environment
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

Create your `.env` file:
```bash
# backend-geniusact/.env
OPENAI_API_KEY=sk-your-real-key-here
CALENDLY_LINK=https://calendly.com/your-handle   # add when ready
```

Start the backend:
```bash
uvicorn main:app --reload --port 8000
```

✅ You'll see: `Uvicorn running on http://0.0.0.0:8000`
✅ Test it: open http://localhost:8000/health — should return `{"status":"ok"}`

---

## STEP 2 — Set Up the Frontend

### 2a. Add env variable to your React project

```bash
cd ~/GitHubRepos/GeniusAct/frontend-geniusact
```

Create/edit `.env` file (in root of frontend project):
```
VITE_AGENT_WS_URL=ws://localhost:8000/chat
```

Make sure `.env` is in `.gitignore`:
```bash
grep ".env" .gitignore   # should show .env
# If not listed, run:
echo ".env" >> .gitignore
```

### 2b. Copy AgentChat.jsx into your components folder

```bash
cp /path/to/AgentChat.jsx src/components/AgentChat.jsx
```

### 2c. Replace your App.jsx with the new one

```bash
cp /path/to/App.jsx src/App.jsx
```

The ONLY change is 2 lines added:
```jsx
import AgentChat from "./components/AgentChat";   // line added at top
<AgentChat />                                      // line added inside <Router>
```

### 2d. Start your frontend

```bash
npm run dev
```

---

## STEP 3 — Test It

Open http://localhost:5173 — you'll see a blue chat bubble (💬) in the bottom-right corner.

Try these messages:

| You say | Agent should do |
|---------|----------------|
| "What is GeniusAct?" | Explain the product |
| "How much are fees?" | Give fee comparison |
| "I do $5000/month, how much do I save?" | Calculate exact savings |
| "I want to book a demo" | Return Calendly link |
| "Show me how it works" | Link to /cart page |

---

## STEP 4 — Customize the Agent

Open `backend-geniusact/main.py` and update the `get_product_info` tool
with any additional info about GeniusAct (real pricing tiers, contact email, etc.)

---

## STEP 5 — GoDaddy Deployment (when ready)

We'll do this together later, but here's the plan:

```
GoDaddy VPS or similar hosting
├── Frontend: build React → upload dist/ folder
└── Backend:  run FastAPI with uvicorn behind nginx
```

For the backend you'll need either:
- GoDaddy VPS (Linux) → run uvicorn directly
- OR deploy backend to Render/Railway (free tier) → update VITE_AGENT_WS_URL

---

## TROUBLESHOOTING

| Problem | Fix |
|---------|-----|
| Chat bubble doesn't appear | Check AgentChat imported in App.jsx |
| "Connecting..." never resolves | Make sure backend is running on :8000 |
| CORS error in browser console | Check your Vite port (5173 or 5174?) is in main.py CORS list |
| OpenAI error | Check OPENAI_API_KEY in backend .env |
| Module not found | Run pip install inside activated venv |

---

## TWO TERMINALS NEEDED

```
Terminal 1:                         Terminal 2:
cd backend-geniusact                cd frontend-geniusact
source venv/bin/activate            npm run dev
uvicorn main:app --reload --port 8000
```
