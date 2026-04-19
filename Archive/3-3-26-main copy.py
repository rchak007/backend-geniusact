"""
GeniusAct — AI Business Agent Backend
======================================
Run: uvicorn main:app --reload --port 8000
"""

import os
import json
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CALENDLY_LINK  = os.getenv("CALENDLY_LINK", "https://calendly.com/geniusact")

# ─────────────────────────────────────────────
# TOOLS — customize these with real info
# ─────────────────────────────────────────────

@tool
def get_product_info(topic: str) -> str:
    """
    Returns information about GeniusAct's product, features, pricing and services.
    Use when users ask about what GeniusAct does, how it works, pricing, or features.
    Args:
        topic: what the user is asking about e.g. 'pricing', 'features', 'crypto', 'paypal'
    """
    info = {
        "about": (
            "GeniusAct is a payment infrastructure platform that lets businesses "
            "accept crypto (USDC stablecoins), credit/debit cards via Stripe, and PayPal "
            "— all from a single checkout. We help businesses cut payment fees by up to 99.7%."
        ),
        "pricing": (
            "GeniusAct charges just 1% per transaction for crypto payments. "
            "Compare that to Stripe and PayPal which charge 2.9% + $0.30 per transaction. "
            "On $10,000/month in sales, you save roughly $190/month switching to crypto payments."
        ),
        "crypto": (
            "We support USDC stablecoin payments on Solana — fees are less than $0.01 per transaction. "
            "USDC is pegged 1:1 to the US dollar so there's no price volatility. "
            "Payments settle instantly to your wallet."
        ),
        "stripe": (
            "Our Stripe integration lets customers pay with any major credit or debit card "
            "(Visa, Mastercard, Amex). It's PCI-compliant with built-in fraud protection. "
            "Just one of three payment options available side-by-side in checkout."
        ),
        "paypal": (
            "Customers can pay with their PayPal account in one tap — no card entry needed. "
            "Great for customers who already trust PayPal. Available alongside crypto and card."
        ),
        "features": (
            "Key features: 1) Crypto checkout (USDC/Solana), 2) Card payments via Stripe, "
            "3) PayPal checkout, 4) All three side-by-side in one checkout flow, "
            "5) Instant settlement, 6) No setup fees, 7) Global payments."
        ),
        "setup": (
            "Setup is simple: connect your crypto wallet, add our checkout to your store, "
            "and you're live. No technical expertise needed. "
            "Visit the /cart page to see a live demo of the checkout."
        ),
    }
    topic_lower = topic.lower()
    for key in info:
        if key in topic_lower:
            return info[key]
    return info["about"] + " " + info["features"]


@tool
def get_booking_link(reason: str = "demo") -> str:
    """
    Returns a link to book a call or demo with the GeniusAct team.
    Use when a user wants to talk to someone, book a demo, get a consultation, or learn more.
    Args:
        reason: why they want to book e.g. 'demo', 'consultation', 'onboarding'
    """
    return (
        f"You can book a {reason} call with our team here: {CALENDLY_LINK} — "
        "pick a time that works for you and we'll walk you through everything!"
    )


@tool
def get_fee_savings(monthly_volume: str) -> str:
    """
    Calculates how much a business would save by switching to GeniusAct crypto payments.
    Use when users ask about savings, cost comparison, or mention their sales volume.
    Args:
        monthly_volume: monthly sales volume mentioned by user e.g. '$5000', '10000'
    """
    try:
        # Strip $ and commas, parse number
        amount = float(monthly_volume.replace("$", "").replace(",", "").strip())
        stripe_fee  = (amount * 0.029) + (30 * 0.30)  # assume ~30 transactions
        crypto_fee  = amount * 0.001
        savings     = stripe_fee - crypto_fee
        return (
            f"On ${amount:,.0f}/month in sales: "
            f"Stripe/PayPal would cost ~${stripe_fee:,.2f}/month in fees. "
            f"With GeniusAct crypto payments: ~${crypto_fee:,.2f}/month. "
            f"You'd save ~${savings:,.2f}/month — that's ${savings*12:,.0f}/year! 🎉"
        )
    except Exception:
        return (
            "With GeniusAct you pay just 0.01% in fees vs 2.9%+$0.30 with Stripe/PayPal. "
            "On $10,000/month that's a saving of ~$280/month. "
            "Tell me your monthly volume and I can give you an exact number!"
        )


@tool
def get_demo_link(query: str = "") -> str:
    """
    Returns a link to the live checkout demo.
    Use when users want to see how it works, try it out, or see a demo.
    """
    return (
        "You can see a live demo of the checkout right now — "
        "just visit the /cart page on this site! "
        "It shows all three payment options: crypto (USDC), card (Stripe), and PayPal side-by-side."
    )


tools = [get_product_info, get_booking_link, get_fee_savings, get_demo_link]

# ─────────────────────────────────────────────
# LLM + AGENT
# ─────────────────────────────────────────────

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=OPENAI_API_KEY,
    temperature=0.7,
)

SYSTEM_PROMPT = """You are a friendly, knowledgeable sales assistant for GeniusAct — 
a payment platform that helps businesses accept crypto (USDC), card (Stripe), and PayPal 
payments from one checkout, with fees as low as 0.01%.

Your goals:
1. Answer questions about GeniusAct's features, pricing, and how it works
2. Help visitors understand the fee savings vs Stripe/PayPal
3. Guide interested businesses toward booking a demo or trying the checkout
4. Keep conversations focused on payments, crypto, and business value

Tone: Friendly, confident, concise. Max 3 sentences per response unless detail is needed.
Always end with a clear next step.

Use tools to get accurate info — never make up numbers or features.
If asked about booking/demo → use get_booking_link tool.
If asked about savings/fees with a number → use get_fee_savings tool.
If asked to see how it works → use get_demo_link tool.
For general product questions → use get_product_info tool."""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

agent = create_openai_tools_agent(llm, tools, prompt)

# ─────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────

app = FastAPI(title="GeniusAct Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev server
        "http://localhost:5174",
        "http://localhost:3000",
        "https://keep-empowering.com",
        "https://www.keep-empowering.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory sessions — swap for Redis in production
sessions: dict[str, list] = {}


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.websocket("/chat/{session_id}")
async def chat_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()

    if session_id not in sessions:
        sessions[session_id] = []

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=4,
        handle_parsing_errors=True,
    )

    try:
        while True:
            user_input = await websocket.receive_text()
            chat_history = sessions[session_id]

            try:
                result = executor.invoke({
                    "input": user_input,
                    "chat_history": chat_history,
                })
                response = result["output"]

                # Keep last 10 exchanges (20 messages)
                chat_history.append(HumanMessage(content=user_input))
                chat_history.append(AIMessage(content=response))
                sessions[session_id] = chat_history[-20:]

                await websocket.send_text(json.dumps({
                    "type": "message",
                    "content": response
                }))

            except Exception as e:
                print(f"[Agent Error] {e}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "content": "Sorry, I hit a snag. Could you try asking that again?"
                }))

    except WebSocketDisconnect:
        print(f"[WS] Session {session_id} disconnected")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
