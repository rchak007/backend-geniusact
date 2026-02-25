import { useState, useEffect, useRef, useCallback } from "react";
import { Link } from "react-router-dom";
import { X, Send, MessageCircle, Minimize2 } from "lucide-react";

const WS_BASE = import.meta.env.VITE_AGENT_WS_URL || "ws://localhost:8000/chat";

function getOrCreateSessionId() {
  const stored = sessionStorage.getItem("ga_session_id");
  if (stored) return stored;
  const id = `ga_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
  sessionStorage.setItem("ga_session_id", id);
  return id;
}

// Render message text — turns URLs into clickable buttons,
// turns /cart or /crypto-wallet-setup-guide into React Router links
function MessageContent({ content }) {
  // Split by URLs first
  const urlRegex = /(https?:\/\/[^\s]+)/g;
  const internalLinkRegex = /\/(cart|crypto-wallet-setup-guide|contact-us)/g;

  const parts = content.split(urlRegex);

  return (
    <span>
      {parts.map((part, i) => {
        if (urlRegex.test(part)) {
          return (
            <a
              key={i}
              href={part}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-block mt-2 px-3 py-1.5 bg-[#4b6bff] text-white text-xs font-medium rounded-lg hover:bg-[#3d62ea] transition-colors"
            >
              📅 Book a call →
            </a>
          );
        }
        // Check for internal route mentions
        if (part.includes("/cart")) {
          const segments = part.split("/cart");
          return (
            <span key={i}>
              {segments[0]}
              <Link
                to="/cart"
                className="inline-block mt-1 px-3 py-1.5 bg-[#4b6bff] text-white text-xs font-medium rounded-lg hover:bg-[#3d62ea] transition-colors"
              >
                👀 See live demo →
              </Link>
              {segments[1]}
            </span>
          );
        }
        return part;
      })}
    </span>
  );
}

export default function AgentChat() {
  const [isOpen, setIsOpen]     = useState(false);
  const [messages, setMessages] = useState([]);
  const [input, setInput]       = useState("");
  const [status, setStatus]     = useState("disconnected"); // connecting | connected | disconnected | error
  const [isTyping, setIsTyping] = useState(false);
  const [unread, setUnread]     = useState(0);

  const wsRef          = useRef(null);
  const sessionId      = useRef(getOrCreateSessionId());
  const bottomRef      = useRef(null);
  const inputRef       = useRef(null);

  // Scroll to bottom on new message
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isTyping]);

  // Focus input when opened
  useEffect(() => {
    if (isOpen) {
      setUnread(0);
      setTimeout(() => inputRef.current?.focus(), 150);
    }
  }, [isOpen]);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    setStatus("connecting");
    const ws = new WebSocket(`${WS_BASE}/${sessionId.current}`);
    wsRef.current = ws;

    ws.onopen = () => {
      setStatus("connected");
      if (messages.length === 0) {
        setMessages([{
          id: "welcome",
          role: "bot",
          content: "Hey! 👋 I'm the GeniusAct assistant. Ask me anything about crypto payments, fees, or how to get started!",
          ts: new Date(),
        }]);
      }
    };

    ws.onmessage = (e) => {
      setIsTyping(false);
      try {
        const data = JSON.parse(e.data);
        const newMsg = {
          id: `bot_${Date.now()}`,
          role: "bot",
          content: data.content,
          ts: new Date(),
        };
        setMessages(prev => [...prev, newMsg]);
        if (!isOpen) setUnread(u => u + 1);
      } catch {
        // fallback for plain text
        setMessages(prev => [...prev, {
          id: `bot_${Date.now()}`,
          role: "bot",
          content: e.data,
          ts: new Date(),
        }]);
      }
    };

    ws.onclose = () => setStatus("disconnected");
    ws.onerror = () => setStatus("error");
  }, [messages.length, isOpen]);

  useEffect(() => {
    if (isOpen) connect();
  }, [isOpen, connect]);

  const send = () => {
    const text = input.trim();
    if (!text || status !== "connected") return;

    setMessages(prev => [...prev, {
      id: `user_${Date.now()}`,
      role: "user",
      content: text,
      ts: new Date(),
    }]);
    setInput("");
    setIsTyping(true);
    wsRef.current?.send(text);
  };

  const handleKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  const statusColor = {
    connected:    "bg-green-400",
    connecting:   "bg-yellow-400 animate-pulse",
    disconnected: "bg-slate-400",
    error:        "bg-red-400",
  }[status];

  const statusText = {
    connected:    "Online",
    connecting:   "Connecting...",
    disconnected: "Offline",
    error:        "Connection error",
  }[status];

  // Quick reply suggestions shown before first user message
  const suggestions = [
    "How does crypto checkout work?",
    "How much can I save on fees?",
    "Show me a demo",
    "Book a call",
  ];
  const showSuggestions = messages.length <= 1 && status === "connected";

  return (
    <>
      {/* ── CHAT WINDOW ── */}
      {isOpen && (
        <div
          className="fixed bottom-24 right-4 sm:right-6 z-50 flex flex-col"
          style={{
            width: "min(360px, calc(100vw - 32px))",
            height: "min(520px, calc(100vh - 120px))",
            borderRadius: "16px",
            boxShadow: "0 24px 64px rgba(0,0,0,0.18), 0 0 0 1px rgba(0,0,0,0.06)",
            background: "#fff",
            animation: "gaSlideUp 0.25s cubic-bezier(0.16,1,0.3,1)",
          }}
        >
          {/* Header */}
          <div
            className="flex items-center justify-between px-4 py-3 flex-shrink-0"
            style={{
              background: "linear-gradient(135deg, #4b6bff, #6b8aff)",
              borderRadius: "16px 16px 0 0",
            }}
          >
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-xl bg-white/20 flex items-center justify-center text-lg">
                ⚡
              </div>
              <div>
                <p className="text-white font-semibold text-sm leading-tight">GeniusAct</p>
                <div className="flex items-center gap-1.5 mt-0.5">
                  <span className={`w-1.5 h-1.5 rounded-full ${statusColor}`} />
                  <span className="text-white/75 text-xs">{statusText}</span>
                </div>
              </div>
            </div>
            <button
              onClick={() => setIsOpen(false)}
              className="text-white/70 hover:text-white transition-colors p-1 rounded-lg hover:bg-white/10"
            >
              <X size={18} />
            </button>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto px-3 py-3 space-y-2.5"
            style={{ scrollbarWidth: "thin", scrollbarColor: "#e2e8f0 transparent" }}
          >
            {messages.map(msg => (
              <div
                key={msg.id}
                className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
              >
                <div
                  className={`max-w-[82%] px-3.5 py-2.5 rounded-2xl text-sm leading-relaxed ${
                    msg.role === "user"
                      ? "text-white rounded-br-sm"
                      : "bg-slate-100 text-slate-800 rounded-bl-sm"
                  }`}
                  style={msg.role === "user" ? {
                    background: "linear-gradient(135deg, #4b6bff, #6b8aff)"
                  } : {}}
                >
                  {msg.role === "bot"
                    ? <MessageContent content={msg.content} />
                    : msg.content
                  }
                </div>
              </div>
            ))}

            {/* Quick suggestions */}
            {showSuggestions && (
              <div className="flex flex-wrap gap-2 pt-1">
                {suggestions.map(s => (
                  <button
                    key={s}
                    onClick={() => {
                      setMessages(prev => [...prev, {
                        id: `user_${Date.now()}`, role: "user", content: s, ts: new Date()
                      }]);
                      setIsTyping(true);
                      wsRef.current?.send(s);
                    }}
                    className="text-xs px-3 py-1.5 rounded-full border border-[#4b6bff] text-[#4b6bff] hover:bg-[#4b6bff] hover:text-white transition-colors"
                  >
                    {s}
                  </button>
                ))}
              </div>
            )}

            {/* Typing indicator */}
            {isTyping && (
              <div className="flex justify-start">
                <div className="bg-slate-100 px-4 py-3 rounded-2xl rounded-bl-sm flex gap-1 items-center">
                  {[0, 0.2, 0.4].map((delay, i) => (
                    <span
                      key={i}
                      className="w-1.5 h-1.5 bg-slate-400 rounded-full"
                      style={{ animation: `gaBounce 1.2s ${delay}s infinite` }}
                    />
                  ))}
                </div>
              </div>
            )}

            {/* Reconnect prompt */}
            {(status === "disconnected" || status === "error") && messages.length > 0 && (
              <div className="text-center text-xs text-slate-400 py-1">
                Connection lost.{" "}
                <button onClick={connect} className="text-[#4b6bff] underline">
                  Reconnect
                </button>
              </div>
            )}

            <div ref={bottomRef} />
          </div>

          {/* Input */}
          <div className="flex-shrink-0 px-3 pb-3 pt-2 border-t border-slate-100">
            <div className="flex gap-2 items-center bg-slate-50 rounded-xl px-3 py-2 border border-slate-200 focus-within:border-[#4b6bff] transition-colors">
              <input
                ref={inputRef}
                type="text"
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={handleKey}
                placeholder={status === "connected" ? "Ask anything..." : "Connecting..."}
                disabled={status !== "connected"}
                maxLength={500}
                className="flex-1 bg-transparent text-sm text-slate-800 placeholder-slate-400 outline-none disabled:opacity-50"
              />
              <button
                onClick={send}
                disabled={!input.trim() || status !== "connected"}
                className="w-7 h-7 rounded-lg flex items-center justify-center transition-all disabled:opacity-30 disabled:cursor-not-allowed"
                style={{ background: "linear-gradient(135deg, #4b6bff, #6b8aff)" }}
              >
                <Send size={13} className="text-white" />
              </button>
            </div>
            <p className="text-center text-[10px] text-slate-300 mt-1.5">Powered by GeniusAct AI</p>
          </div>
        </div>
      )}

      {/* ── BUBBLE BUTTON ── */}
      <button
        onClick={() => setIsOpen(o => !o)}
        className="fixed bottom-5 right-4 sm:right-6 z-50 w-14 h-14 rounded-full flex items-center justify-center shadow-lg transition-transform hover:scale-105 active:scale-95"
        style={{ background: "linear-gradient(135deg, #4b6bff, #6b8aff)" }}
        aria-label={isOpen ? "Close chat" : "Open chat"}
      >
        {isOpen
          ? <X size={22} className="text-white" />
          : <MessageCircle size={22} className="text-white" />
        }
        {/* Unread badge */}
        {!isOpen && unread > 0 && (
          <span className="absolute -top-1 -right-1 w-5 h-5 bg-red-500 text-white text-xs font-bold rounded-full flex items-center justify-center">
            {unread}
          </span>
        )}
      </button>

      {/* ── KEYFRAMES ── */}
      <style>{`
        @keyframes gaSlideUp {
          from { opacity: 0; transform: translateY(12px) scale(0.97); }
          to   { opacity: 1; transform: translateY(0) scale(1); }
        }
        @keyframes gaBounce {
          0%, 60%, 100% { transform: translateY(0); }
          30%            { transform: translateY(-5px); }
        }
      `}</style>
    </>
  );
}
