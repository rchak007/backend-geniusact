import React, { useState } from "react";
import Navbar from "./components/Navbar";
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import Cart from "./pages/Cart";
import Landing from "./pages/Landing";
import Dashboard from "./pages/Dashboard";
import Contact from "./pages/Contact";
import Success from "./components/Success";
import Cancel from "./components/Cancel";
import { Buffer } from "buffer";
import ThankYou from "./components/cryptoThankYou";
import CryptoWalletSetup from "./components/CryptoWalletSetup";
import AgentChat from "./components/AgentChat";       // ← NEW

import '@fortawesome/fontawesome-free/css/all.min.css';
window.Buffer = Buffer;

function App() {
  return (
    <>
      <Router>
        <Navbar />

        <Routes>
          <Route path="/old" element={<Landing />} />
          <Route path="/" element={<Dashboard />} />
          <Route path="/cart" element={<Cart />} />
          <Route path="/contact-us" element={<Contact />} />
          <Route path="/success" element={<Success />} />
          <Route path="/cancel" element={<Cancel />} />
          <Route path="/thank-you" element={<ThankYou />} />
          <Route path="crypto-wallet-setup-guide" element={<CryptoWalletSetup />} />
        </Routes>

        <AgentChat />    {/* ← NEW — renders on every page, bottom-right corner */}
      </Router>
    </>
  );
}

export default App;
