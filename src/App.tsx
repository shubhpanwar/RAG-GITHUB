import React, { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { 
  Github, 
  Search, 
  Send, 
  Loader2, 
  Code, 
  Database, 
  Cpu, 
  Layout,
  ChevronRight,
  Terminal,
  AlertCircle,
  CheckCircle2
} from "lucide-react";
import { GoogleGenAI } from "@google/genai";

// Simple Vector Store Implementation for Frontend
class SimpleVectorStore {
  private store: { text: string; metadata: any; embedding: number[] }[] = [];

  add(text: string, metadata: any, embedding: number[]) {
    this.store.push({ text, metadata, embedding });
  }

  search(queryEmbedding: number[], topK: number = 5) {
    const results = this.store.map(item => ({
      ...item,
      score: this.cosineSimilarity(queryEmbedding, item.embedding)
    }));
    return results.sort((a, b) => b.score - a.score).slice(0, topK);
  }

  private cosineSimilarity(vecA: number[], vecB: number[]) {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < vecA.length; i++) {
      dotProduct += vecA[i] * vecB[i];
      normA += vecA[i] * vecA[i];
      normB += vecB[i] * vecB[i];
    }
    const mag = Math.sqrt(normA) * Math.sqrt(normB);
    return mag === 0 ? 0 : dotProduct / mag;
  }

  clear() {
    this.store = [];
  }
}

const vectorStore = new SimpleVectorStore();

interface Message {
  role: "user" | "assistant";
  content: string;
  thoughts?: string[];
}

export default function App() {
  const [repoUrl, setRepoUrl] = useState("");
  const [isIngesting, setIsIngesting] = useState(false);
  const [ingestionStatus, setIngestionStatus] = useState<"idle" | "cloning" | "vectorizing" | "complete" | "error">("idle");
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isChatting, setIsChatting] = useState(false);
  const [currentThoughts, setCurrentThoughts] = useState<string[]>([]);
  const chatEndRef = useRef<HTMLDivElement>(null);

  // Initialize Gemini AI on Frontend
  const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY || "" });

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, currentThoughts]);

  const handleIngest = async () => {
    if (!repoUrl) return;
    setIsIngesting(true);
    setIngestionStatus("cloning");
    vectorStore.clear();
    setCurrentThoughts(["Supervisor: Initiating repository cloning..."]);
    
    try {
      const res = await fetch("/api/ingest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ repoUrl }),
      });
      if (!res.ok) throw new Error("Cloning failed");
      const { chunks } = await res.json();
      
      setIngestionStatus("vectorizing");
      setCurrentThoughts(prev => [...prev, `Retriever: Received ${chunks.length} code chunks. Starting vectorization...`]);

      // Vectorize chunks on frontend
      // We process in small batches to avoid hitting rate limits too hard
      const batchSize = 5;
      for (let i = 0; i < chunks.length; i += batchSize) {
        const batch = chunks.slice(i, i + batchSize);
        await Promise.all(batch.map(async (chunk: any) => {
          try {
            const result = await ai.models.embedContent({
              model: "gemini-embedding-2-preview",
              contents: [{ parts: [{ text: chunk.text }] }]
            });
            vectorStore.add(chunk.text, chunk.metadata, result.embeddings[0].values);
          } catch (e) {
            console.warn("Failed to embed chunk", e);
          }
        }));
        if (i % 20 === 0) {
          setCurrentThoughts(prev => [...prev, `Retriever: Vectorized ${Math.min(i + batchSize, chunks.length)}/${chunks.length} chunks...`]);
        }
      }

      setIngestionStatus("complete");
      setCurrentThoughts(prev => [...prev, "Supervisor: Ingestion and vectorization complete. Ready for queries."]);
    } catch (err: any) {
      console.error(err);
      setIngestionStatus("error");
      setCurrentThoughts(prev => [...prev, `Error: ${err.message}`]);
    } finally {
      setIsIngesting(false);
    }
  };

  const handleChat = async () => {
    if (!input || isChatting) return;
    const userMsg = input;
    setInput("");
    setMessages(prev => [...prev, { role: "user", content: userMsg }]);
    setIsChatting(true);
    setCurrentThoughts(["Supervisor: Analyzing user query..."]);

    try {
      // 1. Vectorize Query
      setCurrentThoughts(prev => [...prev, "Retriever: Searching vector database..."]);
      const queryEmbed = await ai.models.embedContent({
        model: "gemini-embedding-2-preview",
        contents: [{ parts: [{ text: userMsg }] }]
      });
      
      // 2. Search Vector Store
      const results = vectorStore.search(queryEmbed.embeddings[0].values, 5);
      const context = results.map(r => `File: ${r.metadata.path}\nContent:\n${r.text}`).join("\n\n---\n\n");

      // 3. Generate Answer (Streaming)
      setCurrentThoughts(prev => [...prev, "Analyzer: Examining retrieved code...", "Formatter: Synthesizing final answer..."]);
      
      const prompt = `
        You are an expert software architect. Use the following code snippets from a GitHub repository to answer the user's question.
        
        User Question: ${userMsg}
        
        Context:
        ${context}
        
        Analyze the logic and provide a detailed explanation. If the context is insufficient, state what's missing.
      `;

      const result = await ai.models.generateContentStream({
        model: "gemini-3-flash-preview",
        contents: [{ role: "user", parts: [{ text: prompt }] }],
        config: {
          systemInstruction: "You are a helpful AI assistant that analyzes codebases. Format your output in clean Markdown."
        }
      });

      let assistantMsg = "";
      setMessages(prev => [...prev, { role: "assistant", content: "" }]);

      for await (const chunk of result) {
        assistantMsg += chunk.text;
        setMessages(prev => {
          const last = prev[prev.length - 1];
          if (last?.role === "assistant") {
            return [...prev.slice(0, -1), { ...last, content: assistantMsg }];
          }
          return prev;
        });
      }
      
      setMessages(prev => {
        const last = prev[prev.length - 1];
        if (last?.role === "assistant") {
          return [...prev.slice(0, -1), { ...last, thoughts: ["Supervisor: Task complete."] }];
        }
        return prev;
      });
    } catch (err: any) {
      console.error(err);
      setMessages(prev => [...prev, { role: "assistant", content: `Error: ${err.message}` }]);
    } finally {
      setIsChatting(false);
      setCurrentThoughts([]);
    }
  };

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-[#e0e0e0] font-sans selection:bg-[#F27D26] selection:text-white">
      {/* Header */}
      <header className="border-b border-[#1a1a1a] p-4 flex items-center justify-between bg-[#0a0a0a]/80 backdrop-blur-md sticky top-0 z-50">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-[#F27D26] rounded-lg flex items-center justify-center shadow-[0_0_20px_rgba(242,125,38,0.3)]">
            <Github className="text-white w-6 h-6" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight text-white">GitAgent RAG</h1>
            <p className="text-[10px] uppercase tracking-[0.2em] text-[#888] font-mono">Distributed Agentic System</p>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 px-3 py-1 bg-[#1a1a1a] rounded-full border border-[#333]">
            <div className={`w-2 h-2 rounded-full ${ingestionStatus === "complete" ? "bg-green-500 animate-pulse" : "bg-yellow-500"}`} />
            <span className="text-xs font-mono uppercase tracking-wider text-[#aaa]">
              {ingestionStatus === "complete" ? "Index Ready" : "No Repository"}
            </span>
          </div>
        </div>
      </header>

      <main className="flex h-[calc(100vh-73px)] overflow-hidden">
        {/* Sidebar */}
        <aside className="w-80 border-r border-[#1a1a1a] p-6 flex flex-col gap-8 bg-[#0d0d0d]">
          <section>
            <label className="text-[10px] uppercase tracking-[0.2em] text-[#555] font-mono mb-3 block">Repository Ingestion</label>
            <div className="space-y-4">
              <div className="relative">
                <Github className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[#444]" />
                <input 
                  type="text" 
                  placeholder="https://github.com/user/repo"
                  className="w-full bg-[#151515] border border-[#222] rounded-lg py-2 pl-10 pr-4 text-sm focus:outline-none focus:border-[#F27D26] transition-colors"
                  value={repoUrl}
                  onChange={(e) => setRepoUrl(e.target.value)}
                />
              </div>
              <button 
                onClick={handleIngest}
                disabled={isIngesting || !repoUrl}
                className="w-full bg-[#F27D26] hover:bg-[#ff8c3a] disabled:opacity-50 disabled:hover:bg-[#F27D26] text-white font-bold py-2 rounded-lg transition-all flex items-center justify-center gap-2 shadow-lg shadow-[#F27D26]/10"
              >
                {isIngesting ? <Loader2 className="w-4 h-4 animate-spin" /> : <Database className="w-4 h-4" />}
                {isIngesting ? "Ingesting..." : "Build Index"}
              </button>
            </div>
          </section>

          <section className="flex-1">
            <label className="text-[10px] uppercase tracking-[0.2em] text-[#555] font-mono mb-3 block">Pipeline Status</label>
            <div className="space-y-3">
              <StatusItem icon={<Layout className="w-4 h-4" />} label="Ingestion Engine" status={ingestionStatus !== "idle" ? "active" : "idle"} />
              <StatusItem icon={<Code className="w-4 h-4" />} label="AST Parser" status={ingestionStatus === "complete" ? "active" : "idle"} />
              <StatusItem icon={<Database className="w-4 h-4" />} label="Vector Storage" status={ingestionStatus === "complete" ? "active" : "idle"} />
              <StatusItem icon={<Cpu className="w-4 h-4" />} label="Multi-Agent Orchestrator" status="active" />
            </div>
          </section>

          <div className="p-4 bg-[#1a1a1a] rounded-xl border border-[#222]">
            <div className="flex items-center gap-2 mb-2">
              <Terminal className="w-4 h-4 text-[#F27D26]" />
              <span className="text-[10px] font-mono uppercase tracking-wider text-[#888]">Agent Logs</span>
            </div>
            <div className="h-32 overflow-y-auto font-mono text-[10px] text-[#666] space-y-1">
              {currentThoughts.map((t, i) => (
                <div key={i} className="flex gap-2">
                  <span className="text-[#F27D26]">{">"}</span>
                  <span>{t}</span>
                </div>
              ))}
              {currentThoughts.length === 0 && <div className="italic">Waiting for query...</div>}
            </div>
          </div>
        </aside>

        {/* Chat Area */}
        <div className="flex-1 flex flex-col bg-[#0a0a0a]">
          <div className="flex-1 overflow-y-auto p-8 space-y-8 scrollbar-hide">
            {messages.length === 0 && (
              <div className="h-full flex flex-col items-center justify-center text-center max-w-md mx-auto">
                <div className="w-16 h-16 bg-[#1a1a1a] rounded-2xl flex items-center justify-center mb-6 border border-[#222]">
                  <Search className="w-8 h-8 text-[#444]" />
                </div>
                <h2 className="text-2xl font-bold text-white mb-2">Ask about the codebase</h2>
                <p className="text-[#666] text-sm">
                  Once you've indexed a repository, you can ask complex questions about its architecture, logic, and dependencies.
                </p>
              </div>
            )}
            {messages.map((msg, i) => (
              <motion.div 
                key={i}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className={`flex gap-4 ${msg.role === "user" ? "justify-end" : "justify-start"}`}
              >
                {msg.role === "assistant" && (
                  <div className="w-8 h-8 rounded-lg bg-[#1a1a1a] border border-[#222] flex items-center justify-center shrink-0">
                    <Cpu className="w-4 h-4 text-[#F27D26]" />
                  </div>
                )}
                <div className={`max-w-[80%] space-y-2 ${msg.role === "user" ? "items-end" : "items-start"}`}>
                  <div className={`p-4 rounded-2xl text-sm leading-relaxed ${
                    msg.role === "user" 
                      ? "bg-[#F27D26] text-white rounded-tr-none" 
                      : "bg-[#151515] border border-[#222] text-[#ccc] rounded-tl-none"
                  }`}>
                    {msg.content}
                  </div>
                  {msg.thoughts && (
                    <div className="flex flex-wrap gap-2">
                      {msg.thoughts.map((t, j) => (
                        <span key={j} className="text-[9px] font-mono uppercase tracking-wider bg-[#1a1a1a] text-[#555] px-2 py-0.5 rounded border border-[#222]">
                          {t.split(":")[0]}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
                {msg.role === "user" && (
                  <div className="w-8 h-8 rounded-lg bg-[#F27D26] flex items-center justify-center shrink-0">
                    <Github className="w-4 h-4 text-white" />
                  </div>
                )}
              </motion.div>
            ))}
            <div ref={chatEndRef} />
          </div>

          {/* Input Area */}
          <div className="p-6 border-t border-[#1a1a1a] bg-[#0d0d0d]">
            <div className="max-w-4xl mx-auto relative">
              <input 
                type="text" 
                placeholder="Ask a question about the repository..."
                className="w-full bg-[#151515] border border-[#222] rounded-2xl py-4 pl-6 pr-16 text-sm focus:outline-none focus:border-[#F27D26] transition-all shadow-2xl"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleChat()}
              />
              <button 
                onClick={handleChat}
                disabled={isChatting || !input || ingestionStatus !== "complete"}
                className="absolute right-3 top-1/2 -translate-y-1/2 w-10 h-10 bg-[#F27D26] hover:bg-[#ff8c3a] disabled:opacity-30 disabled:hover:bg-[#F27D26] text-white rounded-xl flex items-center justify-center transition-all"
              >
                {isChatting ? <Loader2 className="w-5 h-5 animate-spin" /> : <Send className="w-5 h-5" />}
              </button>
            </div>
            <p className="text-center text-[10px] text-[#444] mt-4 font-mono uppercase tracking-widest">
              Powered by Gemini 2.0 Flash & Distributed Agentic RAG
            </p>
          </div>
        </div>
      </main>
    </div>
  );
}

function StatusItem({ icon, label, status }: { icon: React.ReactNode, label: string, status: "active" | "idle" }) {
  return (
    <div className={`flex items-center gap-3 p-3 rounded-xl border transition-all ${
      status === "active" ? "bg-[#1a1a1a]/50 border-[#F27D26]/30" : "bg-transparent border-[#1a1a1a]"
    }`}>
      <div className={`${status === "active" ? "text-[#F27D26]" : "text-[#333]"}`}>
        {icon}
      </div>
      <span className={`text-xs font-medium ${status === "active" ? "text-[#ccc]" : "text-[#444]"}`}>{label}</span>
      <div className="ml-auto">
        {status === "active" ? (
          <CheckCircle2 className="w-3 h-3 text-green-500" />
        ) : (
          <div className="w-3 h-3 rounded-full border border-[#333]" />
        )}
      </div>
    </div>
  );
}
