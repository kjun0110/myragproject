"use client";

import { useState, useRef, useEffect } from "react";
import ChatMessage from "@/components/v1/ChatMessage";
import ChatInput from "@/components/v1/ChatInput";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
}

export default function GraphPage() {
  const [messages, setMessages] = useState<Message[]>(() => [
    {
      id: "1",
      role: "assistant",
      content: "안녕하세요! 로컬 Graph 모델을 사용하는 LangChain 챗봇입니다. 무엇을 도와드릴까요?",
      timestamp: new Date(),
    },
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async (content: string) => {
    if (!content.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);

    try {
      const apiEndpoint = "/api/v1/graph";
      const requestBody = {
        message: content,
        history: messages.map((m) => ({
          role: m.role,
          content: m.content,
        })),
      };

      const timeout = 120000;
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), timeout);

      let response: Response;
      try {
        response = await fetch(apiEndpoint, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(requestBody),
          signal: controller.signal,
        });
        clearTimeout(timeoutId);
      } catch (fetchError) {
        clearTimeout(timeoutId);
        if (fetchError instanceof Error && fetchError.name === "AbortError") {
          throw new Error(
            `요청 시간이 초과되었습니다 (${timeout / 1000}초). 모델이 로딩 중이거나 응답 생성에 시간이 오래 걸리고 있습니다. 잠시 후 다시 시도해주세요.`
          );
        }
        throw fetchError;
      }

      if (!response.ok) {
        let errorData: { error?: string; detail?: string; message?: string } = {};
        let errorMsg = "응답을 받는 중 오류가 발생했습니다.";

        try {
          const contentType = response.headers.get("content-type");
          if (contentType && contentType.includes("application/json")) {
            errorData = await response.json();
            errorMsg = errorData.error || errorData.detail || errorData.message || response.statusText || errorMsg;
          } else {
            const text = await response.text();
            errorMsg = text || response.statusText || errorMsg;
          }
        } catch (e) {
          errorMsg = response.statusText || `HTTP ${response.status} 오류`;
        }

        if (response.status === 400) {
          if (errorMsg.includes("로컬 환경이 아닙니다")) {
            const errorMessage: Message = {
              id: (Date.now() + 1).toString(),
              role: "assistant",
              content: "ℹ️ 현재 EC2 환경입니다.",
              timestamp: new Date(),
            };
            setMessages((prev) => [...prev, errorMessage]);
            return;
          }
        }

        if (response.status === 503) {
          const errorMessage: Message = {
            id: (Date.now() + 1).toString(),
            role: "assistant",
            content: `⚠️ ${errorMsg}`,
            timestamp: new Date(),
          };
          setMessages((prev) => [...prev, errorMessage]);
          return;
        }

        if (response.status === 404) {
          const errorMessage: Message = {
            id: (Date.now() + 1).toString(),
            role: "assistant",
            content: "⚠️ 백엔드 서버를 찾을 수 없습니다. 서버가 실행 중인지 확인해주세요.",
            timestamp: new Date(),
          };
          setMessages((prev) => [...prev, errorMessage]);
          return;
        }

        throw new Error(errorMsg);
      }

      // Graph 모드는 항상 스트리밍
      if (!response.body) {
        throw new Error("스트리밍 응답을 받을 수 없습니다.");
      }

      const streamingMessageId = (Date.now() + 1).toString();
      const streamingMessage: Message = {
        id: streamingMessageId,
        role: "assistant",
        content: "",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, streamingMessage]);

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let accumulatedText = "";

      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value, { stream: true });
          if (chunk) {
            accumulatedText += chunk;
            setMessages((prev) => {
              const updated = [...prev];
              const msgIndex = updated.findIndex((m) => m.id === streamingMessageId);
              if (msgIndex !== -1) {
                updated[msgIndex] = {
                  ...updated[msgIndex],
                  content: accumulatedText,
                };
              }
              return updated;
            });
          }
        }
      } catch (streamError) {
        console.error("[ERROR] 스트리밍 처리 실패:", streamError);
        setMessages((prev) => {
          const updated = [...prev];
          const msgIndex = updated.findIndex((m) => m.id === streamingMessageId);
          if (msgIndex !== -1) {
            updated[msgIndex] = {
              ...updated[msgIndex],
              content: accumulatedText || "⚠️ 스트리밍 중 오류가 발생했습니다.",
            };
          }
          return updated;
        });
      }
    } catch (error) {
      console.error("[ERROR] 메시지 전송 실패:", error);
      let errorContent = "알 수 없는 오류가 발생했습니다.";

      if (error instanceof Error) {
        if (error.message.includes("시간이 초과")) {
          errorContent = error.message;
        } else if (error.message.includes("Failed to fetch") || error.message.includes("fetch")) {
          errorContent = "⚠️ 백엔드 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인해주세요.";
        } else {
          errorContent = `⚠️ 오류가 발생했습니다: ${error.message}`;
        }
      }

      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: errorContent,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="chat-container">
      <header className="chat-header">
        <h1>🔗 로컬 Graph Chat</h1>
        <p>로컬 Graph 모델을 사용하는 AI 챗봇</p>
      </header>

      <main className="chat-main">
        <div className="messages-container">
          {messages.map((message) => (
            <ChatMessage key={message.id} message={message} />
          ))}
          {isLoading && (
            <div className="loading-indicator">
              <div className="typing-dots">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
        <ChatInput onSend={handleSendMessage} disabled={isLoading} />
      </main>

      <style jsx>{`
        .chat-container {
          display: flex;
          flex-direction: column;
          height: 100vh;
          background: white;
        }

        .chat-header {
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
          padding: 1.5rem;
          text-align: center;
        }

        .chat-header h1 {
          font-size: 1.5rem;
          margin: 0 0 0.5rem 0;
        }

        .chat-header p {
          font-size: 0.9rem;
          opacity: 0.9;
          margin: 0;
        }

        .chat-main {
          flex: 1;
          display: flex;
          flex-direction: column;
          overflow: hidden;
        }

        .messages-container {
          flex: 1;
          overflow-y: auto;
          padding: 1rem;
          background: #f5f5f5;
        }

        .loading-indicator {
          display: flex;
          justify-content: flex-start;
          padding: 1rem;
        }

        .typing-dots {
          display: flex;
          gap: 0.5rem;
          padding: 1rem 1.5rem;
          background: white;
          border-radius: 1.5rem;
          box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        }

        .typing-dots span {
          width: 8px;
          height: 8px;
          border-radius: 50%;
          background: #667eea;
          animation: typing 1.4s infinite;
        }

        .typing-dots span:nth-child(2) {
          animation-delay: 0.2s;
        }

        .typing-dots span:nth-child(3) {
          animation-delay: 0.4s;
        }

        @keyframes typing {
          0%,
          60%,
          100% {
            transform: translateY(0);
            opacity: 0.7;
          }
          30% {
            transform: translateY(-10px);
            opacity: 1;
          }
        }
      `}</style>
    </div>
  );
}
