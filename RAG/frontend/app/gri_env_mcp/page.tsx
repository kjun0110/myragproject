"use client";

import { useState, useRef, useEffect } from "react";
import { useRouter } from "next/navigation";
import ChatMessage from "@/components/ChatMessage";
import ChatInput from "@/components/ChatInput";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
}

export default function GriEnvPage() {
  const router = useRouter();
  const [messages, setMessages] = useState<Message[]>(() => [
    {
      id: "1",
      role: "assistant",
      content: "안녕하세요! GRI 환경 컨텐츠 MCP 챗봇입니다. 무엇을 도와드릴까요?",
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
      // 디버깅: GRI_ENV_MCP 채팅 요청 시작
      console.log("[DEBUG] GRI_ENV_MCP 채팅 요청 시작");

      // GRI 환경 컨텐츠 MCP 라우터로 연결
      const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";
      const apiEndpoint = `${backendUrl}/api/v1/esg/gri-env-contents/chat`;
      console.log("[DEBUG] 백엔드 URL:", backendUrl);
      console.log("[DEBUG] API 엔드포인트:", apiEndpoint);
      // GRI 환경 컨텐츠 MCP 요청 본문
      const requestBody = {
        message: content,
        history: messages.map((m) => ({
          role: m.role,
          content: m.content,
        })),
      };

      console.log("[DEBUG] 요청 본문:", JSON.stringify(requestBody, null, 2));

      // 타임아웃 설정
      const timeout = 30000; // 30초
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), timeout);

      let response: Response;
      try {
        console.log("[DEBUG] GRI_ENV_MCP 요청 전송 중...");
        response = await fetch(apiEndpoint, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(requestBody),
          signal: controller.signal,
        });
        clearTimeout(timeoutId);
        console.log("[DEBUG] GRI_ENV_MCP 응답 수신:", response.status, response.statusText);
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
        // 스트리밍 응답이 아닌 경우에만 JSON 파싱 시도
        let errorData = {};
        let errorMsg = "응답을 받는 중 오류가 발생했습니다.";

        try {
          // Content-Type 확인
          const contentType = response.headers.get("content-type");
          if (contentType && contentType.includes("application/json")) {
            errorData = await response.json();
            errorMsg = errorData.error || errorData.detail || errorData.message || response.statusText || errorMsg;
          } else {
            // JSON이 아닌 경우 텍스트로 읽기
            const text = await response.text();
            errorMsg = text || response.statusText || errorMsg;
          }
        } catch (e) {
          // 파싱 실패 시 상태 코드 기반 메시지
          errorMsg = response.statusText || `HTTP ${response.status} 오류`;
        }

        // 백엔드 환경 불일치 에러 (400)
        if (response.status === 400) {
          // 백엔드가 로컬일 때 OpenAI 선택
          if (errorMsg.includes("로컬환경") && modelType === "openai") {
            const errorMessage: Message = {
              id: (Date.now() + 1).toString(),
              role: "assistant",
              content: "ℹ️ 현재 로컬 환경입니다.",
              timestamp: new Date(),
            };
            setMessages((prev) => [...prev, errorMessage]);
            return;
          }
          // 백엔드가 클라우드일 때 로컬 모델 선택
          if (errorMsg.includes("로컬 환경이 아닙니다") && (modelType === "local" || modelType === "graph")) {
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

        // OpenAI 호출량 초과 에러
        if (response.status === 429) {
          const errorMessage: Message = {
            id: (Date.now() + 1).toString(),
            role: "assistant",
            content: "⚠️ OpenAI API 호출량이 초과되었습니다. 할당량을 확인하고 다시 시도해주세요. 또는 '로컬 모델' 버튼을 선택하여 로컬 모델을 사용해주세요.",
            timestamp: new Date(),
          };
          setMessages((prev) => [...prev, errorMessage]);
          return;
        }

        // 백엔드 연결 오류
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

        // 404 Not Found 에러 처리
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

      // 비스트리밍 처리 (GRI 환경 컨텐츠 MCP)
      console.log("[DEBUG] GRI_ENV_MCP 응답 파싱 중...");
      const data = await response.json();
      console.log("[DEBUG] GRI_ENV_MCP 응답 데이터:", data);

      // 에러 응답인지 확인
      if (data.error || data.detail) {
        const errorMessage: Message = {
          id: (Date.now() + 1).toString(),
          role: "assistant",
          content: `⚠️ ${data.error || data.detail}`,
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, errorMessage]);
        return;
      }

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: data.response || "응답을 생성할 수 없습니다.",
        timestamp: new Date(),
      };

      console.log("[DEBUG] GRI_ENV_MCP 메시지 추가 완료");
      setMessages((prev) => [...prev, assistantMessage]);
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
        <div className="header-top">
          <div className="header-title">
            <h1>🌿 GRI 환경 컨텐츠 MCP</h1>
            <button
              className="spam-button"
              onClick={() => router.push("/")}
              disabled={isLoading}
            >
              🏠 홈으로
            </button>
          </div>
        </div>
        <p>GRI 환경 컨텐츠 MCP 챗봇</p>
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
          max-width: 800px;
          margin: 0 auto;
          background: white;
          box-shadow: 0 0 20px rgba(0, 0, 0, 0.1);
        }

        .chat-header {
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
          padding: 1.5rem;
          text-align: center;
          box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }

        .header-top {
          display: flex;
          justify-content: center;
          align-items: center;
          margin-bottom: 0.5rem;
        }

        .header-title {
          display: flex;
          align-items: center;
          gap: 1rem;
          justify-content: center;
          width: 100%;
        }

        .chat-header h1 {
          font-size: 1.5rem;
          margin: 0;
        }

        .spam-button {
          padding: 0.5rem 1rem;
          border: 2px solid rgba(255, 255, 255, 0.5);
          border-radius: 0.5rem;
          background: rgba(255, 255, 255, 0.15);
          color: white;
          font-size: 0.9rem;
          cursor: pointer;
          transition: all 0.2s;
          white-space: nowrap;
        }

        .spam-button:hover:not(:disabled) {
          background: rgba(255, 255, 255, 0.25);
          border-color: rgba(255, 255, 255, 0.7);
          transform: translateY(-1px);
        }

        .spam-button:active:not(:disabled) {
          transform: translateY(0);
        }

        .spam-button:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .chat-header p {
          font-size: 0.9rem;
          opacity: 0.9;
        }

        .model-selector {
          display: flex;
          gap: 0.5rem;
          margin-top: 1rem;
          justify-content: center;
        }

        .model-button {
          padding: 0.5rem 1rem;
          border: 2px solid rgba(255, 255, 255, 0.3);
          border-radius: 0.5rem;
          background: rgba(255, 255, 255, 0.1);
          color: white;
          font-size: 0.9rem;
          cursor: pointer;
          transition: all 0.2s;
        }

        .model-button:hover:not(:disabled) {
          background: rgba(255, 255, 255, 0.2);
          border-color: rgba(255, 255, 255, 0.5);
        }

        .model-button.active {
          background: rgba(255, 255, 255, 0.3);
          border-color: white;
          font-weight: 600;
        }

        .model-button:disabled {
          opacity: 0.5;
          cursor: not-allowed;
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

        @media (max-width: 768px) {
          .chat-container {
            height: 100vh;
            height: 100dvh; /* 모바일 브라우저 높이 */
          }

          .chat-header h1 {
            font-size: 1.2rem;
          }
        }
      `}</style>
    </div>
  );
}
