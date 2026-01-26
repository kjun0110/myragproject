"use client";

import { useState, useRef, useEffect } from "react";
import { useRouter } from "next/navigation";
import ChatMessage from "@/components/v1/ChatMessage";
import ChatInput from "@/components/v1/ChatInput";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
}

export default function SpamPage() {
  const router = useRouter();
  const [messages, setMessages] = useState<Message[]>(() => [
    {
      id: "1",
      role: "assistant",
      content: "안녕하세요! LangGraph Spam 판독기입니다. 스팸 메일을 분석해드립니다.",
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
      // 타임아웃 설정 (스팸 분석은 시간이 걸릴 수 있으므로 240초)
      const timeout = 240000;
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), timeout);

      let response: Response;
      try {
        response = await fetch("/v1/api/spam-analyze", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            email_text: content,
          }),
          signal: controller.signal,
        });
        clearTimeout(timeoutId);
      } catch (fetchError) {
        clearTimeout(timeoutId);
        if (fetchError instanceof Error && fetchError.name === "AbortError") {
          throw new Error(
            `요청 시간이 초과되었습니다 (${timeout / 1000}초). 스팸 분석에 시간이 오래 걸리고 있습니다. 잠시 후 다시 시도해주세요.`
          );
        }
        throw fetchError;
      }

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        const errorMsg = errorData.detail || errorData.message || "스팸 분석 중 오류가 발생했습니다.";

        // 백엔드 연결 오류
        if (response.status === 503) {
          const errorMessage: Message = {
            id: (Date.now() + 1).toString(),
            role: "assistant",
            content: `⚠️ 모델 로드 실패: ${errorMsg}`,
            timestamp: new Date(),
          };
          setMessages((prev) => [...prev, errorMessage]);
          return;
        }

        throw new Error(errorMsg);
      }

      const data = await response.json();

      // 스팸 분석 결과 포맷팅
      let resultContent = "";

      // KoELECTRA 게이트웨이 결과
      const gateResult = data.gate_result;
      resultContent += `📊 KoELECTRA 게이트웨이 결과:\n`;
      resultContent += `- 스팸 확률: ${(gateResult.spam_prob * 100).toFixed(2)}%\n`;
      resultContent += `- 정상 확률: ${(gateResult.ham_prob * 100).toFixed(2)}%\n`;
      resultContent += `- 판단: ${gateResult.label === "spam" ? "스팸" : "정상"}\n`;
      resultContent += `- 신뢰도: ${gateResult.confidence === "high" ? "높음" : gateResult.confidence === "medium" ? "중간" : "낮음"}\n`;
      resultContent += `- 처리 시간: ${gateResult.latency_ms}ms\n\n`;

      // EXAONE Reader 결과 (있는 경우)
      if (data.exaone_result) {
        resultContent += `🔍 EXAONE Reader 정밀 검사:\n${data.exaone_result}\n\n`;
      } else {
        resultContent += `ℹ️ EXAONE Reader 호출 없음 (신뢰도가 충분하여 게이트웨이 결과만 사용)\n\n`;
      }

      // 최종 결정
      resultContent += `✅ 최종 판단:\n${data.final_decision}`;

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: resultContent,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error("[ERROR] 스팸 분석 실패:", error);
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
            <h1>🔍 LangGraph Spam 판독기</h1>
            <button
              className="spam-button"
              onClick={() => router.push("/")}
              disabled={isLoading}
            >
              💬 챗봇
            </button>
          </div>
        </div>
        <p>KoELECTRA 게이트웨이 + EXAONE Reader 기반 스팸 메일 분석</p>
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
          margin-top: 0.5rem;
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
