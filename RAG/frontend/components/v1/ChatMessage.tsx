"use client";

import { useState, useEffect } from "react";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
}

interface ChatMessageProps {
  message: Message;
}

export default function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === "user";
  const [timeString, setTimeString] = useState<string>("");

  // 클라이언트에서만 시간 포맷팅 (hydration mismatch 방지)
  useEffect(() => {
    setTimeString(
      message.timestamp.toLocaleTimeString("ko-KR", {
        hour: "2-digit",
        minute: "2-digit",
      })
    );
  }, [message.timestamp]);

  // 간단한 마크다운 스타일 처리 (코드 블록, 링크 등)
  const formatContent = (content: string) => {
    // 코드 블록 처리
    content = content.replace(/```(\w+)?\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>');
    // 인라인 코드 처리
    content = content.replace(/`([^`]+)`/g, '<code>$1</code>');
    // 링크 처리
    content = content.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');
    // 볼드 처리
    content = content.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    return content;
  };

  return (
    <div className={`message ${isUser ? "user" : "assistant"}`}>
      <div className="message-content">
        <div
          className="message-text"
          dangerouslySetInnerHTML={{ __html: formatContent(message.content) }}
        />
        <div className="message-time" suppressHydrationWarning>
          {timeString || ""}
        </div>
      </div>

      <style jsx>{`
        .message {
          display: flex;
          margin-bottom: 1rem;
          animation: fadeIn 0.3s ease-in;
        }

        .message.user {
          justify-content: flex-end;
        }

        .message.assistant {
          justify-content: flex-start;
        }

        .message-content {
          max-width: 70%;
          padding: 0.75rem 1rem;
          border-radius: 1.5rem;
          word-wrap: break-word;
        }

        .message.user .message-content {
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
          border-bottom-right-radius: 0.25rem;
        }

        .message.assistant .message-content {
          background: white;
          color: #333;
          box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
          border-bottom-left-radius: 0.25rem;
        }

        .message-text {
          line-height: 1.5;
          white-space: pre-wrap;
        }

        .message-text :global(code) {
          background: rgba(0, 0, 0, 0.05);
          padding: 0.2rem 0.4rem;
          border-radius: 0.25rem;
          font-family: monospace;
          font-size: 0.9em;
        }

        .message-text :global(pre) {
          background: rgba(0, 0, 0, 0.05);
          padding: 1rem;
          border-radius: 0.5rem;
          overflow-x: auto;
          margin: 0.5rem 0;
        }

        .message-text :global(pre code) {
          background: none;
          padding: 0;
        }

        .message-text :global(a) {
          color: #667eea;
          text-decoration: underline;
        }

        .message-text :global(strong) {
          font-weight: 600;
        }

        .message-time {
          font-size: 0.75rem;
          opacity: 0.7;
          margin-top: 0.25rem;
          text-align: right;
        }

        .message.assistant .message-time {
          text-align: left;
        }

        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        @media (max-width: 768px) {
          .message-content {
            max-width: 85%;
          }
        }
      `}</style>
    </div>
  );
}
