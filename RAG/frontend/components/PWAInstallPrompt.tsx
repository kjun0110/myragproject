"use client";

import { useState, useEffect } from "react";

interface BeforeInstallPromptEvent extends Event {
  prompt: () => Promise<void>;
  userChoice: Promise<{ outcome: "accepted" | "dismissed" }>;
}

export default function PWAInstallPrompt() {
  const [deferredPrompt, setDeferredPrompt] =
    useState<BeforeInstallPromptEvent | null>(null);
  const [showPrompt, setShowPrompt] = useState(false);

  useEffect(() => {
    const handler = (e: Event) => {
      e.preventDefault();
      setDeferredPrompt(e as BeforeInstallPromptEvent);
      setShowPrompt(true);
    };

    window.addEventListener("beforeinstallprompt", handler);

    return () => {
      window.removeEventListener("beforeinstallprompt", handler);
    };
  }, []);

  const handleInstall = async () => {
    if (!deferredPrompt) return;

    deferredPrompt.prompt();
    const { outcome } = await deferredPrompt.userChoice;

    if (outcome === "accepted") {
      console.log("PWA 설치 승인됨");
    } else {
      console.log("PWA 설치 거부됨");
    }

    setDeferredPrompt(null);
    setShowPrompt(false);
  };

  const handleDismiss = () => {
    setShowPrompt(false);
  };

  if (!showPrompt || !deferredPrompt) return null;

  return (
    <div className="pwa-prompt">
      <div className="pwa-prompt-content">
        <div className="pwa-prompt-icon">📱</div>
        <div className="pwa-prompt-text">
          <h3>앱 설치</h3>
          <p>홈 화면에 추가하여 더 빠르게 접근하세요!</p>
        </div>
        <div className="pwa-prompt-actions">
          <button className="install-button" onClick={handleInstall}>
            설치
          </button>
          <button className="dismiss-button" onClick={handleDismiss}>
            나중에
          </button>
        </div>
      </div>
      <style jsx>{`
        .pwa-prompt {
          position: fixed;
          bottom: 1rem;
          left: 50%;
          transform: translateX(-50%);
          z-index: 1000;
          max-width: 400px;
          width: calc(100% - 2rem);
          animation: slideUp 0.3s ease-out;
        }

        @keyframes slideUp {
          from {
            transform: translateX(-50%) translateY(100%);
            opacity: 0;
          }
          to {
            transform: translateX(-50%) translateY(0);
            opacity: 1;
          }
        }

        .pwa-prompt-content {
          background: white;
          border-radius: 1rem;
          padding: 1.5rem;
          box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
          display: flex;
          align-items: center;
          gap: 1rem;
        }

        .pwa-prompt-icon {
          font-size: 2.5rem;
          flex-shrink: 0;
        }

        .pwa-prompt-text {
          flex: 1;
        }

        .pwa-prompt-text h3 {
          margin: 0 0 0.25rem 0;
          font-size: 1rem;
          color: #333;
        }

        .pwa-prompt-text p {
          margin: 0;
          font-size: 0.85rem;
          color: #6b7280;
        }

        .pwa-prompt-actions {
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
        }

        .install-button,
        .dismiss-button {
          padding: 0.5rem 1rem;
          border: none;
          border-radius: 0.5rem;
          font-size: 0.9rem;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.2s;
          white-space: nowrap;
        }

        .install-button {
          background: #667eea;
          color: white;
        }

        .install-button:hover {
          background: #5568d3;
        }

        .dismiss-button {
          background: #f3f4f6;
          color: #6b7280;
        }

        .dismiss-button:hover {
          background: #e5e7eb;
        }

        @media (max-width: 768px) {
          .pwa-prompt-content {
            flex-direction: column;
            text-align: center;
          }

          .pwa-prompt-actions {
            flex-direction: row;
            width: 100%;
          }

          .install-button,
          .dismiss-button {
            flex: 1;
          }
        }
      `}</style>
    </div>
  );
}
