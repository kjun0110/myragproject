"use client";

import { useCallback, useState } from "react";
import UploadContent from "@/components/v10/UploadContent";

export default function PlayerUploadPage() {
  const [isEmbedding, setIsEmbedding] = useState(false);
  const [embedError, setEmbedError] = useState<string | null>(null);

  const handleEmbedding = useCallback(async () => {
    setIsEmbedding(true);
    setEmbedError(null);

    try {
      const res = await fetch("/v10/api/player/embed", { method: "GET" });
      const data = await res.json().catch(() => ({}));

      if (!res.ok) {
        throw new Error(data?.detail || data?.error || "임베딩 요청 실패");
      }

      alert(data?.message || "임베딩 요청을 전송했습니다.");
    } catch (e) {
      setEmbedError(e instanceof Error ? e.message : "임베딩 요청 중 오류가 발생했습니다.");
    } finally {
      setIsEmbedding(false);
    }
  }, []);

  return (
    <UploadContent
      itemType="player"
      extraActions={
        <>
          {embedError ? <div className="embed-error">{embedError}</div> : null}
          <button
            className="embed-button"
            onClick={handleEmbedding}
            disabled={isEmbedding}
            type="button"
          >
            {isEmbedding ? "임베딩 중..." : "임베딩"}
          </button>

          <style jsx>{`
            .embed-button {
              width: 100%;
              padding: 1rem;
              background: #10b981;
              color: white;
              border: none;
              border-radius: 0.5rem;
              font-size: 1.2rem;
              font-weight: 600;
              cursor: pointer;
              transition: background 0.2s;
            }

            .embed-button:hover:not(:disabled) {
              background: #059669;
            }

            .embed-button:disabled {
              opacity: 0.6;
              cursor: not-allowed;
            }

            .embed-error {
              margin-bottom: 0.5rem;
              padding: 0.75rem;
              background: #fef2f2;
              border: 1px solid #fecaca;
              border-radius: 0.5rem;
              color: #dc2626;
              font-size: 0.95rem;
            }
          `}</style>
        </>
      }
    />
  );
}
