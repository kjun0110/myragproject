"use client";

export default function OfflinePage() {
  return (
    <div className="offline-container">
      <div className="offline-content">
        <div className="offline-icon">📡</div>
        <h1>오프라인 모드</h1>
        <p>인터넷 연결이 없습니다.</p>
        <p className="offline-description">
          이전에 방문한 페이지는 오프라인에서도 사용할 수 있습니다.
        </p>
        <button
          className="retry-button"
          onClick={() => window.location.reload()}
        >
          다시 시도
        </button>
        <button
          className="home-button"
          onClick={() => (window.location.href = "/")}
        >
          홈으로 가기
        </button>
      </div>
      <style jsx>{`
        .offline-container {
          display: flex;
          align-items: center;
          justify-content: center;
          min-height: 100vh;
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          padding: 2rem;
        }

        .offline-content {
          background: white;
          border-radius: 1rem;
          padding: 3rem;
          text-align: center;
          box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
          max-width: 500px;
          width: 100%;
        }

        .offline-icon {
          font-size: 5rem;
          margin-bottom: 1.5rem;
        }

        .offline-content h1 {
          margin: 0 0 1rem 0;
          font-size: 2rem;
          color: #333;
        }

        .offline-content p {
          margin: 0.5rem 0;
          color: #6b7280;
          font-size: 1rem;
        }

        .offline-description {
          margin: 1.5rem 0 2rem 0;
          font-size: 0.9rem;
          color: #9ca3af;
        }

        .retry-button,
        .home-button {
          padding: 0.75rem 2rem;
          border: none;
          border-radius: 0.5rem;
          font-size: 1rem;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.2s;
          margin: 0.5rem;
        }

        .retry-button {
          background: #667eea;
          color: white;
        }

        .retry-button:hover {
          background: #5568d3;
          transform: translateY(-2px);
        }

        .home-button {
          background: #f3f4f6;
          color: #374151;
        }

        .home-button:hover {
          background: #e5e7eb;
          transform: translateY(-2px);
        }

        @media (max-width: 768px) {
          .offline-content {
            padding: 2rem;
          }

          .offline-content h1 {
            font-size: 1.5rem;
          }

          .offline-icon {
            font-size: 4rem;
          }
        }
      `}</style>
    </div>
  );
}
