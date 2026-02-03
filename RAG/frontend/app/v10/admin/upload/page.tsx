"use client";

import Link from "next/link";

const uploadTypes = [
  { id: "player", label: "선수 (Player)", icon: "⚽", path: "/v10/admin/upload/player" },
  { id: "schedule", label: "경기 일정 (Schedule)", icon: "📅", path: "/v10/admin/upload/schedule" },
  { id: "stadium", label: "경기장 (Stadium)", icon: "🏟️", path: "/v10/admin/upload/stadium" },
  { id: "team", label: "팀 (Team)", icon: "👥", path: "/v10/admin/upload/team" },
];

export default function UploadTypeSelectPage() {
  return (
    <div className="upload-type-select">
      <div className="upload-type-intro">
        <h2 className="upload-type-title">데이터 타입 선택</h2>
        <p className="upload-type-desc">
          아래에서 업로드할 데이터 타입을 선택하세요. 선택한 페이지로 이동한 뒤, JSONL 파일을 드래그 앤 드롭하거나
          파일 선택 버튼으로 업로드할 수 있습니다. 각 타입별로 선수, 경기 일정, 경기장, 팀 정보를 JSONL 형식으로
          등록할 수 있습니다.
        </p>
      </div>
      <div className="upload-type-buttons">
        {uploadTypes.map((item) => (
          <Link
            key={item.id}
            href={item.path}
            prefetch={false}
            className="upload-type-button"
          >
            <div className="upload-type-button-inner">
              <span className="upload-type-icon">{item.icon}</span>
              <span className="upload-type-label">{item.label}</span>
            </div>
          </Link>
        ))}
      </div>
      <style jsx>{`
        .upload-type-select {
          flex: 1;
          background: transparent;
          padding: 2rem 0;
        }

        .upload-type-intro {
          margin-bottom: 2rem;
        }

        .upload-type-title {
          margin: 0 0 0.75rem 0;
          font-size: 1.5rem;
          font-weight: 700;
          color: #1e293b;
        }

        .upload-type-desc {
          margin: 0;
          font-size: 1rem;
          line-height: 1.6;
          color: #64748b;
        }

        .upload-type-buttons {
          display: grid;
          grid-template-columns: repeat(4, 1fr);
          gap: 2rem;
          align-items: stretch;
        }

        .upload-type-button {
          display: block;
          text-decoration: none;
          color: inherit;
          height: 100%;
        }

        .upload-type-button-inner {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          gap: 0.75rem;
          min-height: 180px;
          padding: 2rem 1.5rem;
          background: #ffffff;
          border: 2px solid #94a3b8;
          border-radius: 0.75rem;
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
          color: #1e293b;
          font-weight: 600;
          font-size: 1rem;
          transition: all 0.2s;
          box-sizing: border-box;
        }

        .upload-type-button:hover .upload-type-button-inner {
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          border-color: #667eea;
          color: white;
          box-shadow: 0 4px 12px rgba(102, 126, 234, 0.35);
        }

        .upload-type-icon {
          font-size: 3rem;
        }

        .upload-type-label {
          text-align: center;
          line-height: 1.3;
        }

        @media (max-width: 768px) {
          .upload-type-buttons {
            grid-template-columns: 1fr 1fr;
          }
        }

        @media (max-width: 480px) {
          .upload-type-buttons {
            grid-template-columns: 1fr;
          }
        }
      `}</style>
    </div>
  );
}
