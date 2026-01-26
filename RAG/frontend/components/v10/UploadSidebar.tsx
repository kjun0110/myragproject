"use client";

import { useRouter, usePathname } from "next/navigation";

type ItemType = "player" | "schedule" | "stadium" | "team";

interface ItemConfig {
  label: string;
  description: string;
  icon: string;
  path: string;
}

const itemConfigs: Record<ItemType, ItemConfig> = {
  player: {
    label: "선수 (Player)",
    description: "선수 정보 JSONL 파일을 업로드합니다",
    icon: "⚽",
    path: "/v10/admin/upload/player",
  },
  schedule: {
    label: "경기 일정 (Schedule)",
    description: "경기 일정 정보 JSONL 파일을 업로드합니다",
    icon: "📅",
    path: "/v10/admin/upload/schedule",
  },
  stadium: {
    label: "경기장 (Stadium)",
    description: "경기장 정보 JSONL 파일을 업로드합니다",
    icon: "🏟️",
    path: "/v10/admin/upload/stadium",
  },
  team: {
    label: "팀 (Team)",
    description: "팀 정보 JSONL 파일을 업로드합니다",
    icon: "👥",
    path: "/v10/admin/upload/team",
  },
};

export default function UploadSidebar() {
  const router = useRouter();
  const pathname = usePathname();

  const handleItemTypeClick = (path: string) => {
    router.push(path);
  };

  return (
    <aside className="sidebar">
      <h2 className="sidebar-title">데이터 타입 선택</h2>
      <div className="sidebar-buttons">
        {(Object.keys(itemConfigs) as ItemType[]).map((itemType) => {
          const config = itemConfigs[itemType];
          const isActive = pathname === config.path;

          return (
            <button
              key={itemType}
              className={`sidebar-button ${isActive ? "active" : ""}`}
              onClick={() => handleItemTypeClick(config.path)}
            >
              <span className="button-icon">{config.icon}</span>
              <span className="button-label">{config.label}</span>
            </button>
          );
        })}
      </div>

      <style jsx>{`
        .sidebar {
          width: 280px;
          background: white;
          border-radius: 0.5rem;
          padding: 1.5rem;
          box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
          flex-shrink: 0;
        }

        .sidebar-title {
          margin: 0 0 1.5rem 0;
          font-size: 1.2rem;
          color: #333;
          font-weight: 600;
        }

        .sidebar-buttons {
          display: flex;
          flex-direction: column;
          gap: 0.75rem;
        }

        .sidebar-button {
          display: flex;
          align-items: center;
          gap: 0.75rem;
          padding: 1rem;
          background: #f9fafb;
          border: 2px solid #e5e7eb;
          border-radius: 0.5rem;
          cursor: pointer;
          transition: all 0.2s;
          text-align: left;
          width: 100%;
        }

        .sidebar-button:hover {
          background: #f3f4f6;
          border-color: #d1d5db;
        }

        .sidebar-button.active {
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          border-color: #667eea;
          color: white;
        }

        .button-icon {
          font-size: 1.5rem;
        }

        .button-label {
          font-size: 1rem;
          font-weight: 500;
        }

        @media (max-width: 1024px) {
          .sidebar {
            width: 100%;
          }

          .sidebar-buttons {
            flex-direction: row;
            flex-wrap: wrap;
          }

          .sidebar-button {
            flex: 1;
            min-width: 150px;
          }
        }
      `}</style>
    </aside>
  );
}

export { itemConfigs };
export type { ItemType };
