import { ReactNode } from "react";

interface DashboardSectionProps {
  title: string;
  onViewAll?: () => void;
  children: ReactNode;
}

export default function DashboardSection({
  title,
  onViewAll,
  children,
}: DashboardSectionProps) {
  return (
    <div className="dashboard-section">
      <div className="section-header">
        <h3>{title}</h3>
        {onViewAll && (
          <button className="view-all-button" onClick={onViewAll}>
            전체 보기 →
          </button>
        )}
      </div>
      {children}
      <style jsx>{`
        .dashboard-section {
          background: #f9fafb;
          border-radius: 0.5rem;
          padding: 1.5rem;
        }

        .section-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 1rem;
        }

        .section-header h3 {
          margin: 0;
          font-size: 1.2rem;
          color: #333;
        }

        .view-all-button {
          padding: 0.5rem 1rem;
          border: 1px solid #d1d5db;
          border-radius: 0.25rem;
          background: white;
          color: #667eea;
          font-size: 0.85rem;
          cursor: pointer;
          transition: all 0.2s;
        }

        .view-all-button:hover {
          background: #667eea;
          color: white;
          border-color: #667eea;
        }
      `}</style>
    </div>
  );
}
