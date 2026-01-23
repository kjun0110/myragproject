interface StatCardProps {
  icon: string;
  title: string;
  value: string | number;
}

export default function StatCard({ icon, title, value }: StatCardProps) {
  return (
    <div className="stat-card">
      <div className="stat-icon">{icon}</div>
      <div className="stat-info">
        <h3>{title}</h3>
        <p className="stat-value">{value}</p>
      </div>
      <style jsx>{`
        .stat-card {
          display: flex;
          align-items: center;
          gap: 1rem;
          padding: 1.5rem;
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          border-radius: 0.5rem;
          color: white;
          box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        }

        .stat-icon {
          font-size: 3rem;
        }

        .stat-info h3 {
          margin: 0 0 0.5rem 0;
          font-size: 0.9rem;
          opacity: 0.9;
        }

        .stat-value {
          margin: 0;
          font-size: 2rem;
          font-weight: 700;
        }
      `}</style>
    </div>
  );
}
