import { Match, BettingOdds } from "./types";

interface BettingSummaryCardProps {
  match: Match;
  odds: BettingOdds;
}

export default function BettingSummaryCard({
  match,
  odds,
}: BettingSummaryCardProps) {
  return (
    <div className="betting-summary-card">
      <h4>{match.homeTeam} vs {match.awayTeam}</h4>
      <div className="betting-odds-summary">
        <div className="odds-summary-item">
          <span>홈 승</span>
          <span className="odds-value-small">{odds.homeWin.toFixed(2)}</span>
        </div>
        <div className="odds-summary-item">
          <span>무</span>
          <span className="odds-value-small">{odds.draw.toFixed(2)}</span>
        </div>
        <div className="odds-summary-item">
          <span>원정 승</span>
          <span className="odds-value-small">{odds.awayWin.toFixed(2)}</span>
        </div>
      </div>
      <style jsx>{`
        .betting-summary-card {
          background: white;
          border: 1px solid #e5e7eb;
          border-radius: 0.5rem;
          padding: 1rem;
          transition: transform 0.2s, box-shadow 0.2s;
        }

        .betting-summary-card:hover {
          transform: translateY(-2px);
          box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        }

        .betting-summary-card h4 {
          margin: 0 0 0.75rem 0;
          font-size: 1rem;
          color: #333;
        }

        .betting-odds-summary {
          display: flex;
          gap: 1rem;
        }

        .odds-summary-item {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 0.25rem;
          flex: 1;
          padding: 0.5rem;
          background: #f9fafb;
          border-radius: 0.25rem;
        }

        .odds-summary-item span:first-child {
          font-size: 0.75rem;
          color: #6b7280;
        }

        .odds-value-small {
          font-size: 1.1rem;
          font-weight: 700;
          color: #667eea;
        }
      `}</style>
    </div>
  );
}
