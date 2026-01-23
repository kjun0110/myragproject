import { Match, BettingOdds } from "./types";

interface RecentMatchCardProps {
  match: Match;
  odds?: BettingOdds;
}

export default function RecentMatchCard({ match, odds }: RecentMatchCardProps) {
  return (
    <div className="recent-match-card">
      <div className="recent-match-header">
        <h4>{match.homeTeam} vs {match.awayTeam}</h4>
        <span className="match-date">{match.date} {match.time}</span>
      </div>
      <div className="recent-match-info">
        <span>💺 {match.availableSeats}석 남음</span>
        <span>💰 {match.price.toLocaleString()}원</span>
      </div>
      {odds && (
        <div className="recent-odds">
          <span>홈 {odds.homeWinProbability}%</span>
          <span>무 {odds.drawProbability}%</span>
          <span>원정 {odds.awayWinProbability}%</span>
        </div>
      )}
      <style jsx>{`
        .recent-match-card {
          background: white;
          border: 1px solid #e5e7eb;
          border-radius: 0.5rem;
          padding: 1rem;
          transition: transform 0.2s, box-shadow 0.2s;
        }

        .recent-match-card:hover {
          transform: translateY(-2px);
          box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        }

        .recent-match-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 0.5rem;
        }

        .recent-match-header h4 {
          margin: 0;
          font-size: 1rem;
          color: #333;
        }

        .match-date {
          font-size: 0.85rem;
          color: #6b7280;
        }

        .recent-match-info {
          display: flex;
          gap: 1rem;
          margin-bottom: 0.5rem;
          font-size: 0.85rem;
          color: #6b7280;
        }

        .recent-odds {
          display: flex;
          gap: 0.5rem;
          font-size: 0.8rem;
          color: #6b7280;
        }

        .recent-odds span {
          padding: 0.25rem 0.5rem;
          background: #f3f4f6;
          border-radius: 0.25rem;
        }
      `}</style>
    </div>
  );
}
