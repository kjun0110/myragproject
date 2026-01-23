import { Match, BettingOdds } from "./types";

interface BettingCardProps {
  match: Match;
  odds: BettingOdds;
  onBet?: () => void;
}

export default function BettingCard({ match, odds, onBet }: BettingCardProps) {
  return (
    <div className="betting-card">
      <div className="betting-header">
        <h3>{match.homeTeam} vs {match.awayTeam}</h3>
        <p>{match.date} {match.time}</p>
      </div>
      <div className="betting-odds">
        <div className="odds-item">
          <span className="team">{match.homeTeam} 승</span>
          <span className="odds-value">{odds.homeWin.toFixed(2)}</span>
          <div className="probability-bar">
            <div
              className="probability-fill home"
              style={{ width: `${odds.homeWinProbability}%` }}
            />
          </div>
          <span className="probability">{odds.homeWinProbability}%</span>
        </div>
        <div className="odds-item">
          <span className="team">무승부</span>
          <span className="odds-value">{odds.draw.toFixed(2)}</span>
          <div className="probability-bar">
            <div
              className="probability-fill draw"
              style={{ width: `${odds.drawProbability}%` }}
            />
          </div>
          <span className="probability">{odds.drawProbability}%</span>
        </div>
        <div className="odds-item">
          <span className="team">{match.awayTeam} 승</span>
          <span className="odds-value">{odds.awayWin.toFixed(2)}</span>
          <div className="probability-bar">
            <div
              className="probability-fill away"
              style={{ width: `${odds.awayWinProbability}%` }}
            />
          </div>
          <span className="probability">{odds.awayWinProbability}%</span>
        </div>
      </div>
      <div className="betting-actions">
        <button className="bet-button" onClick={onBet}>배팅하기</button>
      </div>
      <style jsx>{`
        .betting-card {
          border: 1px solid #e5e7eb;
          border-radius: 0.5rem;
          padding: 1.5rem;
          background: white;
          box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
          transition: transform 0.2s, box-shadow 0.2s;
        }

        .betting-card:hover {
          transform: translateY(-2px);
          box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        }

        .betting-header {
          margin-bottom: 1rem;
        }

        .betting-header h3 {
          margin: 0 0 0.5rem 0;
          font-size: 1.2rem;
        }

        .betting-header p {
          margin: 0;
          color: #6b7280;
          font-size: 0.9rem;
        }

        .betting-odds {
          display: flex;
          flex-direction: column;
          gap: 1rem;
        }

        .odds-item {
          display: grid;
          grid-template-columns: 1fr auto auto;
          gap: 0.5rem;
          align-items: center;
        }

        .team {
          font-weight: 600;
        }

        .odds-value {
          font-size: 1.2rem;
          font-weight: 700;
          color: #667eea;
        }

        .probability-bar {
          width: 100%;
          height: 8px;
          background: #e5e7eb;
          border-radius: 4px;
          overflow: hidden;
          grid-column: 1 / -1;
        }

        .probability-fill {
          height: 100%;
          transition: width 0.3s;
        }

        .probability-fill.home {
          background: #3b82f6;
        }

        .probability-fill.draw {
          background: #6b7280;
        }

        .probability-fill.away {
          background: #ef4444;
        }

        .probability {
          font-size: 0.85rem;
          color: #6b7280;
        }

        .betting-actions {
          margin-top: 1rem;
        }

        .bet-button {
          width: 100%;
          padding: 0.75rem;
          border: none;
          border-radius: 0.5rem;
          background: #667eea;
          color: white;
          font-size: 1rem;
          font-weight: 600;
          cursor: pointer;
          transition: background 0.2s;
        }

        .bet-button:hover {
          background: #5568d3;
        }
      `}</style>
    </div>
  );
}
