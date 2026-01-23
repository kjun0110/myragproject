import { Match, BettingOdds } from "./types";

interface MatchCardProps {
  match: Match;
  odds?: BettingOdds;
  selected?: boolean;
  ticketQuantity?: number;
  onSelect?: () => void;
  onPurchase?: () => void;
  onCancel?: () => void;
  onQuantityChange?: (quantity: number) => void;
}

export default function MatchCard({
  match,
  odds,
  selected = false,
  ticketQuantity = 1,
  onSelect,
  onPurchase,
  onCancel,
  onQuantityChange,
}: MatchCardProps) {
  return (
    <div className="match-card">
      <div className="match-header">
        <h3>{match.homeTeam} vs {match.awayTeam}</h3>
        <span className="venue">{match.venue}</span>
      </div>
      <div className="match-info">
        <p>📅 {match.date} {match.time}</p>
        <p>💺 남은 좌석: {match.availableSeats}석</p>
        <p>💰 가격: {match.price.toLocaleString()}원</p>
      </div>
      {odds && (
        <div className="odds-preview">
          <p>승률 추론:</p>
          <div className="odds-bar">
            <div
              className="odds-segment home"
              style={{ width: `${odds.homeWinProbability}%` }}
            >
              홈 {odds.homeWinProbability}%
            </div>
            <div
              className="odds-segment draw"
              style={{ width: `${odds.drawProbability}%` }}
            >
              무 {odds.drawProbability}%
            </div>
            <div
              className="odds-segment away"
              style={{ width: `${odds.awayWinProbability}%` }}
            >
              원정 {odds.awayWinProbability}%
            </div>
          </div>
        </div>
      )}
      <div className="match-actions">
        {selected ? (
          <div className="purchase-form">
            <label>
              수량:
              <input
                type="number"
                min="1"
                max={match.availableSeats}
                value={ticketQuantity}
                onChange={(e) =>
                  onQuantityChange?.(parseInt(e.target.value) || 1)
                }
              />
            </label>
            <div className="button-group">
              <button
                className="purchase-button"
                onClick={onPurchase}
              >
                예매하기 ({(match.price * ticketQuantity).toLocaleString()}원)
              </button>
              <button
                className="cancel-button"
                onClick={onCancel}
              >
                취소
              </button>
            </div>
          </div>
        ) : (
          <button
            className="select-button"
            onClick={onSelect}
          >
            예매하기
          </button>
        )}
      </div>
      <style jsx>{`
        .match-card {
          border: 1px solid #e5e7eb;
          border-radius: 0.5rem;
          padding: 1.5rem;
          background: white;
          box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
          transition: transform 0.2s, box-shadow 0.2s;
        }

        .match-card:hover {
          transform: translateY(-2px);
          box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        }

        .match-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 1rem;
        }

        .match-header h3 {
          margin: 0;
          font-size: 1.2rem;
          color: #333;
        }

        .venue {
          font-size: 0.85rem;
          color: #6b7280;
        }

        .match-info p {
          margin: 0.5rem 0;
          color: #6b7280;
        }

        .odds-preview {
          margin: 1rem 0;
          padding: 1rem;
          background: #f9fafb;
          border-radius: 0.5rem;
        }

        .odds-bar {
          display: flex;
          height: 30px;
          border-radius: 0.25rem;
          overflow: hidden;
          margin-top: 0.5rem;
        }

        .odds-segment {
          display: flex;
          align-items: center;
          justify-content: center;
          color: white;
          font-size: 0.75rem;
          font-weight: 600;
        }

        .odds-segment.home {
          background: #3b82f6;
        }

        .odds-segment.draw {
          background: #6b7280;
        }

        .odds-segment.away {
          background: #ef4444;
        }

        .match-actions {
          margin-top: 1rem;
        }

        .select-button,
        .purchase-button {
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

        .select-button:hover,
        .purchase-button:hover {
          background: #5568d3;
        }

        .purchase-form {
          display: flex;
          flex-direction: column;
          gap: 1rem;
        }

        .purchase-form label {
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }

        .purchase-form input {
          flex: 1;
          padding: 0.5rem;
          border: 1px solid #d1d5db;
          border-radius: 0.25rem;
        }

        .button-group {
          display: flex;
          gap: 0.5rem;
        }

        .cancel-button {
          padding: 0.75rem;
          border: 1px solid #d1d5db;
          border-radius: 0.5rem;
          background: white;
          color: #6b7280;
          cursor: pointer;
          transition: background 0.2s;
        }

        .cancel-button:hover {
          background: #f9fafb;
        }
      `}</style>
    </div>
  );
}
