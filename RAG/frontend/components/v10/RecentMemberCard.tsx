import { Member } from "./types";

interface RecentMemberCardProps {
  member: Member;
  getMembershipColor: (level: string) => string;
}

export default function RecentMemberCard({
  member,
  getMembershipColor,
}: RecentMemberCardProps) {
  return (
    <div className="recent-member-card">
      <div className="member-info-row">
        <span className="member-name">{member.name}</span>
        <span
          className="membership-badge-small"
          style={{
            backgroundColor: getMembershipColor(member.membershipLevel),
          }}
        >
          {member.membershipLevel.toUpperCase()}
        </span>
      </div>
      <div className="member-details">
        <span>{member.email}</span>
        <span>총 구매액: {member.totalSpent.toLocaleString()}원</span>
      </div>
      <style jsx>{`
        .recent-member-card {
          background: white;
          border: 1px solid #e5e7eb;
          border-radius: 0.5rem;
          padding: 1rem;
          transition: transform 0.2s, box-shadow 0.2s;
        }

        .recent-member-card:hover {
          transform: translateY(-2px);
          box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        }

        .member-info-row {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 0.5rem;
        }

        .member-name {
          font-weight: 600;
          color: #333;
        }

        .membership-badge-small {
          padding: 0.2rem 0.5rem;
          border-radius: 0.25rem;
          font-size: 0.7rem;
          font-weight: 600;
          color: white;
        }

        .member-details {
          display: flex;
          flex-direction: column;
          gap: 0.25rem;
          font-size: 0.85rem;
          color: #6b7280;
        }
      `}</style>
    </div>
  );
}
