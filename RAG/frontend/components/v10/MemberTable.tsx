import { Member } from "./types";

interface MemberTableProps {
  members: Member[];
  getMembershipColor: (level: string) => string;
  onEdit?: (member: Member) => void;
  onDelete?: (member: Member) => void;
}

export default function MemberTable({
  members,
  getMembershipColor,
  onEdit,
  onDelete,
}: MemberTableProps) {
  return (
    <div className="members-table">
      <table>
        <thead>
          <tr>
            <th>이름</th>
            <th>이메일</th>
            <th>전화번호</th>
            <th>멤버십</th>
            <th>가입일</th>
            <th>총 구매액</th>
            <th>액션</th>
          </tr>
        </thead>
        <tbody>
          {members.map((member) => (
            <tr key={member.id}>
              <td>{member.name}</td>
              <td>{member.email}</td>
              <td>{member.phone}</td>
              <td>
                <span
                  className="membership-badge"
                  style={{
                    backgroundColor: getMembershipColor(member.membershipLevel),
                  }}
                >
                  {member.membershipLevel.toUpperCase()}
                </span>
              </td>
              <td>{member.joinDate}</td>
              <td>{member.totalSpent.toLocaleString()}원</td>
              <td>
                <button
                  className="edit-button"
                  onClick={() => onEdit?.(member)}
                >
                  수정
                </button>
                <button
                  className="delete-button"
                  onClick={() => onDelete?.(member)}
                >
                  삭제
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <style jsx>{`
        .members-table {
          overflow-x: auto;
        }

        .members-table table {
          width: 100%;
          border-collapse: collapse;
        }

        .members-table th,
        .members-table td {
          padding: 1rem;
          text-align: left;
          border-bottom: 1px solid #e5e7eb;
        }

        .members-table th {
          background: #f9fafb;
          font-weight: 600;
          color: #374151;
        }

        .membership-badge {
          padding: 0.25rem 0.75rem;
          border-radius: 0.25rem;
          font-size: 0.75rem;
          font-weight: 600;
          color: white;
        }

        .edit-button,
        .delete-button {
          padding: 0.5rem;
          border: 1px solid #d1d5db;
          border-radius: 0.25rem;
          background: white;
          cursor: pointer;
          transition: background 0.2s;
          margin-right: 0.5rem;
        }

        .edit-button {
          color: #3b82f6;
        }

        .edit-button:hover {
          background: #eff6ff;
        }

        .delete-button {
          color: #ef4444;
        }

        .delete-button:hover {
          background: #fef2f2;
        }
      `}</style>
    </div>
  );
}
